use std::{
    env,
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use rand::RngExt;

mod cube_env;

use tch::{
    Device, Kind, TchError, Tensor,
    nn::{self, Module, OptimizerConfig},
};
use tensorboard_rs::summary_writer::SummaryWriter;

use crate::cube_env::{CubeEnv, ReplayBuffer, Transition};

// TODO: Train from checkpoints
// TODO: Seeding for reproducibility
// TODO: README file and TODO file

const CUBE_SIZE: usize = 2;
const INPUT_SIZE: usize = 6 * CUBE_SIZE * CUBE_SIZE * 6;
const OUTPUT_SIZE: usize = 6 * 3;

struct TrainingConfig {
    episodes: usize,
    batch_size: usize,
    buffer_size: usize,
    learning_rate: f64,
    alpha_lr: f64,
    tau: f64,
    gamma: f64,
    max_steps_cap: usize,
    min_steps: usize,
    max_scramble: usize,
    curriculum_threshold: usize,
    curriculum_min_episodes: usize,
    target_entropy_scale: f64,
    log_alpha_init: f64,
    bootstrap_truncations: bool,
    clear_replay_on_advance: bool,
    update_every: usize,
    target_network_frequency: usize,
    adam_eps: f64,
    learning_starts: usize,
    num_envs: usize,
    eval_every: usize,
    eval_episodes: usize,
    log_every: usize,
    save_every: usize,
    run_name: String,
    log_dir: PathBuf,
    net_dir: PathBuf,
}

impl TrainingConfig {
    fn from_env() -> Self {
        let run_name = env::var("RL_RUN_NAME").unwrap_or_else(|_| {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock is before unix epoch")
                .as_secs();
            format!("run-{now}")
        });
        let log_root = env::var("RL_LOG_ROOT").unwrap_or_else(|_| "./rl/logs".to_string());
        let net_root = env::var("RL_NET_ROOT").unwrap_or_else(|_| "./rl/nets".to_string());
        let min_steps = env_parse_min("RL_MIN_STEPS", 3, 1);
        let max_steps_cap = env_parse_min("RL_MAX_STEPS_CAP", 40, min_steps);

        Self {
            episodes: env_parse_min("RL_EPISODES", 50_000, 1),
            batch_size: env_parse_min("RL_BATCH_SIZE", 256, 1),
            buffer_size: env_parse_min("RL_BUFFER_SIZE", 200_000, 1),
            learning_rate: env_parse("RL_LEARNING_RATE", 3e-4),
            alpha_lr: env_parse("RL_ALPHA_LR", 3e-4),
            tau: env_parse("RL_TAU", 1.0),
            gamma: env_parse("RL_GAMMA", 0.99),
            max_steps_cap,
            min_steps,
            max_scramble: env_parse_min("RL_MAX_SCRAMBLE", 20, 1),
            curriculum_threshold: env_parse_clamped("RL_CURRICULUM_THRESHOLD", 10, 1, 100),
            curriculum_min_episodes: env_parse_min("RL_CURRICULUM_MIN_EPISODES", 1, 1),
            target_entropy_scale: env_parse("RL_TARGET_ENTROPY_SCALE", 0.2),
            log_alpha_init: env_parse("RL_LOG_ALPHA_INIT", -2.0),
            bootstrap_truncations: env_parse_bool("RL_BOOTSTRAP_TRUNCATIONS", true),
            clear_replay_on_advance: env_parse_bool("RL_CLEAR_REPLAY_ON_ADVANCE", false),
            update_every: env_parse_min("RL_UPDATE_EVERY", 4, 1),
            target_network_frequency: env_parse_min("RL_TARGET_NETWORK_FREQUENCY", 8_000, 1),
            adam_eps: env_parse("RL_ADAM_EPS", 1e-4),
            learning_starts: env_parse("RL_LEARNING_STARTS", 5_000),
            num_envs: env_parse_min("RL_NUM_ENVS", 16, 1),
            eval_every: env_parse("RL_EVAL_EVERY", 0),
            eval_episodes: env_parse_min("RL_EVAL_EPISODES", 64, 1),
            log_every: env_parse_min("RL_LOG_EVERY", 1, 1),
            save_every: env_parse_min("RL_SAVE_EVERY", 100, 1),
            log_dir: PathBuf::from(log_root).join(&run_name),
            net_dir: PathBuf::from(net_root).join(&run_name),
            run_name,
        }
    }

    fn max_steps(&self, scramble_depth: usize) -> usize {
        (scramble_depth * 3).clamp(self.min_steps, self.max_steps_cap)
    }
}

fn env_parse<T>(key: &str, default: T) -> T
where
    T: std::str::FromStr,
{
    env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_parse_min<T>(key: &str, default: T, min: T) -> T
where
    T: std::str::FromStr + Ord,
{
    env_parse(key, default).max(min)
}

fn env_parse_clamped<T>(key: &str, default: T, min: T, max: T) -> T
where
    T: std::str::FromStr + Ord,
{
    env_parse(key, default).clamp(min, max)
}

fn env_parse_bool(key: &str, default: bool) -> bool {
    env::var(key)
        .ok()
        .and_then(|value| match value.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
        .unwrap_or(default)
}

struct EpisodeState {
    env: CubeEnv,
    state: Tensor,
    reward: f32,
    solved: bool,
    truncated: bool,
    max_solved_faces: usize,
}

impl EpisodeState {
    fn new(scramble_depth: usize, max_steps: usize) -> Self {
        let mut env = CubeEnv::new();
        let state = env.reset(scramble_depth, max_steps);
        let max_solved_faces = env.get_cube().count_solved_faces();

        Self {
            env,
            state,
            reward: 0.,
            solved: false,
            truncated: false,
            max_solved_faces,
        }
    }

    fn reset(&mut self, scramble_depth: usize, max_steps: usize) {
        self.state = self.env.reset(scramble_depth, max_steps);
        self.reward = 0.;
        self.solved = false;
        self.truncated = false;
        self.max_solved_faces = self.env.get_cube().count_solved_faces();
    }
}

#[derive(Default)]
struct UpdateMetricTotals {
    actor_loss: f32,
    critic_loss: f32,
    alpha_loss: f32,
    entropy: f32,
    entropy_error: f32,
    policy_max_prob: f32,
    target_q: f32,
    q1: f32,
    q2: f32,
    replay_truncation: f32,
    steps: usize,
}

impl UpdateMetricTotals {
    fn add(&mut self, metrics: UpdateMetrics) {
        self.actor_loss += metrics.actor_loss;
        self.critic_loss += metrics.critic_loss;
        self.alpha_loss += metrics.alpha_loss;
        self.entropy += metrics.entropy;
        self.entropy_error += metrics.entropy_error;
        self.policy_max_prob += metrics.policy_max_prob;
        self.target_q += metrics.target_q;
        self.q1 += metrics.q1;
        self.q2 += metrics.q2;
        self.replay_truncation += metrics.replay_truncation;
        self.steps += 1;
    }

    fn average(&self, value: f32) -> f32 {
        value / self.steps as f32
    }
}

struct UpdateMetrics {
    actor_loss: f32,
    critic_loss: f32,
    alpha_loss: f32,
    entropy: f32,
    entropy_error: f32,
    policy_max_prob: f32,
    target_q: f32,
    q1: f32,
    q2: f32,
    replay_truncation: f32,
}

fn main() {
    let _ = train_vectorized();
}

pub fn get_device() -> Device {
    Device::cuda_if_available()
}

fn adam(config: &TrainingConfig) -> nn::Adam {
    nn::Adam::default().eps(config.adam_eps)
}

#[allow(clippy::too_many_arguments)]
fn sac_update(
    config: &TrainingConfig,
    target_entropy: f64,
    replay_buffer: &ReplayBuffer,
    actor: &ResidualNetwork,
    critic1: &ResidualNetwork,
    critic2: &ResidualNetwork,
    target_critic1: &ResidualNetwork,
    target_critic2: &ResidualNetwork,
    actor_opt: &mut nn::Optimizer,
    critic1_opt: &mut nn::Optimizer,
    critic2_opt: &mut nn::Optimizer,
    log_alpha: &Tensor,
    alpha_opt: &mut nn::Optimizer,
) -> UpdateMetrics {
    let batch = replay_buffer.sample_tensors(config.batch_size);
    let states = batch.states;
    let next_states = batch.next_states;
    let actions = batch.actions;
    let rewards = batch.rewards;
    let terminated = batch.terminated;
    let truncated = batch.truncated;

    let target = tch::no_grad(|| {
        let next_logits = actor.forward(&next_states);
        let next_probs = next_logits.softmax(-1, Kind::Float);
        let next_log_probs = next_logits.log_softmax(-1, Kind::Float);

        let next_q1 = target_critic1.forward(&next_states);
        let next_q2 = target_critic2.forward(&next_states);
        let next_min_q = next_q1.minimum(&next_q2);

        let next_v = (&next_probs * (&next_min_q - log_alpha.exp() * &next_log_probs))
            .sum_dim_intlist(&[1i64][..], false, Kind::Float);

        let bootstrap = if config.bootstrap_truncations {
            1. - &terminated
        } else {
            1. - terminated.maximum(&truncated)
        };

        &rewards + config.gamma * &next_v * bootstrap
    });

    let q1 = critic1
        .forward(&states)
        .gather(1, &actions.unsqueeze(1), false)
        .squeeze_dim(1);
    let critic1_loss = q1.mse_loss(&target, tch::Reduction::Mean);
    critic1_opt.zero_grad();
    critic1_loss.backward();
    critic1_opt.step();

    let q2 = critic2
        .forward(&states)
        .gather(1, &actions.unsqueeze(1), false)
        .squeeze_dim(1);
    let critic2_loss = q2.mse_loss(&target, tch::Reduction::Mean);
    critic2_opt.zero_grad();
    critic2_loss.backward();
    critic2_opt.step();

    let logits = actor.forward(&states);
    let probs = logits.softmax(-1, Kind::Float);
    let log_probs = logits.log_softmax(-1, Kind::Float);

    let q1_detached = tch::no_grad(|| critic1.forward(&states));
    let q2_detached = tch::no_grad(|| critic2.forward(&states));
    let q_min = q1_detached.minimum(&q2_detached);

    let alpha = log_alpha.exp();
    let actor_loss = (&probs * (&alpha * &log_probs - &q_min))
        .sum_dim_intlist(&[1i64][..], false, Kind::Float)
        .mean(Kind::Float);
    actor_opt.zero_grad();
    actor_loss.backward();
    actor_opt.step();

    let entropy = tch::no_grad(|| {
        let logits = actor.forward(&states);
        let probs = logits.softmax(-1, Kind::Float);
        let log_probs = logits.log_softmax(-1, Kind::Float);
        -(&probs * &log_probs)
            .sum_dim_intlist(&[1i64][..], false, Kind::Float)
            .mean(Kind::Float)
    });
    let entropy_error = &entropy - target_entropy;
    let alpha_loss = (&probs.detach() * (-log_alpha * (&log_probs + target_entropy).detach()))
        .sum_dim_intlist(&[1i64][..], false, Kind::Float)
        .mean(Kind::Float);

    alpha_opt.zero_grad();
    alpha_loss.backward();
    alpha_opt.step();

    UpdateMetrics {
        actor_loss: f32::try_from(&actor_loss).expect("loss calculation failed"),
        critic_loss: f32::try_from(&critic1_loss).expect("loss calculation failed")
            + f32::try_from(&critic2_loss).expect("loss calculation failed"),
        alpha_loss: f32::try_from(&alpha_loss).expect("loss calculation failed"),
        entropy: f32::try_from(&entropy).expect("entropy calculation failed"),
        entropy_error: f32::try_from(&entropy_error).expect("entropy calculation failed"),
        policy_max_prob: f32::try_from(&probs.max_dim(1, false).0.mean(Kind::Float))
            .expect("policy max probability calculation failed"),
        target_q: f32::try_from(&target.mean(Kind::Float)).expect("target calculation failed"),
        q1: f32::try_from(&q1.mean(Kind::Float)).expect("q1 calculation failed"),
        q2: f32::try_from(&q2.mean(Kind::Float)).expect("q2 calculation failed"),
        replay_truncation: f32::try_from(&truncated.mean(Kind::Float))
            .expect("truncation calculation failed"),
    }
}

fn update_target_networks(
    config: &TrainingConfig,
    critic1_vs: &nn::VarStore,
    critic2_vs: &nn::VarStore,
    target_critic1_vs: &mut nn::VarStore,
    target_critic2_vs: &mut nn::VarStore,
) {
    tch::no_grad(|| {
        for (tp, pp) in target_critic1_vs
            .trainable_variables()
            .iter_mut()
            .zip(critic1_vs.trainable_variables().iter())
        {
            let updated = pp * config.tau + &*tp * (1. - config.tau);
            tp.copy_(&updated);
        }
        for (tp, pp) in target_critic2_vs
            .trainable_variables()
            .iter_mut()
            .zip(critic2_vs.trainable_variables().iter())
        {
            let updated = pp * config.tau + &*tp * (1. - config.tau);
            tp.copy_(&updated);
        }
    });
}

struct EvalMetrics {
    solve_rate: f32,
    average_reward: f32,
    average_steps: f32,
}

fn evaluate_greedy(
    actor: &ResidualNetwork,
    scramble_depth: usize,
    max_steps: usize,
    episodes: usize,
) -> EvalMetrics {
    let mut env = CubeEnv::new();
    let mut solves = 0usize;
    let mut total_reward = 0f32;
    let mut total_steps = 0usize;

    for _ in 0..episodes {
        let mut state = env.reset(scramble_depth, max_steps);

        for step_idx in 1..=max_steps {
            let action = tch::no_grad(|| {
                actor
                    .forward(&state.unsqueeze(0))
                    .argmax(1, false)
                    .int64_value(&[0]) as usize
            });
            let step = env.step(action);
            total_reward += step.reward;
            total_steps += 1;
            state = step.next_state;

            if step.terminated {
                solves += 1;
                break;
            }
            if step.truncated {
                break;
            }

            if step_idx == max_steps {
                break;
            }
        }
    }

    EvalMetrics {
        solve_rate: solves as f32 / episodes as f32,
        average_reward: total_reward / episodes as f32,
        average_steps: total_steps as f32 / episodes as f32,
    }
}

fn train_vectorized() -> Result<(), TchError> {
    let config = TrainingConfig::from_env();
    std::fs::create_dir_all(&config.net_dir).expect("failed to create net directory");
    std::fs::create_dir_all(&config.log_dir).expect("failed to create log directory");

    let alpha_vs = nn::VarStore::new(get_device());
    let log_alpha = alpha_vs
        .root()
        .var("log_alpha", &[], nn::Init::Const(config.log_alpha_init));
    let target_entropy = config.target_entropy_scale * (OUTPUT_SIZE as f64).ln();
    let mut alpha_opt = adam(&config).build(&alpha_vs, config.alpha_lr)?;

    let actor_vs = nn::VarStore::new(get_device());
    let critic1_vs = nn::VarStore::new(get_device());
    let critic2_vs = nn::VarStore::new(get_device());
    let mut target_critic1_vs = nn::VarStore::new(get_device());
    let mut target_critic2_vs = nn::VarStore::new(get_device());

    let actor = initialize_network(&actor_vs.root());
    let critic1 = initialize_network(&critic1_vs.root());
    let critic2 = initialize_network(&critic2_vs.root());
    let target_critic1 = initialize_network(&target_critic1_vs.root());
    let target_critic2 = initialize_network(&target_critic2_vs.root());

    target_critic1_vs.copy(&critic1_vs)?;
    target_critic2_vs.copy(&critic2_vs)?;

    let mut actor_opt = adam(&config).build(&actor_vs, config.learning_rate)?;
    let mut critic1_opt = adam(&config).build(&critic1_vs, config.learning_rate)?;
    let mut critic2_opt = adam(&config).build(&critic2_vs, config.learning_rate)?;

    let mut replay_buffer = ReplayBuffer::new(config.buffer_size);
    let mut last_100_solves = [false; 100];
    let mut scramble_depth = 1usize;
    let mut episodes_at_depth = 0usize;
    let mut completed_episodes = 0usize;
    let mut env_steps = 0usize;
    let mut learner_updates = 0usize;
    let mut curriculum_active = config.learning_starts == 0;
    let mut update_metrics = UpdateMetricTotals::default();

    let max_steps = config.max_steps(scramble_depth);
    let mut envs = (0..config.num_envs)
        .map(|_| EpisodeState::new(scramble_depth, max_steps))
        .collect::<Vec<_>>();

    println!(
        "Beginning training run {} for cube of size: {} on device: {:?}",
        config.run_name,
        CUBE_SIZE,
        get_device()
    );
    println!(
        "logs: {} | nets: {} | target entropy: {:.4} | target entropy scale: {:.3} | log alpha init: {:.3} | alpha lr: {:.1e} | adam eps: {:.1e} | tau: {:.4} | bootstrap truncations: {} | update every: {} | target sync: {} | learning starts: {} | envs: {} | curriculum threshold: {}%/{}eps | clear replay: {} | eval: {}x{}",
        config.log_dir.display(),
        config.net_dir.display(),
        target_entropy,
        config.target_entropy_scale,
        config.log_alpha_init,
        config.alpha_lr,
        config.adam_eps,
        config.tau,
        config.bootstrap_truncations,
        config.update_every,
        config.target_network_frequency,
        config.learning_starts,
        config.num_envs,
        config.curriculum_threshold,
        config.curriculum_min_episodes,
        config.clear_replay_on_advance,
        config.eval_every,
        config.eval_episodes,
    );

    let start_time = Instant::now();
    let mut last_log_time = start_time;
    let mut last_log_env_steps = 0usize;
    let log_dir = config.log_dir.to_string_lossy().into_owned();
    let mut writer = SummaryWriter::new(&log_dir);
    let mut rng = rand::rng();

    while completed_episodes < config.episodes {
        let actions = if env_steps < config.learning_starts {
            Tensor::from_slice(
                &(0..envs.len())
                    .map(|_| rng.random_range(0..OUTPUT_SIZE) as i64)
                    .collect::<Vec<_>>(),
            )
        } else {
            let states = Tensor::stack(
                &envs
                    .iter()
                    .map(|episode| episode.state.shallow_clone())
                    .collect::<Vec<_>>(),
                0,
            );
            actor
                .forward(&states)
                .softmax(-1, Kind::Float)
                .multinomial(1, true)
                .squeeze_dim(1)
        };

        for env_idx in 0..envs.len() {
            if completed_episodes >= config.episodes {
                break;
            }

            let action = actions.int64_value(&[env_idx as i64]) as usize;
            let episode = &mut envs[env_idx];
            let previous_state = episode.state.shallow_clone();
            let step = episode.env.step(action);
            let done = step.terminated || step.truncated;

            env_steps += 1;
            episode.reward += step.reward;
            episode.solved |= step.terminated;
            episode.truncated |= step.truncated;
            episode.max_solved_faces = episode
                .max_solved_faces
                .max(episode.env.get_cube().count_solved_faces());

            replay_buffer.push(Transition::new(
                &previous_state,
                action,
                step.reward,
                &step.next_state,
                step.terminated,
                step.truncated,
            ));
            episode.state = step.next_state;

            if env_steps > config.learning_starts
                && replay_buffer.len() >= config.batch_size
                && env_steps % config.update_every == 0
            {
                update_metrics.add(sac_update(
                    &config,
                    target_entropy,
                    &replay_buffer,
                    &actor,
                    &critic1,
                    &critic2,
                    &target_critic1,
                    &target_critic2,
                    &mut actor_opt,
                    &mut critic1_opt,
                    &mut critic2_opt,
                    &log_alpha,
                    &mut alpha_opt,
                ));
                learner_updates += 1;
            }

            if env_steps > config.learning_starts
                && env_steps % config.target_network_frequency == 0
            {
                update_target_networks(
                    &config,
                    &critic1_vs,
                    &critic2_vs,
                    &mut target_critic1_vs,
                    &mut target_critic2_vs,
                );
            }

            if !done {
                continue;
            }

            let episode_reward = episode.reward;
            let episode_solve = episode.solved;
            let episode_truncated = episode.truncated;
            let max_solved_faces = episode.max_solved_faces;

            last_100_solves[episodes_at_depth % 100] = episode_solve;
            episodes_at_depth += 1;
            completed_episodes += 1;
            if !curriculum_active && env_steps > config.learning_starts {
                curriculum_active = true;
                episodes_at_depth = 0;
                last_100_solves = [false; 100];
            }
            let recent_solves = last_100_solves.iter().filter(|&&s| s).count();

            if config.eval_every > 0 && completed_episodes % config.eval_every == 0 {
                let eval_metrics = evaluate_greedy(
                    &actor,
                    scramble_depth,
                    config.max_steps(scramble_depth),
                    config.eval_episodes,
                );
                writer.add_scalar(
                    "eval/greedy_solve_rate",
                    eval_metrics.solve_rate,
                    completed_episodes,
                );
                writer.add_scalar(
                    "eval/greedy_average_reward",
                    eval_metrics.average_reward,
                    completed_episodes,
                );
                writer.add_scalar(
                    "eval/greedy_average_steps",
                    eval_metrics.average_steps,
                    completed_episodes,
                );
            }

            if update_metrics.steps > 0 && completed_episodes % config.log_every == 0 {
                let now = Instant::now();
                let elapsed_secs = (now - start_time).as_secs_f32().max(f32::EPSILON);
                let recent_elapsed_secs = (now - last_log_time).as_secs_f32().max(f32::EPSILON);
                let sps = env_steps as f32 / elapsed_secs;
                let recent_sps = (env_steps - last_log_env_steps) as f32 / recent_elapsed_secs;

                writer.add_scalar(
                    "curriculum/scramble_depth",
                    scramble_depth as f32,
                    completed_episodes,
                );
                writer.add_scalar(
                    "curriculum/max_steps",
                    config.max_steps(scramble_depth) as f32,
                    completed_episodes,
                );
                writer.add_scalar(
                    "curriculum/replay_size",
                    replay_buffer.len() as f32,
                    completed_episodes,
                );
                writer.add_scalar("episode/reward", episode_reward, completed_episodes);
                writer.add_scalar(
                    "episode/solve",
                    if episode_solve { 1. } else { 0. },
                    completed_episodes,
                );
                writer.add_scalar(
                    "episode/truncated",
                    if episode_truncated { 1. } else { 0. },
                    completed_episodes,
                );
                writer.add_scalar(
                    "episode/max_solved_faces",
                    max_solved_faces as f32,
                    completed_episodes,
                );
                writer.add_scalar(
                    "episode/recent_solve_rate",
                    recent_solves as f32 / 100.,
                    completed_episodes,
                );
                writer.add_scalar(
                    "loss/critic_avg",
                    update_metrics.average(update_metrics.critic_loss) / 2.,
                    completed_episodes,
                );
                writer.add_scalar(
                    "loss/actor",
                    update_metrics.average(update_metrics.actor_loss),
                    completed_episodes,
                );
                writer.add_scalar(
                    "loss/alpha",
                    update_metrics.average(update_metrics.alpha_loss),
                    completed_episodes,
                );
                writer.add_scalar(
                    "alpha/value",
                    log_alpha.exp().double_value(&[]) as f32,
                    completed_episodes,
                );
                writer.add_scalar(
                    "alpha/log_value",
                    log_alpha.double_value(&[]) as f32,
                    completed_episodes,
                );
                writer.add_scalar(
                    "entropy/policy",
                    update_metrics.average(update_metrics.entropy),
                    completed_episodes,
                );
                writer.add_scalar("entropy/target", target_entropy as f32, completed_episodes);
                writer.add_scalar(
                    "entropy/error_policy_minus_target",
                    update_metrics.average(update_metrics.entropy_error),
                    completed_episodes,
                );
                writer.add_scalar(
                    "policy/max_probability",
                    update_metrics.average(update_metrics.policy_max_prob),
                    completed_episodes,
                );
                writer.add_scalar(
                    "q/target",
                    update_metrics.average(update_metrics.target_q),
                    completed_episodes,
                );
                writer.add_scalar(
                    "q/q1_taken",
                    update_metrics.average(update_metrics.q1),
                    completed_episodes,
                );
                writer.add_scalar(
                    "q/q2_taken",
                    update_metrics.average(update_metrics.q2),
                    completed_episodes,
                );
                writer.add_scalar(
                    "replay/truncation_rate",
                    update_metrics.average(update_metrics.replay_truncation),
                    completed_episodes,
                );
                writer.add_scalar("performance/sps", sps, completed_episodes);
                writer.add_scalar("performance/recent_sps", recent_sps, completed_episodes);
                writer.add_scalar(
                    "performance/env_steps",
                    env_steps as f32,
                    completed_episodes,
                );
                writer.add_scalar(
                    "performance/learner_updates",
                    learner_updates as f32,
                    completed_episodes,
                );
                writer.add_scalar(
                    "config/num_envs",
                    config.num_envs as f32,
                    completed_episodes,
                );
                writer.add_scalar(
                    "config/learning_starts",
                    config.learning_starts as f32,
                    completed_episodes,
                );
                writer.add_scalar(
                    "config/bootstrap_truncations",
                    if config.bootstrap_truncations { 1. } else { 0. },
                    completed_episodes,
                );
                writer.add_scalar(
                    "config/update_every",
                    config.update_every as f32,
                    completed_episodes,
                );
                writer.add_scalar(
                    "config/curriculum_threshold",
                    config.curriculum_threshold as f32,
                    completed_episodes,
                );
                writer.add_scalar(
                    "config/curriculum_min_episodes",
                    config.curriculum_min_episodes as f32,
                    completed_episodes,
                );
                writer.add_scalar(
                    "config/clear_replay_on_advance",
                    if config.clear_replay_on_advance {
                        1.
                    } else {
                        0.
                    },
                    completed_episodes,
                );
                writer.add_scalar(
                    "config/eval_episodes",
                    config.eval_episodes as f32,
                    completed_episodes,
                );
                writer.add_scalar(
                    "config/eval_every",
                    config.eval_every as f32,
                    completed_episodes,
                );
                writer.add_scalar(
                    "config/adam_eps",
                    config.adam_eps as f32,
                    completed_episodes,
                );
                writer.add_scalar(
                    "config/target_entropy_scale",
                    config.target_entropy_scale as f32,
                    completed_episodes,
                );
                writer.add_scalar(
                    "config/log_alpha_init",
                    config.log_alpha_init as f32,
                    completed_episodes,
                );
                writer.add_scalar(
                    "config/target_network_frequency",
                    config.target_network_frequency as f32,
                    completed_episodes,
                );

                println!(
                    "Episode {:6}/{:6} | depth: {:2} | solves: {:2}% | sps: {:7.1} | reward: {:6.2} | actor: {:7.4} | critic: {:7.4} | alpha: {:7.4} | entropy: {:6.4}/{:.4} | pmax: {:.3}",
                    completed_episodes,
                    config.episodes,
                    scramble_depth,
                    recent_solves,
                    recent_sps,
                    episode_reward,
                    update_metrics.average(update_metrics.actor_loss),
                    update_metrics.average(update_metrics.critic_loss) / 2.,
                    log_alpha.exp().double_value(&[]) as f32,
                    update_metrics.average(update_metrics.entropy),
                    target_entropy,
                    update_metrics.average(update_metrics.policy_max_prob),
                );

                update_metrics = UpdateMetricTotals::default();
                last_log_time = now;
                last_log_env_steps = env_steps;
            }

            if completed_episodes % config.save_every == 0 {
                actor_vs.save(config.net_dir.join("actor.ot"))?;
                critic1_vs.save(config.net_dir.join("critic1.ot"))?;
                critic2_vs.save(config.net_dir.join("critic2.ot"))?;
                alpha_vs.save(config.net_dir.join("alpha.ot"))?;
            }

            if curriculum_active
                && recent_solves >= config.curriculum_threshold
                && episodes_at_depth >= config.curriculum_min_episodes
                && scramble_depth < config.max_scramble
            {
                scramble_depth += 1;
                episodes_at_depth = 0;
                last_100_solves = [false; 100];
                if config.clear_replay_on_advance {
                    replay_buffer.clear();
                    update_metrics = UpdateMetricTotals::default();
                }
                println!("Advanced curriculum to depth {scramble_depth}");

                let max_steps = config.max_steps(scramble_depth);
                for episode in &mut envs {
                    episode.reset(scramble_depth, max_steps);
                }
            } else {
                envs[env_idx].reset(scramble_depth, config.max_steps(scramble_depth));
            }
        }
    }

    actor_vs.save(config.net_dir.join("actor.ot"))?;
    critic1_vs.save(config.net_dir.join("critic1.ot"))?;
    critic2_vs.save(config.net_dir.join("critic2.ot"))?;
    alpha_vs.save(config.net_dir.join("alpha.ot"))?;

    println!(
        "Finished training in {:.3}ms",
        (Instant::now() - start_time).as_millis()
    );
    Ok(())
}

fn relu_sq(xs: &Tensor) -> Tensor {
    let activated = xs.relu();
    &activated * &activated
}

fn linear(vs: nn::Path, in_dim: i64, out_dim: i64, ws_init: nn::Init) -> nn::Linear {
    nn::linear(
        vs,
        in_dim,
        out_dim,
        nn::LinearConfig {
            ws_init,
            bs_init: Some(nn::Init::Const(0.0)),
            bias: true,
        },
    )
}

fn hidden_linear(vs: nn::Path, in_dim: i64, out_dim: i64) -> nn::Linear {
    linear(vs, in_dim, out_dim, nn::init::DEFAULT_KAIMING_NORMAL)
}

fn residual_output_linear(vs: nn::Path, in_dim: i64, out_dim: i64) -> nn::Linear {
    linear(vs, in_dim, out_dim, nn::Init::Const(0.0))
}

fn head_linear(vs: nn::Path, in_dim: i64, out_dim: i64) -> nn::Linear {
    linear(
        vs,
        in_dim,
        out_dim,
        nn::Init::Randn {
            mean: 0.0,
            stdev: 0.01,
        },
    )
}

#[derive(Debug)]
struct ResidualBlock {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

#[derive(Debug)]
struct ResidualNetwork {
    input: nn::Linear,
    blocks: [ResidualBlock; 3],
    head: nn::Linear,
}

impl Module for ResidualNetwork {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut xs = relu_sq(&self.input.forward(xs));
        for block in &self.blocks {
            let residual = xs.shallow_clone();
            let hidden = relu_sq(&block.fc1.forward(&xs));
            xs = residual + block.fc2.forward(&hidden);
        }
        self.head.forward(&xs)
    }
}

fn initialize_network(vs: &nn::Path) -> ResidualNetwork {
    ResidualNetwork {
        input: hidden_linear(vs / "input", INPUT_SIZE as i64, 256),
        blocks: std::array::from_fn(|idx| ResidualBlock {
            fc1: hidden_linear(vs / format!("block{idx}_fc1"), 256, 256),
            fc2: residual_output_linear(vs / format!("block{idx}_fc2"), 256, 256),
        }),
        head: head_linear(vs / "head", 256, OUTPUT_SIZE as i64),
    }
}
