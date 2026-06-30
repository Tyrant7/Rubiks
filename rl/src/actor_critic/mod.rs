pub mod network;
pub mod sac_logging;

use rand::RngExt;
use std::{
    fmt,
    time::{Duration, Instant},
};

use tch::{
    Kind, TchError, Tensor,
    nn::{self, Module, OptimizerConfig, VarStore},
};
use tensorboard_rs::summary_writer::SummaryWriter;

use crate::{
    ACTIONS, CUBE_SIZE, TrainingConfig,
    actor_critic::network::{INPUT_SIZE, ResNetwork},
    cube_env::SampleBuffer,
    env_parse, env_parse_bool, env_parse_clamped, env_parse_min, get_device,
};
use crate::{
    cube_env::{CubeEnv, ReplayBuffer, Transition},
    evaluate_model,
};

use crate::actor_critic::network::initialize_network;
use crate::actor_critic::sac_logging::{
    AlphaMetrics, CurriculumMetrics, EpisodeMetrics, PerformanceMetrics, UpdateMetricTotals,
    UpdateMetrics,
};
use crate::logging::{Loggable, write_scalars};

pub struct SacConfig {
    // Replay buffer
    buffer_size: usize,
    batch_size: usize,
    // Optimizer
    learning_rate: f64,
    alpha_lr: f64,
    adam_eps: f64,
    // TD learning
    gamma: f64,
    tau: f64,
    // Entropy / alpha
    target_entropy_scale: f64,
    log_alpha_init: f64,
    // Update schedule
    update_every: usize,
    target_network_frequency: usize,
    // Curriculum
    curriculum_threshold: usize,
    curriculum_min_episodes: usize,
    clear_replay_on_advance: bool,
}

impl SacConfig {
    fn from_env() -> Self {
        Self {
            buffer_size: env_parse_min("RL_BUFFER_SIZE", 200_000, 1),
            batch_size: env_parse_min("RL_BATCH_SIZE", 256, 1),
            learning_rate: env_parse("RL_LEARNING_RATE", 3e-4),
            alpha_lr: env_parse("RL_ALPHA_LR", 3e-4),
            adam_eps: env_parse("RL_ADAM_EPS", 1e-4),
            gamma: env_parse("RL_GAMMA", 0.99),
            tau: env_parse("RL_TAU", 1.0),
            target_entropy_scale: env_parse("RL_TARGET_ENTROPY_SCALE", 0.2),
            log_alpha_init: env_parse("RL_LOG_ALPHA_INIT", -2.0),
            update_every: env_parse_min("RL_UPDATE_EVERY", 4, 1),
            target_network_frequency: env_parse_min("RL_TARGET_NETWORK_FREQUENCY", 8_000, 1),
            curriculum_threshold: env_parse_clamped("RL_CURRICULUM_THRESHOLD", 10, 1, 100),
            curriculum_min_episodes: env_parse_min("RL_CURRICULUM_MIN_EPISODES", 1, 1),
            clear_replay_on_advance: env_parse_bool("RL_CLEAR_REPLAY_ON_ADVANCE", false),
        }
    }
}

struct LogSnapshot<'a> {
    completed_episodes: usize,
    total_episodes: usize,
    scramble_depth: usize,
    recent_solves: usize,
    recent_sps: f32,
    episode_reward: f32,
    update_metrics: &'a UpdateMetricTotals,
    log_alpha: f32,
    target_entropy: f32,
}

impl fmt::Display for LogSnapshot<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Episode {:6}/{:6} | depth: {:2} | solves: {:2}% | sps: {:7.1} | reward: {:6.2} | actor: {:7.4} | critic: {:7.4} | alpha: {:7.4} | entropy: {:6.4}/{:.4} | pmax: {:.3}",
            self.completed_episodes,
            self.total_episodes,
            self.scramble_depth,
            self.recent_solves,
            self.recent_sps,
            self.episode_reward,
            self.update_metrics.average(self.update_metrics.actor_loss),
            self.update_metrics.average(self.update_metrics.critic_loss) / 2.,
            self.log_alpha,
            self.update_metrics.average(self.update_metrics.entropy),
            self.target_entropy,
            self.update_metrics
                .average(self.update_metrics.policy_max_prob),
        )
    }
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

fn adam(config: &SacConfig) -> nn::Adam {
    nn::Adam::default().eps(config.adam_eps)
}

#[allow(clippy::too_many_arguments)]
fn sac_update(
    config: &SacConfig,
    target_entropy: f64,
    samples: &SampleBuffer,
    actor: &ResNetwork,
    critic1: &ResNetwork,
    critic2: &ResNetwork,
    target_critic1: &ResNetwork,
    target_critic2: &ResNetwork,
    actor_opt: &mut nn::Optimizer,
    critic1_opt: &mut nn::Optimizer,
    critic2_opt: &mut nn::Optimizer,
    actor_vs: &VarStore,
    critic1_vs: &VarStore,
    critic2_vs: &VarStore,
    log_alpha: &Tensor,
    alpha_opt: &mut nn::Optimizer,
    update_metrics: bool,
) -> Option<UpdateMetrics> {
    let t = Instant::now();
    let states = &samples.states;
    let next_states = &samples.next_states;
    let actions = &samples.actions;
    let rewards = &samples.rewards;
    let terminated = &samples.terminated;
    let truncated = &samples.truncated;
    let t_sample = t.elapsed();

    let t = Instant::now();
    let target = tch::no_grad(|| {
        let next_logits = actor.forward(next_states);
        let next_probs = next_logits.softmax(-1, Kind::Float);
        let next_log_probs = next_logits.log_softmax(-1, Kind::Float);
        let next_q1 = target_critic1.forward(next_states);
        let next_q2 = target_critic2.forward(next_states);
        let next_min_q = next_q1.minimum(&next_q2);
        let next_v = (&next_probs * (&next_min_q - log_alpha.exp() * &next_log_probs))
            .sum_dim_intlist(&[1i64][..], false, Kind::Float);
        rewards + config.gamma * &next_v * (1. - terminated)
    });
    let t_target = t.elapsed();

    let t = Instant::now();
    let q1 = critic1.forward(states);
    let q2 = critic2.forward(states);
    let q_min = q1.minimum(&q2);

    // Optimization -> calculating both critic losses at once
    let q1 = q1.gather(1, &actions.unsqueeze(1), false).squeeze_dim(1);
    let q2 = q2.gather(1, &actions.unsqueeze(1), false).squeeze_dim(1);
    let critic1_loss = q1.mse_loss(&target, tch::Reduction::Mean);
    let critic2_loss = q2.mse_loss(&target, tch::Reduction::Mean);
    let critic_loss = &critic1_loss + &critic2_loss;

    critic1_opt.zero_grad();
    critic2_opt.zero_grad();

    critic_loss.backward();
    let critic1_grad_norm = grad_norm(critic1_vs);
    let critic2_grad_norm = grad_norm(critic2_vs);

    critic1_opt.step();
    critic2_opt.step();

    let t_critic = t.elapsed();

    let t = Instant::now();
    let logits = actor.forward(states);
    let probs = logits.softmax(-1, Kind::Float);
    let log_probs = logits.log_softmax(-1, Kind::Float);
    let alpha = log_alpha.exp();
    let actor_loss = (&probs * (&alpha * &log_probs - &q_min.detach()))
        .sum_dim_intlist(&[1i64][..], false, Kind::Float)
        .mean(Kind::Float);
    actor_opt.zero_grad();
    actor_loss.backward();
    let actor_grad_norm = grad_norm(actor_vs);
    actor_opt.step();
    let t_actor = t.elapsed();

    let t = Instant::now();
    let entropy = -(&probs * &log_probs)
        .sum_dim_intlist(&[1i64][..], false, Kind::Float)
        .mean(Kind::Float);
    let entropy_error = &entropy - target_entropy;
    let alpha_loss = (-log_alpha * (&log_probs + target_entropy).detach())
        .sum_dim_intlist(&[1i64][..], false, Kind::Float)
        .mean(Kind::Float);
    alpha_opt.zero_grad();
    alpha_loss.backward();
    alpha_opt.step();
    let t_alpha = t.elapsed();

    if update_metrics {
        println!(
            "SAC breakdown | sample: {:.2}ms | target: {:.2}ms | critics: {:.2}ms | actor: {:.2}ms | alpha: {:.2}ms",
            t_sample.as_secs_f64() * 1000.,
            t_target.as_secs_f64() * 1000.,
            t_critic.as_secs_f64() * 1000.,
            t_actor.as_secs_f64() * 1000.,
            t_alpha.as_secs_f64() * 1000.,
        );

        Some(UpdateMetrics {
            actor_loss: f32::try_from(&actor_loss).expect("loss calculation failed"),
            critic_loss: f32::try_from(&critic1_loss).expect("loss calculation failed")
                + f32::try_from(&critic2_loss).expect("loss calculation failed"),
            critic1_grad_norm: critic1_grad_norm as f32,
            critic2_grad_norm: critic2_grad_norm as f32,
            actor_grad_norm: actor_grad_norm as f32,
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
        })
    } else {
        None
    }
}

pub fn train_vectorized(config: &TrainingConfig) -> Result<(), TchError> {
    if tch::Cuda::is_available() {
        tch::Cuda::cudnn_set_benchmark(true);
    }

    let sac_config = SacConfig::from_env();
    std::fs::create_dir_all(&config.net_dir).expect("failed to create net directory");
    std::fs::create_dir_all(&config.log_dir).expect("failed to create log directory");

    let alpha_vs = nn::VarStore::new(get_device());
    let log_alpha =
        alpha_vs
            .root()
            .var("log_alpha", &[], nn::Init::Const(sac_config.log_alpha_init));
    let target_entropy = sac_config.target_entropy_scale * (ACTIONS as f64).ln();
    let mut alpha_opt = adam(&sac_config).build(&alpha_vs, sac_config.alpha_lr)?;

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

    let mut actor_opt = adam(&sac_config).build(&actor_vs, sac_config.learning_rate)?;
    let mut critic1_opt = adam(&sac_config).build(&critic1_vs, sac_config.learning_rate)?;
    let mut critic2_opt = adam(&sac_config).build(&critic2_vs, sac_config.learning_rate)?;

    let mut replay_buffer = ReplayBuffer::new(sac_config.buffer_size);
    let mut sample_buffer = SampleBuffer::new(sac_config.batch_size as i64, INPUT_SIZE);
    let mut last_100_solves = [false; 100];
    let mut scramble_depth = 1usize;
    let mut episodes_at_depth = 0usize;
    let mut completed_episodes = 0usize;
    let mut env_steps = 0usize;
    let mut learner_updates = 0usize;
    let mut curriculum_active = config.learning_starts == 0;
    let mut update_metrics = UpdateMetricTotals::default();
    let mut updates_since_metrics = 0usize;
    const METRICS_EVERY: usize = 64;

    // Profiling accumulators
    let mut time_actions = Duration::ZERO;
    let mut time_steps = Duration::ZERO;
    let mut time_update = Duration::ZERO;
    let mut time_target = Duration::ZERO;
    let mut time_bookkeeping = Duration::ZERO;

    let mut action_batches = 0usize;
    let mut env_transitions = 0usize;
    let mut updates_run = 0usize;
    let mut target_updates_run = 0usize;

    let max_steps = config.max_steps(scramble_depth);
    let mut envs = (0..config.num_envs)
        .map(|_| EpisodeState::new(scramble_depth, max_steps))
        .collect::<Vec<_>>();

    println!(
        "Beginning training run {} for cube of size {}x{}x{} on device {:?} for actor with {} params",
        config.run_name,
        CUBE_SIZE,
        CUBE_SIZE,
        CUBE_SIZE,
        get_device(),
        actor_vs
            .trainable_variables()
            .iter()
            .map(|t| t.numel())
            .sum::<usize>()
    );
    println!(
        "logs: {} | nets: {} | batch_size: {:4} | target entropy: {:.4} | target entropy scale: {:.3} | log alpha init: {:.3} | alpha lr: {:.1e} | adam eps: {:.1e} | tau: {:.4} | update every: {} | target sync: {} | learning starts: {} | envs: {} | curriculum threshold: {}%/{}eps | clear replay: {} | eval: {}x{}",
        config.log_dir.display(),
        config.net_dir.display(),
        sac_config.batch_size,
        target_entropy,
        sac_config.target_entropy_scale,
        sac_config.log_alpha_init,
        sac_config.alpha_lr,
        sac_config.adam_eps,
        sac_config.tau,
        sac_config.update_every,
        sac_config.target_network_frequency,
        config.learning_starts,
        config.num_envs,
        sac_config.curriculum_threshold,
        sac_config.curriculum_min_episodes,
        sac_config.clear_replay_on_advance,
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
        // --- Action selection ---
        let t = Instant::now();
        let actions = if env_steps < config.learning_starts {
            Tensor::from_slice(
                &(0..envs.len())
                    .map(|_| rng.random_range(0..ACTIONS) as i64)
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
        time_actions += t.elapsed();
        action_batches += 1;

        for env_idx in 0..envs.len() {
            if completed_episodes >= config.episodes {
                break;
            }

            let action = actions.int64_value(&[env_idx as i64]) as usize;
            let episode = &mut envs[env_idx];

            // --- Env step ---
            let t = Instant::now();
            let previous_state = episode.state.shallow_clone();
            let step = episode.env.step(action);
            let done = step.terminated || step.truncated;
            time_steps += t.elapsed();
            env_transitions += 1;

            // --- Bookkeeping ---
            let t = Instant::now();
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
            time_bookkeeping += t.elapsed();

            // --- SAC update ---
            if env_steps > config.learning_starts
                && replay_buffer.len() >= sac_config.batch_size
                && env_steps.is_multiple_of(sac_config.update_every)
            {
                let t = Instant::now();

                updates_since_metrics += 1;
                let extract = updates_since_metrics >= METRICS_EVERY;
                if extract {
                    updates_since_metrics = 0;
                }

                replay_buffer.sample_tensors(&mut sample_buffer);
                if let Some(metrics) = sac_update(
                    &sac_config,
                    target_entropy,
                    &sample_buffer,
                    &actor,
                    &critic1,
                    &critic2,
                    &target_critic1,
                    &target_critic2,
                    &mut actor_opt,
                    &mut critic1_opt,
                    &mut critic2_opt,
                    &critic1_vs,
                    &critic2_vs,
                    &actor_vs,
                    &log_alpha,
                    &mut alpha_opt,
                    extract,
                ) {
                    update_metrics.add(metrics);
                }

                time_update += t.elapsed();
                updates_run += 1;

                learner_updates += 1;
            }

            // --- Target update ---
            if env_steps > config.learning_starts
                && env_steps.is_multiple_of(sac_config.target_network_frequency)
            {
                let t = Instant::now();

                update_target_networks(
                    &sac_config,
                    &critic1_vs,
                    &critic2_vs,
                    &mut target_critic1_vs,
                    &mut target_critic2_vs,
                );

                time_target += t.elapsed();
                target_updates_run += 1;
            }

            if !done {
                continue;
            }

            let episode_metrics = EpisodeMetrics {
                reward: episode.reward,
                solved: episode.solved,
                truncated: episode.truncated,
                max_solved_faces: episode.max_solved_faces,
                recent_solve_rate: {
                    last_100_solves[episodes_at_depth % 100] = episode.solved;
                    episodes_at_depth += 1;
                    completed_episodes += 1;
                    if !curriculum_active && env_steps > config.learning_starts {
                        curriculum_active = true;
                        episodes_at_depth = 0;
                        last_100_solves = [false; 100];
                    }
                    last_100_solves.iter().filter(|&&s| s).count() as f32 / 100.
                },
            };
            let recent_solves = (episode_metrics.recent_solve_rate * 100.) as usize;

            if config.eval_every > 0 && completed_episodes.is_multiple_of(config.eval_every) {
                evaluate_model(&actor, config, &mut writer, completed_episodes);
            }

            if update_metrics.steps > 0 && completed_episodes.is_multiple_of(config.log_every) {
                let now = Instant::now();
                let elapsed_secs = (now - start_time).as_secs_f32().max(f32::EPSILON);
                let recent_elapsed_secs = (now - last_log_time).as_secs_f32().max(f32::EPSILON);

                let perf_metrics = PerformanceMetrics {
                    sps: env_steps as f32 / elapsed_secs,
                    recent_sps: (env_steps - last_log_env_steps) as f32 / recent_elapsed_secs,
                    env_steps,
                    learner_updates,
                };
                let curriculum_metrics = CurriculumMetrics {
                    scramble_depth,
                    max_steps: config.max_steps(scramble_depth),
                    replay_size: replay_buffer.len(),
                };
                let alpha_metrics = AlphaMetrics {
                    value: log_alpha.exp().double_value(&[]) as f32,
                    log_value: log_alpha.double_value(&[]) as f32,
                    target_entropy: target_entropy as f32,
                };

                write_scalars(
                    &mut writer,
                    &curriculum_metrics.scalars(),
                    completed_episodes,
                );
                write_scalars(&mut writer, &episode_metrics.scalars(), completed_episodes);
                write_scalars(&mut writer, &update_metrics.scalars(), completed_episodes);
                write_scalars(&mut writer, &alpha_metrics.scalars(), completed_episodes);
                write_scalars(&mut writer, &perf_metrics.scalars(), completed_episodes);
                write_scalars(&mut writer, &sac_config.scalars(), completed_episodes);
                write_scalars(&mut writer, &config.scalars(), completed_episodes);

                println!(
                    "{}",
                    LogSnapshot {
                        completed_episodes,
                        total_episodes: config.episodes,
                        scramble_depth,
                        recent_solves,
                        recent_sps: perf_metrics.recent_sps,
                        episode_reward: episode_metrics.reward,
                        update_metrics: &update_metrics,
                        log_alpha: alpha_metrics.value,
                        target_entropy: target_entropy as f32,
                    }
                );

                update_metrics = UpdateMetricTotals::default();
                last_log_time = now;
                last_log_env_steps = env_steps;

                // Profiling
                println!(
                    "Profile | actions: {:.3}ms/batch | steps: {:.3}ms/step | bookkeeping: {:.3}ms/step | update: {:.3}ms/update | target: {:.3}ms/update",
                    time_actions.as_secs_f64() * 1000.0 / action_batches.max(1) as f64,
                    time_steps.as_secs_f64() * 1000.0 / env_transitions.max(1) as f64,
                    time_bookkeeping.as_secs_f64() * 1000.0 / env_transitions.max(1) as f64,
                    time_update.as_secs_f64() * 1000.0 / updates_run.max(1) as f64,
                    time_target.as_secs_f64() * 1000.0 / target_updates_run.max(1) as f64,
                );

                time_actions = Duration::ZERO;
                time_steps = Duration::ZERO;
                time_update = Duration::ZERO;
                time_target = Duration::ZERO;
                time_bookkeeping = Duration::ZERO;

                action_batches = 0;
                env_transitions = 0;
                updates_run = 0;
                target_updates_run = 0;
            }

            if completed_episodes.is_multiple_of(config.save_every) {
                actor_vs.save(config.net_dir.join("actor.ot"))?;
                critic1_vs.save(config.net_dir.join("critic1.ot"))?;
                critic2_vs.save(config.net_dir.join("critic2.ot"))?;
                alpha_vs.save(config.net_dir.join("alpha.ot"))?;
            }

            if curriculum_active
                && recent_solves >= sac_config.curriculum_threshold
                && episodes_at_depth >= sac_config.curriculum_min_episodes
                && scramble_depth < config.max_scramble
            {
                scramble_depth += 1;
                episodes_at_depth = 0;
                last_100_solves = [false; 100];
                if sac_config.clear_replay_on_advance {
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

fn update_target_networks(
    sac_config: &SacConfig,
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
            let updated = pp * sac_config.tau + &*tp * (1. - sac_config.tau);
            tp.copy_(&updated);
        }
        for (tp, pp) in target_critic2_vs
            .trainable_variables()
            .iter_mut()
            .zip(critic2_vs.trainable_variables().iter())
        {
            let updated = pp * sac_config.tau + &*tp * (1. - sac_config.tau);
            tp.copy_(&updated);
        }
    });
}

fn grad_norm(vs: &nn::VarStore) -> f64 {
    let total: f64 = vs
        .trainable_variables()
        .iter()
        .filter_map(|p| {
            let grad = p.grad();
            if grad.defined() {
                Some(grad.norm().double_value(&[]).powi(2))
            } else {
                None
            }
        })
        .sum();
    total.sqrt()
}
