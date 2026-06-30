mod actor_critic;
mod cube_env;
mod logging;

use std::{
    env,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use rand::{Rng, SeedableRng, rngs::StdRng};
use tch::{Device, Tensor, nn::Module};
use tensorboard_rs::summary_writer::SummaryWriter;

use crate::{actor_critic::train_vectorized, cube_env::CubeEnv};

const CUBE_SIZE: usize = 2;
const ACTIONS: usize = 3 * 6;

fn main() {
    let _ = train_vectorized(&TrainingConfig::from_env());
}

struct EpisodeState {
    env: CubeEnv,
    state: Tensor,
    reward: f32,
    solved: bool,
    truncated: bool,
}

impl EpisodeState {
    fn new(scramble_depth: usize, max_steps: usize) -> Self {
        let mut env = CubeEnv::new();
        let state = env.reset(scramble_depth, max_steps);

        Self {
            env,
            state,
            reward: 0.,
            solved: false,
            truncated: false,
        }
    }

    fn reset(&mut self, scramble_depth: usize, max_steps: usize) {
        self.state = self.env.reset(scramble_depth, max_steps);
        self.reward = 0.;
        self.solved = false;
        self.truncated = false;
    }

    fn seeded_reset(&mut self, scramble_depth: usize, max_steps: usize, seed: u64) {
        self.state = self.env.seeded_reset(scramble_depth, max_steps, seed);
        self.reward = 0.;
        self.solved = false;
        self.truncated = false;
    }
}

pub struct TrainingConfig {
    // Run identity
    run_name: String,
    log_dir: PathBuf,
    net_dir: PathBuf,
    // Episode structure
    episodes: usize,
    num_envs: usize,
    learning_starts: usize,
    min_steps: usize,
    max_steps_cap: usize,
    max_scramble: usize,
    // Evaluation
    eval_every: usize,
    eval_episodes: usize,
    // Logging & saving
    log_every: usize,
    save_every: usize,
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
            run_name: run_name.clone(),
            log_dir: PathBuf::from(log_root).join(&run_name),
            net_dir: PathBuf::from(net_root).join(&run_name),
            episodes: env_parse_min("RL_EPISODES", 50_000, 1),
            num_envs: env_parse_min("RL_NUM_ENVS", 16, 1),
            learning_starts: env_parse("RL_LEARNING_STARTS", 5_000),
            min_steps,
            max_steps_cap,
            max_scramble: env_parse_min("RL_MAX_SCRAMBLE", 20, 1),
            eval_every: env_parse("RL_EVAL_EVERY", 0),
            eval_episodes: env_parse_min("RL_EVAL_EPISODES", 64, 1),
            log_every: env_parse_min("RL_LOG_EVERY", 1, 1),
            save_every: env_parse_min("RL_SAVE_EVERY", 100, 1),
        }
    }

    pub fn max_steps(&self, scramble_depth: usize) -> usize {
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

fn get_device() -> Device {
    Device::cuda_if_available()
}

fn evaluate_model(
    model: &impl Module,
    envs: &mut [EpisodeState],
    config: &TrainingConfig,
    writer: &mut SummaryWriter,
    episode: usize,
) {
    // Perform a fixed seeded evaluation test for models
    let mut rng = StdRng::seed_from_u64(42);
    for depth in [5, 8, 11] {
        let (solve_rate, avg_reward, avg_steps) = evaluate_greedy(
            model,
            envs,
            &mut rng,
            depth,
            config.max_steps(depth),
            config.eval_episodes,
        );

        writer.add_scalar(
            &format!("eval/depth_{}/greedy_solve_rate", depth),
            solve_rate,
            episode,
        );
        writer.add_scalar(
            &format!("eval/depth_{}/greedy_average_reward", depth),
            avg_reward,
            episode,
        );
        writer.add_scalar(
            &format!("eval/depth_{}/greedy_average_steps", depth),
            avg_steps,
            episode,
        );
    }
}

fn evaluate_greedy(
    model: &impl Module,
    envs: &mut [EpisodeState],
    rng: &mut StdRng,
    scramble_depth: usize,
    max_steps: usize,
    episodes: usize,
) -> (f32, f32, f32) {
    let mut solves = 0usize;
    let mut total_reward = 0f32;
    let mut total_steps = 0usize;

    for env in envs.iter_mut() {
        env.seeded_reset(scramble_depth, max_steps, rng.next_u64());
    }

    let mut completed_episodes = 0;
    while completed_episodes < episodes {
        let states = Tensor::stack(&envs.iter().map(|e| &e.state).collect::<Vec<_>>(), 0);
        let actions = tch::no_grad(|| model.forward(&states).argmax(1, false)); // .squeeze_dim(1)

        for (env_idx, episode) in envs.iter_mut().enumerate() {
            let action = actions.int64_value(&[env_idx as i64]) as usize;

            let step = episode.env.step(action);
            episode.state = step.next_state;

            total_reward += step.reward;
            total_steps += 1;
            if step.terminated {
                solves += 1;
            }

            if step.terminated || step.truncated {
                completed_episodes += 1;
                if completed_episodes >= episodes {
                    break;
                }

                episode.seeded_reset(scramble_depth, max_steps, rng.next_u64());
            }
        }
    }

    (
        solves as f32 / episodes as f32,
        total_reward / episodes as f32,
        total_steps as f32 / episodes as f32,
    )
}
