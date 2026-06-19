use std::{array, env};

use rand::{SeedableRng, rngs::StdRng};
use tch::{Device, nn::Module};

use crate::{cube_env::CubeEnv, sac::train_vectorized};

mod cube_env;
mod logging;
mod sac;

const CUBE_SIZE: usize = 2;
const INPUT_SIZE: usize = 6 * CUBE_SIZE * CUBE_SIZE * 6;
const OUTPUT_SIZE: usize = 6 * 3;

// TODO: Train from checkpoints
// TODO: Seeding for reproducibility
// TODO: README file and TODO file

fn main() {
    let _ = train_vectorized();
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

pub fn get_device() -> Device {
    Device::cuda_if_available()
}

pub fn evaluate_model(model: &impl Module, model_name: &str) {
    // Perform a fixed seeded evaluation test for models
    let rng = StdRng::from_seed([42u8; 32]);
    let envs: [CubeEnv; 100] = array::from_fn(|_| CubeEnv::new());
}

fn evaluate_greedy(model: &impl Module, envs: &[CubeEnv; 100]) {
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
}
