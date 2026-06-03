use std::time::Instant;

mod cube_env;

use rand::seq::SliceRandom;
use tch::{
    Device, Kind, TchError, Tensor,
    nn::{self, Module, OptimizerConfig},
};
use tensorboard_rs::summary_writer::SummaryWriter;

use crate::cube_env::{CubeEnv, ReplayBuffer, Transition};

// TODO: train from checkpoints
// TODO: charts with Tensorboard << next
// TODO: double DQN
// TODO: Seeding for reproducibility

const INPUT_SIZE: usize = 6 * 3 * 3 * 6;
const OUTPUT_SIZE: usize = 6 * 3;

fn main() {
    let _ = train();
}

pub fn get_device() -> Device {
    Device::cuda_if_available()
}

fn train() -> Result<(), TchError> {
    // Define hyperparameters
    let episodes = 50000;
    let batch_size = 64;
    let buffer_size = 50000;
    let learning_rate = 3e-4;
    let tau = 0.002;
    let gamma = 0.99;
    let mut alpha_start = 0.25;
    let alpha_floor = 0.03;
    let alpha_steady_state = 0.064;
    let alpha_decay = 300.;
    let mut alpha = alpha_start;
    let max_max_steps = 40;
    let min_steps = 3;
    let max_scramble = 20;

    // Initialize models
    let policy_vs = nn::VarStore::new(get_device());
    let target_vs = nn::VarStore::new(get_device());
    let policy_vs_root = policy_vs.root();
    let target_vs_root = target_vs.root();
    let policy_network = initialize_network(&policy_vs_root);
    let target_network = initialize_network(&target_vs_root);
    let mut opt = nn::Adam::default().build(&policy_vs, learning_rate)?;

    // Setup environment
    let mut cube_env = CubeEnv::new();
    let mut replay_buffer = ReplayBuffer::new(buffer_size);
    let mut last_100_solves = [false; 100];
    let mut scramble_depth = 1;
    let mut episodes_at_depth = 0;

    // Initialize logging
    println!("Beginning training on device: {:?}", get_device());
    let start_time = Instant::now();
    let mut writer = SummaryWriter::new("./logs");

    // Train loop
    for episode in 0..episodes {
        // Logging variables
        let mut episode_reward = 0.;
        let mut episode_loss = 0.;
        let mut loss_steps = 0;
        let mut episode_solve = false;

        // 1. Encode fresh environment
        let recent_solves = last_100_solves.iter().filter(|&&s| s).count();
        if recent_solves > 90 {
            scramble_depth += 1;
            episodes_at_depth = 0;
            if scramble_depth > max_scramble {
                scramble_depth = max_scramble;
            }

            // Seed solve buffer based on greedy solves
            // by running N greedy episodes to assess baseline performance at new depth
            let eval_episodes = 100;
            let mut greedy_solves = 0;
            let max_steps = (scramble_depth * 3).clamp(min_steps, max_max_steps);
            for _ in 0..eval_episodes {
                let mut s = cube_env.reset(scramble_depth, max_steps);
                for _ in 0..max_steps {
                    let q = policy_network.forward(&s.unsqueeze(0));
                    let a = q.argmax(1, false).int64_value(&[0]) as usize;
                    let (next_s, _, done) = cube_env.step(a);
                    s = next_s;
                    if done {
                        if cube_env.get_cube().is_solved() {
                            greedy_solves += 1;
                        }
                        break;
                    }
                }
            }
            let greedy_solve_rate = greedy_solves as f32 / eval_episodes as f32;
            println!(
                "Greedy baseline at depth {}: {:.0}%",
                scramble_depth,
                greedy_solve_rate * 100.0
            );

            // Randomly order the seeded solves to avoid overwriting them in order
            let seeded_solves = (greedy_solve_rate * 100.0) as usize;
            last_100_solves = [false; 100];
            let mut indices: Vec<usize> = (0..100).collect();
            indices.shuffle(&mut rand::rng());
            for i in 0..seeded_solves.min(100) {
                last_100_solves[indices[i]] = true;
            }

            // Give a small boost to alpha at the new depth
            let solve_rate = greedy_solves as f64 / eval_episodes as f64;
            alpha_start =
                alpha_floor + (alpha_steady_state - alpha_floor) * (1.0 + (1.0 - solve_rate) * 1.5);
        }
        episodes_at_depth += 1;

        let max_steps = (scramble_depth * 3).clamp(min_steps, max_max_steps);
        let mut state = cube_env.reset(scramble_depth, max_steps);

        for _ in 0..max_steps {
            // 2. Action selection using soft Q learning
            let state_batch = state.unsqueeze(0); // [324] -> [1, 324]
            let q_values = policy_network.forward(&state_batch);
            let v_values = alpha * (&q_values / alpha).exp().sum(Kind::Float).log();
            let dist = ((&q_values - v_values) / alpha).exp();
            let action_probs = &dist / dist.sum(Kind::Float);
            let action = action_probs.multinomial(1, true).int64_value(&[0]) as usize;

            // 3. Step environment → (next_state, reward, done)
            let (next_state, reward, done) = cube_env.step(action);
            episode_reward += reward;
            if cube_env.get_cube().is_solved() {
                episode_solve = true;
            }

            // 4. Push transition to replay buffer
            replay_buffer.push(Transition::new(&state, action, reward, &next_state, done));

            // Move to the next state
            state = next_state;

            // 5. If buffer large enough:
            if replay_buffer.len() >= batch_size {
                let batch = replay_buffer.sample(batch_size);

                // Stack into tensors
                let states = Tensor::stack(
                    &batch
                        .iter()
                        .map(|t| t.state.shallow_clone())
                        .collect::<Vec<_>>(),
                    0,
                )
                .to_device(get_device()); // [batch, 324]
                let next_states = Tensor::stack(
                    &batch
                        .iter()
                        .map(|t| t.next_state.shallow_clone())
                        .collect::<Vec<_>>(),
                    0,
                )
                .to_device(get_device()); // [batch, 324]
                let actions =
                    Tensor::from_slice(&batch.iter().map(|t| t.action as i64).collect::<Vec<_>>())
                        .to_device(get_device()); // [batch]
                let rewards =
                    Tensor::from_slice(&batch.iter().map(|t| t.reward).collect::<Vec<_>>())
                        .to_device(get_device()); // [batch]
                let dones = Tensor::from_slice(
                    &batch
                        .iter()
                        .map(|t| if t.done { 1f32 } else { 0f32 })
                        .collect::<Vec<_>>(),
                )
                .to_device(get_device()); // [batch]

                // Q(s, a) from policy network -> gather the Q-value for each action taken
                let q_values = policy_network
                    .forward(&states)
                    .gather(1, &actions.unsqueeze(1), false)
                    .squeeze_dim(1); // [batch]

                // Bellman targets from target network
                let next_q_values = tch::no_grad(|| {
                    let next_q = target_network.forward(&next_states); // [batch, 18]
                    let scaled = &next_q / alpha;
                    let max_q = scaled.max_dim(1, true).0; // [batch, 1] for numerical stability
                    let stable = (scaled - &max_q)
                        .exp()
                        .sum_dim_intlist(&[1i64][..], false, Kind::Float)
                        .log();
                    alpha * (stable + max_q.squeeze_dim(1)) // [batch]
                });
                let targets = &rewards + gamma * &next_q_values * (1. - &dones);

                // Huber loss and backprop
                let loss = q_values.huber_loss(&targets, tch::Reduction::Mean, 0.5);
                opt.zero_grad();
                loss.backward();
                // Clip gradients
                policy_vs.trainable_variables().iter().for_each(|v| {
                    let _ = v.grad().clamp_(-0.5, 0.5);
                });
                opt.step();

                // 6. Soft update
                tch::no_grad(|| {
                    for (target_param, policy_param) in target_vs
                        .trainable_variables()
                        .iter_mut()
                        .zip(policy_vs.trainable_variables().iter())
                    {
                        let updated = policy_param * tau + &*target_param * (1. - tau);
                        target_param.copy_(&updated);
                    }
                });

                // Logging
                episode_loss += f32::try_from(&loss).expect("loss calculation failed");
                loss_steps += 1;
            }

            // Logging
            writer.add_scalar("scramble depth", scramble_depth as f32, episode);
            writer.add_scalar("solve rate", recent_solves as f32, episode);
            writer.add_scalar("loss", episode_loss as f32, episode);
            writer.add_scalar("alpha", alpha as f32, episode);

            // 7. If done: reset environment (new scramble)
            if done {
                break;
            }
        }

        // Decay alpha
        alpha = alpha_floor
            + (alpha_start - alpha_floor) * (-(episodes_at_depth as f64) / alpha_decay).exp();

        // Update tracking
        last_100_solves[episodes_at_depth % 100] = episode_solve;

        // Logging
        println!(
            "Episode {:6}/{:6} | scramble depth: {:2} | solves: {:2}% | reward: {:6.2} | loss: {:7.4} | alpha: {:6.3}",
            episode + 1,
            episodes,
            scramble_depth,
            recent_solves,
            episode_reward,
            if loss_steps > 0 {
                episode_loss / loss_steps as f32
            } else {
                0.
            },
            alpha
        );

        // Save to file
        if episode % 100 == 0 {
            policy_vs
                .save("policy.ot")
                .expect("Failed to save policy net weights");
        }
    }

    println!(
        "Finished training in {:.3}ms",
        (Instant::now() - start_time).as_millis()
    );
    Result::Ok(())
}

fn initialize_network(vs: &nn::Path) -> impl Module {
    nn::seq()
        .add(nn::linear(
            vs / "layer1",
            INPUT_SIZE as i64,
            512,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs / "layer2", 512, 512, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs / "layer3", 512, 256, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(
            vs / "layer4",
            256,
            OUTPUT_SIZE as i64,
            Default::default(),
        ))
}
