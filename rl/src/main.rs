use std::time::Instant;

mod cube_env;

use rand::seq::SliceRandom;
use tch::{
    Device, Kind, TchError, Tensor,
    nn::{self, Module, OptimizerConfig},
};
use tensorboard_rs::summary_writer::SummaryWriter;

use crate::cube_env::{CubeEnv, ReplayBuffer, Transition};

// TODO: Train from checkpoints
// TODO: Seeding for reproducibility
// TODO: SAC
// TODO: README file and TODO file

const CUBE_SIZE: usize = 2;
const INPUT_SIZE: usize = 6 * CUBE_SIZE * CUBE_SIZE * 6;
const OUTPUT_SIZE: usize = 6 * 3;

fn main() {
    let _ = train();
}

pub fn get_device() -> Device {
    Device::cuda_if_available()
}

fn train() -> Result<(), TchError> {
    // Initialize network directory
    std::fs::create_dir_all("./rl/nets").expect("Failed to create nets directory");

    // Define hyperparameters
    let episodes = 50000;
    let batch_size = 512;
    let buffer_size = 200000;
    let learning_rate = 3e-4;
    let alpha_lr = 1e-4;
    let tau = 0.005;
    let gamma = 0.99;
    let max_max_steps = 40;
    let min_steps = 3;
    let max_scramble = 20;

    let alpha_vs = nn::VarStore::new(get_device());
    let log_alpha = alpha_vs.root().var("log_alpha", &[], nn::Init::Const(-2.0));
    let target_entropy = -0.89 * (OUTPUT_SIZE as f64).ln();
    let mut alpha_opt = nn::Adam::default().build(&alpha_vs, alpha_lr)?;

    // Initialize models
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

    let mut actor_opt = nn::Adam::default().build(&actor_vs, learning_rate)?;
    let mut critic1_opt = nn::Adam::default().build(&critic1_vs, learning_rate)?;
    let mut critic2_opt = nn::Adam::default().build(&critic2_vs, learning_rate)?;

    // Setup environment
    let mut cube_env = CubeEnv::new();
    let mut replay_buffer = ReplayBuffer::new(buffer_size);
    let mut last_100_solves = [false; 100];
    let mut scramble_depth = 1;
    let mut episodes_at_depth = 0;

    // Initialize logging
    println!(
        "Beginning training for cube of size: {} on device: {:?}",
        CUBE_SIZE,
        get_device()
    );
    let start_time = Instant::now();
    let mut writer = SummaryWriter::new("./rl/logs");

    // Train loop
    for episode in 0..episodes {
        // Logging variables
        let mut episode_reward = 0.;
        let mut critic_episode_loss = 0.;
        let mut actor_episode_loss = 0.;
        let mut loss_steps = 0;
        let mut episode_solve = false;
        let mut episode_entropy = 0.;

        // Encode fresh environment
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
                    let logits = actor.forward(&s.unsqueeze(0));
                    let probs = logits.softmax(-1, Kind::Float);
                    let a = probs.argmax(1, false).int64_value(&[0]) as usize;
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
        }

        let max_steps = (scramble_depth * 3).clamp(min_steps, max_max_steps);
        let mut state = cube_env.reset(scramble_depth, max_steps);

        for _ in 0..max_steps {
            let logits = actor.forward(&state.unsqueeze(0)); // [INPUT_SIZE] -> [1, INPUT_SIZE]
            let probs = logits.softmax(-1, Kind::Float);

            // Fall back to argmax if distribution is degenerate
            let action = if probs.max().double_value(&[]) > 0.999 {
                probs.argmax(1, false).int64_value(&[0]) as usize
            } else {
                probs.multinomial(1, true).int64_value(&[0]) as usize
            };

            // Step environment -> (next_state, reward, done)
            let (next_state, reward, done) = cube_env.step(action);
            episode_reward += reward;
            if cube_env.get_cube().is_solved() {
                episode_solve = true;
            }

            // Push transition to replay buffer
            replay_buffer.push(Transition::new(&state, action, reward, &next_state, done));

            // Move to the next state
            state = next_state;

            // If buffer large enough:
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
                .to_device(get_device()); // [batch, INPUT_SIZE]
                let next_states = Tensor::stack(
                    &batch
                        .iter()
                        .map(|t| t.next_state.shallow_clone())
                        .collect::<Vec<_>>(),
                    0,
                )
                .to_device(get_device()); // [batch, INPUT_SIZE]
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

                // === CRITIC UPDATE ===
                // Compute target using actor's current policy and min of target critics
                let target = tch::no_grad(|| {
                    let next_logits = actor.forward(&next_states); // [batch, 18]
                    let next_probs = next_logits.softmax(-1, Kind::Float);
                    let next_log_probs = next_logits.log_softmax(-1, Kind::Float);

                    let next_q1 = target_critic1.forward(&next_states);
                    let next_q2 = target_critic2.forward(&next_states);
                    let next_min_q = next_q1.minimum(&next_q2);

                    // Soft value: expectations over all actions of (Q - alpha * log pi)
                    let next_v = (&next_probs * (&next_min_q - log_alpha.exp() * &next_log_probs))
                        .sum_dim_intlist(&[1i64][..], false, Kind::Float); // [batch]

                    &rewards + gamma * &next_v * (1. - &dones) // [batch]
                });

                // Update critic 1
                let q1 = critic1
                    .forward(&states)
                    .gather(1, &actions.unsqueeze(1), false)
                    .squeeze_dim(1);
                let critic1_loss = q1.huber_loss(&target, tch::Reduction::Mean, 0.5);
                critic1_opt.zero_grad();
                critic1_loss.backward();
                critic1_vs.trainable_variables().iter().for_each(|v| {
                    let _ = v.grad().clamp_(-0.5, 0.5);
                });
                critic1_opt.step();

                // Update critic 2
                let q2 = critic2
                    .forward(&states)
                    .gather(1, &actions.unsqueeze(1), false)
                    .squeeze_dim(1);
                let critic2_loss = q2.huber_loss(&target, tch::Reduction::Mean, 0.5);
                critic2_opt.zero_grad();
                critic2_loss.backward();
                critic2_vs.trainable_variables().iter().for_each(|v| {
                    let _ = v.grad().clamp_(-0.5, 0.5);
                });
                critic2_opt.step();

                // === ACTOR UPDATE ===
                let logits = actor.forward(&states); // [batch, 18]
                let probs = logits.softmax(-1, Kind::Float);
                let log_probs = logits.log_softmax(-1, Kind::Float);

                let q1_detached = tch::no_grad(|| critic1.forward(&states));
                let q2_detached = tch::no_grad(|| critic2.forward(&states));
                let q_min = q1_detached.minimum(&q2_detached);

                // Actor loss: wants high Q-values but penalized for being too certain'
                let actor_loss = (&probs * (log_alpha.exp() * &log_probs - &q_min))
                    .sum_dim_intlist(&[1i64][..], false, Kind::Float)
                    .mean(Kind::Float);
                actor_opt.zero_grad();
                actor_loss.backward();
                actor_vs.trainable_variables().iter().for_each(|v| {
                    let _ = v.grad().clamp_(-0.5, 0.5);
                });
                actor_opt.step();

                // === ALPHA UPDATE ===
                let entropy = tch::no_grad(|| {
                    let logits = tch::no_grad(|| actor.forward(&states));
                    let probs = logits.softmax(-1, Kind::Float);
                    let log_probs = logits.log_softmax(-1, Kind::Float);
                    -(&probs * &log_probs)
                        .sum_dim_intlist(&[1i64][..], false, Kind::Float)
                        .mean(Kind::Float)
                });
                let alpha_loss = log_alpha.exp() * (&entropy - target_entropy);

                alpha_opt.zero_grad();
                alpha_loss.backward();
                alpha_opt.step();

                // Soft updates
                tch::no_grad(|| {
                    for (tp, pp) in target_critic1_vs
                        .trainable_variables()
                        .iter_mut()
                        .zip(critic1_vs.trainable_variables().iter())
                    {
                        let updated = pp * tau + &*tp * (1. - tau);
                        tp.copy_(&updated);
                    }
                    for (tp, pp) in target_critic2_vs
                        .trainable_variables()
                        .iter_mut()
                        .zip(critic2_vs.trainable_variables().iter())
                    {
                        let updated = pp * tau + &*tp * (1. - tau);
                        tp.copy_(&updated);
                    }
                });

                // Logging
                actor_episode_loss += f32::try_from(&actor_loss).expect("loss calculation failed");
                critic_episode_loss += f32::try_from(&critic1_loss)
                    .expect("loss calculation failed")
                    + f32::try_from(&critic2_loss).expect("loss calculation failed");
                episode_entropy += f32::try_from(&entropy).expect("entropy calculation failed");
                loss_steps += 1;
            }

            // If done: reset environment (new scramble)
            if done {
                break;
            }
        }

        // Logging
        if loss_steps > 0 {
            writer.add_scalar("scramble depth", scramble_depth as f32, episode);
            writer.add_scalar("solve rate", recent_solves as f32 / 100., episode);
            writer.add_scalar(
                "critic_loss (avg)",
                critic_episode_loss / loss_steps as f32 / 2.,
                episode,
            );
            writer.add_scalar(
                "actor_loss",
                actor_episode_loss / loss_steps as f32,
                episode,
            );
            writer.add_scalar("alpha", log_alpha.exp().double_value(&[]) as f32, episode);
            writer.add_scalar("entropy", episode_entropy / loss_steps as f32, episode);

            println!(
                "Episode {:6}/{:6} | scramble depth: {:2} | solves: {:2}% | reward: {:6.2} | actor loss: {:7.4} | critic loss (avg): {:7.4} | alpha: {:5.3} | entropy: {:6.4}",
                episode + 1,
                episodes,
                scramble_depth,
                recent_solves,
                episode_reward,
                actor_episode_loss / loss_steps as f32,
                critic_episode_loss / 2. / loss_steps as f32,
                log_alpha.exp().double_value(&[]) as f32,
                episode_entropy / loss_steps as f32,
            );
        }

        // Update tracking
        last_100_solves[episodes_at_depth % 100] = episode_solve;
        episodes_at_depth += 1;

        // Save to file
        if episode % 100 == 0 {
            actor_vs.save("./rl/nets/actor.ot")?;
            critic1_vs.save("./rl/nets/critic1.ot")?;
            critic2_vs.save("./rl/nets/critic2.ot")?;
            alpha_vs.save("./rl/nets/alpha.ot")?;
        }
    }

    println!(
        "Finished training in {:.3}ms",
        (Instant::now() - start_time).as_millis()
    );
    Result::Ok(())
}

fn initialize_network(vs: &nn::Path) -> nn::Sequential {
    nn::seq()
        .add(nn::linear(
            vs / "layer1",
            INPUT_SIZE as i64,
            256,
            Default::default(),
        ))
        .add(nn::layer_norm(vs / "ln1", vec![256], Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs / "layer2", 256, 256, Default::default()))
        .add(nn::layer_norm(vs / "ln2", vec![256], Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs / "layer3", 256, 128, Default::default()))
        .add(nn::layer_norm(vs / "ln3", vec![128], Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(
            vs / "layer4",
            128,
            OUTPUT_SIZE as i64,
            Default::default(),
        ))
}
