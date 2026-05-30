use std::time::Instant;

use rand::seq::IndexedRandom;
use rubiks::{CUBE_SIZE, Cube, FaceType, Turn, TurnType};
use tch::{
    Device, TchError, Tensor,
    nn::{self, Module, OptimizerConfig},
};

// TODO: train from checkpoints too

const INPUT_SIZE: usize = 6 * 3 * 3 * 6;
const OUTPUT_SIZE: usize = 6 * 3;

fn main() {
    let _ = train();
}

fn train() -> Result<(), TchError> {
    // Define hyperparameters
    let episodes = 1000;
    let batch_size = 16;
    let epsilon_start = 0.9;
    let epsilon_end = 0.05;
    let epsilon_decay = 0.003;
    let learning_rate = 1e-3;
    let tau = 0.005;
    let gamma = 0.99;
    let max_steps = 40;
    let max_scramble = 20;

    // Initialize models
    let policy_vs = nn::VarStore::new(Device::Cpu);
    let target_vs = nn::VarStore::new(Device::Cpu);
    let policy_vs_root = policy_vs.root();
    let target_vs_root = target_vs.root();
    let policy_network = initialize_network(&policy_vs_root);
    let target_network = initialize_network(&target_vs_root);
    let mut opt = nn::Adam::default().build(&policy_vs, learning_rate)?;

    // Setup environment
    let mut cube_env = CubeEnv::new(max_steps);
    let mut replay_buffer = ReplayBuffer::new(10000);
    let mut last_100_rewards = [0f32; 100];
    let mut last_reward_idx = 0;
    let mut scramble_depth = 1;
    let mut epsilon = epsilon_start;

    // Initialize logging
    println!("Beginning training...");
    let start_time = Instant::now();

    // Train loop
    for episode in 0..episodes {
        // Logging variables
        let mut episode_reward = 0.;
        let mut episode_loss = 0.;
        let mut loss_steps = 0;

        // 1. Encode fresh environment
        let recent_solves = last_100_rewards.iter().filter(|&&r| r > 0.).count();
        if recent_solves > 90 {
            scramble_depth += 1;
            if scramble_depth > max_scramble {
                scramble_depth = max_scramble;
            }
            epsilon = epsilon_start * 0.5;
            last_100_rewards = [0f32; 100];
        }
        let mut state = cube_env.reset(scramble_depth);

        // Epsilon with linear decay
        epsilon -= epsilon_decay;
        if epsilon < epsilon_end {
            epsilon = epsilon_end;
        }

        for _ in 0..max_steps {
            // 2. ε-greedy action selection
            let action = if rand::random::<f32>() < epsilon {
                // with prob ε: random action
                rand::random_range(0..OUTPUT_SIZE)
            } else {
                // with prob 1-ε: argmax over Q(s, ·)
                let state_batch = state.unsqueeze(0); // [324] -> [1, 324]
                let q_values = policy_network.forward(&state_batch);
                q_values.argmax(1, false).int64_value(&[0]) as usize
            };

            // 3. Step environment → (next_state, reward, done)
            let (next_state, reward, done) = cube_env.step(action);
            episode_reward += reward;

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
                ); // [batch, 324]
                let next_states = Tensor::stack(
                    &batch
                        .iter()
                        .map(|t| t.next_state.shallow_clone())
                        .collect::<Vec<_>>(),
                    0,
                ); // [batch, 324]
                let actions =
                    Tensor::from_slice(&batch.iter().map(|t| t.action as i64).collect::<Vec<_>>()); // [batch]
                let rewards =
                    Tensor::from_slice(&batch.iter().map(|t| t.reward).collect::<Vec<_>>()); // [batch]
                let dones = Tensor::from_slice(
                    &batch
                        .iter()
                        .map(|t| if t.done { 1f32 } else { 0f32 })
                        .collect::<Vec<_>>(),
                ); // [batch]

                // Q(s, a) from policy network -> gather the Q-value for each action taken
                let q_values = policy_network
                    .forward(&states)
                    .gather(1, &actions.unsqueeze(1), false)
                    .squeeze_dim(1); // [batch]

                // Bellman targets from target network
                let next_q_values = tch::no_grad(|| {
                    target_network.forward(&next_states).max_dim(1, false).0 // [batch]
                });
                let targets = &rewards + gamma * &next_q_values * (1. - &dones);

                // MSE loss and backprop
                let loss = q_values.mse_loss(&targets, tch::Reduction::Mean);
                opt.zero_grad();
                loss.backward();
                // Clip gradients to max norm of 1.0
                policy_vs.trainable_variables().iter().for_each(|v| {
                    let _ = v.grad().clamp_(-1.0, 1.0);
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
                episode_loss += f32::try_from(&loss).expect("bruh");
                loss_steps += 1;
            }

            // Update tracking
            last_100_rewards[last_reward_idx % 100] = episode_reward;
            last_reward_idx += 1;

            // 7. If done: reset environment (new scramble)
            if done {
                break;
            }
        }

        // Logging
        println!(
            "Episode {:3}/{:3} | scramble depth: {:1} | reward: {:.2} | loss: {:.4} | epsilon: {:.3}",
            episode + 1,
            episodes,
            scramble_depth,
            episode_reward,
            if loss_steps > 0 {
                episode_loss / loss_steps as f32
            } else {
                0.
            },
            epsilon
        );

        // Save to file
        policy_vs
            .save("policy.ot")
            .expect("Failure to save policy net weights");
    }

    println!(
        "Finished training in {:.3}ms",
        (Instant::now() - start_time).as_millis()
    );
    Result::Ok(())
}

/// Generates a one-hot encoding for the given cube of dimensions
/// faces * width * height * colour
fn encode_cube(cube: &Cube) -> Tensor {
    let mut data = Vec::with_capacity(INPUT_SIZE);

    for face in cube.get_faces() {
        for row in 0..CUBE_SIZE {
            for col in 0..CUBE_SIZE {
                let colour_idx = face.get_tile_colour(row, col) as usize;
                for c in 0..6 {
                    data.push(if c == colour_idx { 1.0f32 } else { 0.0 });
                }
            }
        }
    }

    Tensor::from_slice(&data) // shape [324]
}

fn initialize_network(vs: &nn::Path) -> impl Module {
    nn::seq()
        .add(nn::linear(
            vs / "layer1",
            INPUT_SIZE as i64,
            256,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs / "layer2", 256, 128, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(
            vs / "layer3",
            128,
            OUTPUT_SIZE as i64,
            Default::default(),
        ))
}

fn calculate_reward(cube: &Cube) -> f32 {
    if cube.is_solved() { 10. } else { 0. }
}

struct CubeEnv {
    cube: Cube,
    max_steps: usize,
    steps: usize,
}

impl CubeEnv {
    /// Initializes a new environment with a new unscrambled cube
    fn new(max_steps: usize) -> Self {
        CubeEnv {
            cube: Cube::default(),
            max_steps,
            steps: 0,
        }
    }

    /// Scrambles this environment's cube and returns the associated state
    fn reset(&mut self, moves: usize) -> Tensor {
        self.cube = Cube::default();
        self.cube.scramble(moves, rubiks::ScrambleType::Random);
        self.steps = 0;
        encode_cube(&self.cube)
    }

    /// Apply a turn, returns (next_state, reward, done)
    fn step(&mut self, action: usize) -> (Tensor, f32, bool) {
        let turn = CubeEnv::map_action(action);
        self.cube.make_turn(turn);
        self.steps += 1;
        (
            encode_cube(&self.cube),
            calculate_reward(&self.cube),
            self.cube.is_solved() || self.steps >= self.max_steps,
        )
    }

    /// Maps an action to a turn that can be applied to the cube
    fn map_action(action: usize) -> Turn {
        // Since action is in [0, 11], this will allow us to map it to 3 groups of 6
        let ft = match action / 3 {
            0 => FaceType::Top,
            1 => FaceType::Bottom,
            2 => FaceType::Front,
            3 => FaceType::Back,
            4 => FaceType::Left,
            _ => FaceType::Right,
        };
        let tt = match action % 3 {
            0 => TurnType::Clockwise,
            1 => TurnType::CounterClockwise,
            _ => TurnType::Half,
        };
        Turn::new(ft, tt)
    }
}

struct ReplayBuffer {
    capacity: usize,
    transitions: Vec<Transition>,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        ReplayBuffer {
            capacity,
            transitions: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, transition: Transition) {
        if self.transitions.len() < self.capacity {
            self.transitions.push(transition);
        } else {
            // Overwrite oldest transitions when the buffer is full
            let idx = self.transitions.len() % self.capacity;
            self.transitions[idx] = transition;
        }
    }

    pub fn sample(&self, batch_size: usize) -> Vec<&Transition> {
        self.transitions
            .sample(&mut rand::rng(), batch_size)
            .collect()
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }
}

struct Transition {
    state: Tensor,
    action: usize,
    reward: f32,
    next_state: Tensor,
    done: bool,
}

impl Transition {
    pub fn new(
        state: &Tensor,
        action: usize,
        reward: f32,
        next_state: &Tensor,
        done: bool,
    ) -> Self {
        Transition {
            state: state.shallow_clone(),
            action,
            reward,
            next_state: next_state.shallow_clone(),
            done,
        }
    }
}
