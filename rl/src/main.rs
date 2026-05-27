use rand::seq::IndexedRandom;
use rubiks::{CUBE_SIZE, Cube};
use tch::{
    Device, Tensor,
    nn::{self, Module, OptimizerConfig},
};

const INPUT_SIZE: usize = 6 * 3 * 3 * 6;
const OUTPUT_SIZE: usize = 4 * 3;

fn main() {
    // let cube = Cube::default();
    // let encoding = encode_cube(&cube);
    train();
}

fn train() {
    // Define hyperparameters
    let episodes = 1;
    let batch_size = 16;
    let update_step = 10;
    let epsilon_start = 0.9;
    let epsilon_end = 0.01;
    let epsilon_decay = 2500;
    let tau = 0.005;
    let learning_rate = 1e-3;
    let gamma = 0.99;
    let max_steps = 40;

    // Initialize models
    let policy_vs = nn::VarStore::new(Device::Cpu);
    let target_vs = nn::VarStore::new(Device::Cpu);
    let policy_vs_root = policy_vs.root();
    let target_vs_root = target_vs.root();
    let policy_network = initialize_network(&policy_vs_root);
    let target_network = initialize_network(&target_vs_root);
    let mut opt = nn::Adam::default().build(&target_vs, learning_rate);

    // Setup environment
    let mut cube_env = CubeEnv::new();
    let mut replay_buffer = ReplayBuffer::new(10000);
    let mut steps = 0;

    // Train loop
    for episode in 0..episodes {
        // 1. Encode current state
        let mut state = cube_env.reset();

        // Epsilon with exponential decay
        let epsilon = epsilon_end
            + (epsilon_start - epsilon_end) * f32::exp(-1. * steps as f32 / epsilon_decay as f32);

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

            // 4. Push transition to replay buffer
            replay_buffer.push(Transition::new(&state, action, reward, &next_state, done));

            // Move to the next state
            state = next_state;

            // 5. If buffer large enough:
            todo!()

            // a. Sample minibatch
            // b. Compute targets:
            //  - if done:  target = reward
            //  - else:     target = reward + γ · max_a Q_target(next_state, a)
            // c. Compute loss: MSE(Q_policy(state, action), target)
            // d. Backprop + optimiser step

            // 6. If done: reset environment (new scramble)
            if done {
                break;
            }

            // 7. Every N steps: copy policy weights → target network
            todo!()
        }

        steps += 1;
    }
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

fn calculate_reward(space: &CubeEnv) -> f32 {
    unimplemented!()
}

struct CubeEnv {
    cube: Cube,
    max_moves: usize,
    steps: usize,
}

impl CubeEnv {
    fn new() -> Self {
        unimplemented!()
    }

    /// Scrambles this environment's cube and returns the associated state
    fn reset(&mut self) -> Tensor {
        unimplemented!()
    }

    /// Apply a turn, returns (next_state, reward, done)
    fn step(&mut self, action: usize) -> (Tensor, f32, bool) {
        unimplemented!()
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
        self.transitions.push(transition);
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
            state: state.clone(out),
            action,
            reward,
            next_state: next_state.clone(out),
            done,
        }
    }
}
