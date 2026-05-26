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
    let epochs = 1000;
    let update_step = 10;
    let epsilon_start = 0.9;
    let mut epsilon = epsilon_start;
    let epsilon_end = 0.05;
    let epsilon_decay = 0.7;
    let gamma = 0.99;
    let max_steps = 40;

    // Initialize models
    let policy_vs = nn::VarStore::new(Device::Cpu);
    let target_vs = nn::VarStore::new(Device::Cpu);
    let policy_vs_root = policy_vs.root();
    let target_vs_root = target_vs.root();
    let policy_network = initialize_network(&policy_vs_root);
    let target_network = initialize_network(&target_vs_root);
    let mut opt = nn::Adam::default().build(&target_vs, 1e-3);

    // Setup environment
    let mut cube_env = CubeEnv::new();
    let mut replay_buffer = ReplayBuffer::new();

    // Train loop
    for i in 0..epochs {
        // 1. Encode current state
        let mut state = cube_env.reset();

        for _ in 0..max_steps {
            // 2. ε-greedy action selection
            let action = if rand::random::<f32>() < epsilon {
                // - with prob ε: random action
                todo!()
            } else {
                // - with prob 1-ε: argmax over Q(s, ·)
                todo!()
            };

            // 3. Step environment → (next_state, reward, done)
            let (next_state, reward, done) = cube_env.step(action);

            // 4. Push transition to replay buffer
            // replay_buffer.push()

            // 5. If buffer large enough:
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
            todo!();

            // 8. Decay ε
            todo!();

            // 9. Setup next step
            state = next_state;
        }
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
    states: Vec<Tensor>,
    actions: Vec<usize>,
    rewards: Vec<f32>,
    next_states: Vec<Tensor>,
    dones: Vec<bool>,
}

impl ReplayBuffer {
    fn new() -> Self {
        unimplemented!()
    }
    // fn push(&mut self, transition: )
    // fn sample(&self, batch_size: usize) -> ...
}
