use rubiks::{CUBE_SIZE, Cube};
use tch::{
    Device, Tensor,
    nn::{self, Module},
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
    let epsilon = 0.9;
    let epsilon_decay = 0.5;
    let gamma = 0.99;

    // Initialize models
    let vs = nn::VarStore::new(Device::Cpu);
    let policy_network = initialize_network(vs / "policy_net");
    let target_network = initialize_network(vs / "target_net");

    // Train loop
    for i in 0..epochs {}

    /*
    1. Encode current state
    2. ε-greedy action selection
         - with prob ε: random action
         - with prob 1-ε: argmax over Q(s, ·)
    3. Step environment → (next_state, reward, done)
    4. Push transition to replay buffer
    5. If buffer large enough:
         a. Sample minibatch
         b. Compute targets:
              - if done:  target = reward
              - else:     target = reward + γ · max_a Q_target(next_state, a)
         c. Compute loss: MSE(Q_policy(state, action), target)
         d. Backprop + optimiser step
    6. If done: reset environment (new scramble)
    7. Every N steps: copy policy weights → target network
    8. Decay ε
    */
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
    // fn push(&mut self, transition: )
    // fn sample(&self, batch_size: usize) -> ...
}
