use rand::seq::IndexedRandom;
use rubiks::{CUBE_SIZE, Cube, FaceType, Turn, TurnType};
use tch::Tensor;

use crate::INPUT_SIZE;

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

/// Calculates reward for the current cube based on correctly placed
/// tile counts and whether or not the cube is solved
fn calculate_reward(cube: &Cube) -> f32 {
    if cube.is_solved() {
        return 1.0;
    }

    let faces = [
        FaceType::Top,
        FaceType::Bottom,
        FaceType::Front,
        FaceType::Back,
        FaceType::Left,
        FaceType::Right,
    ];

    let correct_tiles: f32 = faces
        .iter()
        .map(|&ft| {
            let face = cube.get_face(ft);
            let solved_colour = ft.get_solved_colour();
            (0..CUBE_SIZE)
                .flat_map(|r| (0..CUBE_SIZE).map(move |c| (r, c)))
                .filter(|&(r, c)| face.get_tile_colour(r, c) == solved_colour)
                .count() as f32
        })
        .sum();

    // Normalised to [-1, 1] range roughly
    (correct_tiles / 54.0) * 0.1 - 0.1
}

pub struct CubeEnv {
    cube: Cube,
    max_steps: usize,
    steps: usize,
}

impl CubeEnv {
    /// Initializes a new environment with a new unscrambled cube
    pub fn new() -> Self {
        CubeEnv {
            cube: Cube::default(),
            max_steps: 0,
            steps: 0,
        }
    }

    /// Scrambles this environment's cube and returns the associated state
    pub fn reset(&mut self, moves: usize, max_steps: usize) -> Tensor {
        self.cube = Cube::default();
        self.cube.scramble(moves, rubiks::ScrambleType::Random);
        self.steps = 0;
        self.max_steps = max_steps;
        encode_cube(&self.cube)
    }

    /// Apply a turn, returns (next_state, reward, done)
    pub fn step(&mut self, action: usize) -> (Tensor, f32, bool) {
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

    /// Cube getter
    pub fn get_cube(&self) -> &Cube {
        &self.cube
    }
}

pub struct ReplayBuffer {
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

pub struct Transition {
    pub state: Tensor,
    pub action: usize,
    pub reward: f32,
    pub next_state: Tensor,
    pub done: bool,
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
