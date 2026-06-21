use rand::seq::IndexedRandom;
use rubiks::{Cube, FaceType, Turn, TurnType};
use tch::Tensor;

use crate::{CUBE_SIZE, INPUT_SIZE, get_device};

/// Returns a vector of all colours on the given cube in order
fn encode_cube_indices(cube: &Cube<CUBE_SIZE>) -> Tensor {
    let mut data = Vec::with_capacity(INPUT_SIZE);

    for face in cube.get_faces() {
        for row in 0..CUBE_SIZE {
            for col in 0..CUBE_SIZE {
                data.push(face.get_tile_colour(row, col) as i64);
            }
        }
    }

    Tensor::from_slice(&data).to_device(get_device())
}

fn count_correct_facelets(cube: &Cube<CUBE_SIZE>) -> usize {
    let faces = [
        FaceType::Top,
        FaceType::Bottom,
        FaceType::Front,
        FaceType::Back,
        FaceType::Left,
        FaceType::Right,
    ];

    faces
        .iter()
        .map(|&ft| {
            // Score dominant tile colour on each face
            // May potentially have issue of scoring same colour multiple times across faces, but is unlikely
            let face = cube.get_face(ft);
            let mut colour_counts = [0; 6];
            (0..CUBE_SIZE)
                .flat_map(|r| (0..CUBE_SIZE).map(move |c| (r, c)))
                .for_each(|(r, c)| colour_counts[face.get_tile_colour(r, c) as usize] += 1);

            *colour_counts.iter().max().unwrap()
        })
        .sum()
}

/// Calculates reward for the current cube based on correctly placed
/// facelet counts and whether or not the cube is solved.
fn calculate_reward(cube: &Cube<CUBE_SIZE>) -> f32 {
    if cube.is_solved() {
        return 1.0;
    }

    const FACELETS: usize = CUBE_SIZE * CUBE_SIZE * 6;
    (count_correct_facelets(cube) as f32 / FACELETS as f32) * 0.1 - 0.1
}

pub struct CubeEnv {
    cube: Cube<CUBE_SIZE>,
    max_steps: usize,
    steps: usize,
}

pub struct StepResult {
    pub next_state: Tensor,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
}

impl CubeEnv {
    /// Initializes a new environment with a new unscrambled cube
    pub fn new() -> Self {
        CubeEnv {
            cube: Cube::<CUBE_SIZE>::default(),
            max_steps: 0,
            steps: 0,
        }
    }

    /// Performs a seeded scramble on this environment's cube and returns the associated state
    pub fn seeded_reset(&mut self, moves: usize, max_steps: usize, seed: u64) -> Tensor {
        self.cube = Cube::default();
        self.cube
            .scramble(moves, rubiks::ScrambleType::Seeded(seed));
        self.steps = 0;
        self.max_steps = max_steps;
        encode_cube_indices(&self.cube)
    }

    /// Scrambles this environment's cube and returns the associated state
    pub fn reset(&mut self, moves: usize, max_steps: usize) -> Tensor {
        self.cube = Cube::default();
        self.cube.scramble(moves, rubiks::ScrambleType::Random);
        self.steps = 0;
        self.max_steps = max_steps;
        encode_cube_indices(&self.cube)
    }

    /// Apply a turn and report true terminals separately from time-limit truncation.
    pub fn step(&mut self, action: usize) -> StepResult {
        let turn = CubeEnv::map_action(action);
        self.cube.make_turn(turn);
        self.steps += 1;
        let terminated = self.cube.is_solved();
        StepResult {
            next_state: encode_cube_indices(&self.cube),
            reward: calculate_reward(&self.cube),
            terminated,
            truncated: !terminated && self.steps >= self.max_steps,
        }
    }

    /// Maps an action to a turn that can be applied to the cube
    fn map_action(action: usize) -> Turn<CUBE_SIZE> {
        // Since action is in [0, 18), this will allow us to map it to 3 groups of 6
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
    pub fn get_cube(&self) -> &Cube<CUBE_SIZE> {
        &self.cube
    }
}

pub struct ReplayBuffer {
    capacity: usize,
    insertions: usize,
    transitions: Vec<Transition>,
}

pub struct SampleBatch {
    pub states: Tensor,
    pub actions: Tensor,
    pub rewards: Tensor,
    pub next_states: Tensor,
    pub terminated: Tensor,
    pub truncated: Tensor,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        ReplayBuffer {
            capacity,
            insertions: 0,
            transitions: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, transition: Transition) {
        if self.transitions.len() < self.capacity {
            self.transitions.push(transition);
        } else {
            // Overwrite oldest transitions when the buffer is full
            self.transitions[self.insertions] = transition;
        }
        self.insertions += 1;
        self.insertions %= self.capacity;
    }

    pub fn sample(&self, batch_size: usize) -> Vec<&Transition> {
        self.transitions
            .sample(&mut rand::rng(), batch_size)
            .collect()
    }

    pub fn sample_tensors(&self, batch_size: usize) -> SampleBatch {
        let batch = self.sample(batch_size);

        SampleBatch {
            states: Tensor::stack(
                &batch
                    .iter()
                    .map(|t| t.state.shallow_clone())
                    .collect::<Vec<_>>(),
                0,
            ),
            actions: Tensor::from_slice(&batch.iter().map(|t| t.action as i64).collect::<Vec<_>>())
                .to_device(get_device()),
            rewards: Tensor::from_slice(&batch.iter().map(|t| t.reward).collect::<Vec<_>>())
                .to_device(get_device()),
            next_states: Tensor::stack(
                &batch
                    .iter()
                    .map(|t| t.next_state.shallow_clone())
                    .collect::<Vec<_>>(),
                0,
            ),
            terminated: Tensor::from_slice(
                &batch
                    .iter()
                    .map(|t| if t.terminated { 1f32 } else { 0f32 })
                    .collect::<Vec<_>>(),
            )
            .to_device(get_device()),
            truncated: Tensor::from_slice(
                &batch
                    .iter()
                    .map(|t| if t.truncated { 1f32 } else { 0f32 })
                    .collect::<Vec<_>>(),
            )
            .to_device(get_device()),
        }
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    pub fn clear(&mut self) {
        self.transitions.clear();
        self.insertions = 0;
    }
}

pub struct Transition {
    pub state: Tensor,
    pub action: usize,
    pub reward: f32,
    pub next_state: Tensor,
    pub terminated: bool,
    pub truncated: bool,
}

impl Transition {
    pub fn new(
        state: &Tensor,
        action: usize,
        reward: f32,
        next_state: &Tensor,
        terminated: bool,
        truncated: bool,
    ) -> Self {
        Transition {
            state: state.shallow_clone(),
            action,
            reward,
            next_state: next_state.shallow_clone(),
            terminated,
            truncated,
        }
    }
}
