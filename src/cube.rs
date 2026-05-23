use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::IteratorRandom;

use crate::{
    face::{Face, FaceType},
    turn::{Turn, TurnType},
};

#[derive(PartialEq, Debug)]
pub struct Cube {
    faces: [Face; 6],
}

pub enum ScrambleType {
    Random,
    Seeded(u64),
}

impl Cube {
    pub fn new() -> Self {
        Cube {
            faces: [
                Face::new(FaceType::Top.get_solved_colour()),
                Face::new(FaceType::Bottom.get_solved_colour()),
                Face::new(FaceType::Front.get_solved_colour()),
                Face::new(FaceType::Back.get_solved_colour()),
                Face::new(FaceType::Left.get_solved_colour()),
                Face::new(FaceType::Right.get_solved_colour()),
            ],
        }
    }

    pub fn scramble(&mut self, moves: usize, scramble_type: ScrambleType) {
        let mut rng: Box<dyn rand::Rng> = match scramble_type {
            ScrambleType::Seeded(seed) => Box::new(StdRng::seed_from_u64(seed)),
            ScrambleType::Random => Box::new(rand::rng()),
        };
        let faces = [
            FaceType::Top,
            FaceType::Bottom,
            FaceType::Front,
            FaceType::Back,
            FaceType::Left,
            FaceType::Right,
        ];
        let turn_types = [
            TurnType::Clockwise,
            TurnType::CounterClockwise,
            TurnType::Half,
        ];
        for _ in 0..moves {
            self.make_turn(Turn::new(
                *faces.iter().choose(&mut rng).unwrap(),
                *turn_types.iter().choose(&mut rng).unwrap(),
            ));
        }
    }

    pub fn make_turn(&mut self, turn: Turn) {
        self.get_face_mut(turn.face_type).make_turn(turn.turn_type);
        self.cycle_edges(turn);
    }

    fn cycle_edges(&mut self, turn: Turn) {
        let edges = turn.face_type.get_edges();

        // Read all 4 edges first
        let mut data: [_; 4] = std::array::from_fn(|i| {
            let e = &edges[i];
            let mut tiles = if e.is_row {
                self.get_face(e.face).get_row(e.index)
            } else {
                self.get_face(e.face).get_col(e.index)
            };
            if e.reversed {
                tiles.reverse();
            }
            tiles
        });

        // Rotate the data array by the cycle amount
        let shift = match turn.turn_type {
            TurnType::Clockwise => 3, // rotate right by 1 = left by 3
            TurnType::CounterClockwise => 1,
            TurnType::Half => 2,
        };
        data.rotate_left(shift);

        // Write back
        for (i, e) in edges.iter().enumerate() {
            let mut tiles = data[i];
            if e.reversed {
                tiles.reverse();
            }
            if e.is_row {
                self.get_face_mut(e.face).set_row(e.index, tiles);
            } else {
                self.get_face_mut(e.face).set_col(e.index, tiles);
            }
        }
    }

    pub fn get_face(&self, face_type: FaceType) -> &Face {
        &self.faces[face_type as usize]
    }

    fn get_face_mut(&mut self, face_type: FaceType) -> &mut Face {
        &mut self.faces[face_type as usize]
    }

    pub fn is_solved(&self) -> bool {
        self.faces.iter().all(|x| x.is_solved())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn new_cube() {
        let cube = Cube::new();

        // Ensure all faces are unique
        let mut set = HashSet::new();
        assert!(cube.faces.iter().all(|x| set.insert(x)));

        // Ensure all faces are solved
        assert!(cube.faces.iter().all(|x| x.is_solved()));
    }

    #[test]
    fn scramble_not_solved() {
        let mut cube = Cube::new();
        cube.scramble(20, ScrambleType::Seeded(42));
        assert!(!cube.is_solved());
    }

    #[test]
    fn scramble_seeded_deterministic() {
        let mut cube_a = Cube::new();
        let mut cube_b = Cube::new();
        cube_a.scramble(20, ScrambleType::Seeded(20));
        cube_b.scramble(20, ScrambleType::Seeded(20));
        assert_eq!(cube_a, cube_b);
    }

    #[test]
    fn scramble_different_seeds_differ() {
        let mut cube_a = Cube::new();
        let mut cube_b = Cube::new();
        cube_a.scramble(20, ScrambleType::Seeded(50));
        cube_b.scramble(20, ScrambleType::Seeded(51));
        assert_ne!(cube_a, cube_b);
    }

    #[test]
    fn make_turn_clockwise() {
        let mut cube = Cube::new();
        cube.make_turn(Turn::new(FaceType::Bottom, TurnType::Clockwise));

        // Check adjacent faces to the turned one
        assert_eq!(
            cube.get_face(FaceType::Front).get_tile_colour(2, 0),
            FaceType::Left.get_solved_colour()
        );
        assert_eq!(
            cube.get_face(FaceType::Left).get_tile_colour(2, 0),
            FaceType::Back.get_solved_colour()
        );
        assert_eq!(
            cube.get_face(FaceType::Back).get_tile_colour(2, 1),
            FaceType::Right.get_solved_colour()
        );
        assert_eq!(
            cube.get_face(FaceType::Right).get_tile_colour(2, 2),
            FaceType::Front.get_solved_colour()
        );
    }

    #[test]
    fn make_turn_counterclockwise() {
        let mut cube = Cube::new();
        cube.make_turn(Turn::new(FaceType::Bottom, TurnType::CounterClockwise));

        // Check one tile on each of the adjacent faces to the turned one
        assert_eq!(
            cube.get_face(FaceType::Front).get_tile_colour(2, 0),
            FaceType::Right.get_solved_colour()
        );
        assert_eq!(
            cube.get_face(FaceType::Right).get_tile_colour(2, 0),
            FaceType::Back.get_solved_colour()
        );
        assert_eq!(
            cube.get_face(FaceType::Back).get_tile_colour(2, 1),
            FaceType::Left.get_solved_colour()
        );
        assert_eq!(
            cube.get_face(FaceType::Left).get_tile_colour(2, 2),
            FaceType::Front.get_solved_colour()
        );
    }

    #[test]
    fn make_turn_half() {
        let mut cube = Cube::new();
        cube.make_turn(Turn::new(FaceType::Bottom, TurnType::Half));

        // Check one tile on each of the adjacent faces to the turned one
        assert_eq!(
            cube.get_face(FaceType::Front).get_tile_colour(2, 0),
            FaceType::Back.get_solved_colour()
        );
        assert_eq!(
            cube.get_face(FaceType::Right).get_tile_colour(2, 0),
            FaceType::Left.get_solved_colour()
        );
        assert_eq!(
            cube.get_face(FaceType::Back).get_tile_colour(2, 1),
            FaceType::Front.get_solved_colour()
        );
        assert_eq!(
            cube.get_face(FaceType::Left).get_tile_colour(2, 2),
            FaceType::Right.get_solved_colour()
        );
    }

    #[test]
    fn make_turns_reversible() {
        let mut cube = Cube::new();

        // Turn in
        cube.make_turn(Turn::new(FaceType::Bottom, TurnType::CounterClockwise));
        cube.make_turn(Turn::new(FaceType::Top, TurnType::Clockwise));
        cube.make_turn(Turn::new(FaceType::Right, TurnType::Half));
        cube.make_turn(Turn::new(FaceType::Left, TurnType::Clockwise));
        cube.make_turn(Turn::new(FaceType::Front, TurnType::Clockwise));
        cube.make_turn(Turn::new(FaceType::Back, TurnType::Half));

        // Turn out
        cube.make_turn(Turn::new(FaceType::Back, TurnType::Half));
        cube.make_turn(Turn::new(FaceType::Front, TurnType::CounterClockwise));
        cube.make_turn(Turn::new(FaceType::Left, TurnType::CounterClockwise));
        cube.make_turn(Turn::new(FaceType::Right, TurnType::Half));
        cube.make_turn(Turn::new(FaceType::Top, TurnType::CounterClockwise));
        cube.make_turn(Turn::new(FaceType::Bottom, TurnType::Clockwise));

        assert!(cube.is_solved());
    }

    #[test]
    fn is_solved_true() {
        let cube = Cube::new();
        assert!(cube.is_solved());
    }

    #[test]
    fn is_solved_false() {
        let mut cube = Cube::new();
        cube.get_face_mut(FaceType::Back)
            .set_tile_colour(0, 0, crate::face::Colour::Yellow);
        assert!(!cube.is_solved());
    }
}
