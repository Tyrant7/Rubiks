use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::IteratorRandom;

use crate::{
    face::{Face, FaceType},
    turn::{Turn, TurnType},
};

/// Represents a Rubik's cube as an array of six [`Face`]s.
#[derive(PartialEq, Clone, Debug)]
pub struct Cube<const SIZE: usize> {
    /// Indexed by [`FaceType`] cast to `usize`.
    faces: [Face<SIZE>; 6],
}

/// Controls how the random number generator is seeded during a scramble.
#[derive(Clone, Copy, Debug)]
pub enum ScrambleType {
    /// Uses a non-deterministic random source.
    Random,
    /// Uses a fixed seed, producing the same scramble every time.
    Seeded(u64),
}

impl<const SIZE: usize> Default for Cube<SIZE> {
    /// Creates a new solved cube with each face set to its default colour.
    fn default() -> Self {
        Self {
            faces: [
                Face::new(FaceType::Top::<SIZE>.get_solved_colour()),
                Face::new(FaceType::Bottom::<SIZE>.get_solved_colour()),
                Face::new(FaceType::Front::<SIZE>.get_solved_colour()),
                Face::new(FaceType::Back::<SIZE>.get_solved_colour()),
                Face::new(FaceType::Left::<SIZE>.get_solved_colour()),
                Face::new(FaceType::Right::<SIZE>.get_solved_colour()),
            ],
        }
    }
}

impl<const SIZE: usize> Cube<SIZE> {
    /// Scrambles the cube by applying a number of random turns.
    /// Use [`ScrambleType::Seeded`] for a reproducible scramble.
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

        // Make one turn if we accidentally unscrambled the cube (especially likely at lower depth scrambles)
        if self.is_solved() {
            self.make_turn(Turn::new(
                *faces.iter().choose(&mut rng).unwrap(),
                *turn_types.iter().choose(&mut rng).unwrap(),
            ));
        }
    }

    /// Applies a single turn to the cube, rotating the face and cycling
    /// the edges of all adjacent faces.
    pub fn make_turn(&mut self, turn: Turn<SIZE>) {
        self.get_face_mut(turn.face_type).make_turn(turn.turn_type);
        self.cycle_edges(turn);
    }

    /// Cycles the edges of the four faces adjacent to the turned face.
    fn cycle_edges(&mut self, turn: Turn<SIZE>) {
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

    /// Returns a reference to the face with the given [`FaceType`].
    pub fn get_face(&self, face_type: FaceType<SIZE>) -> &Face<SIZE> {
        &self.faces[face_type as usize]
    }

    /// Returns a mutable reference to the face with the given [`FaceType`].
    fn get_face_mut(&mut self, face_type: FaceType<SIZE>) -> &mut Face<SIZE> {
        &mut self.faces[face_type as usize]
    }

    /// Returns a reference to the faces of this cube in consistent order.
    pub fn get_faces(&self) -> &[Face<SIZE>; 6] {
        &self.faces
    }

    /// Returns true if all faces are solved, i.e. each face is a single colour.
    pub fn is_solved(&self) -> bool {
        self.faces.iter().all(|x| x.is_solved())
    }

    /// Returns the number of solved faces on this cube.
    pub fn count_solved_faces(&self) -> usize {
        self.faces.iter().filter(|&x| x.is_solved()).count()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn new_cube() {
        let cube = Cube::<3>::default();

        // Ensure all faces are unique
        let mut set = HashSet::new();
        assert!(cube.faces.iter().all(|x| set.insert(x)));

        // Ensure all faces are solved
        assert!(cube.faces.iter().all(|x| x.is_solved()));
    }

    #[test]
    fn scramble_not_solved() {
        let mut cube = Cube::<3>::default();
        cube.scramble(30, ScrambleType::Seeded(42));
        assert!(!cube.is_solved());
    }

    #[test]
    fn scramble_not_solved_2x2() {
        let mut cube = Cube::<2>::default();
        cube.scramble(30, ScrambleType::Seeded(42));
        assert!(!cube.is_solved());
    }

    #[test]
    fn scramble_seeded_deterministic() {
        let mut cube_a = Cube::<3>::default();
        let mut cube_b = Cube::default();
        cube_a.scramble(30, ScrambleType::Seeded(20));
        cube_b.scramble(30, ScrambleType::Seeded(20));
        assert_eq!(cube_a, cube_b);
    }

    #[test]
    fn scramble_different_seeds_differ() {
        let mut cube_a = Cube::<3>::default();
        let mut cube_b = Cube::default();
        cube_a.scramble(30, ScrambleType::Seeded(50));
        cube_b.scramble(30, ScrambleType::Seeded(51));
        assert_ne!(cube_a, cube_b);
    }

    #[test]
    fn make_turn_clockwise() {
        const CUBE_SIZE: usize = 3;
        let mut cube = Cube::default();
        cube.make_turn(Turn::new(FaceType::Bottom, TurnType::Clockwise));

        assert_eq!(
            cube.get_face(FaceType::Front).get_row(2),
            [FaceType::<CUBE_SIZE>::Left.get_solved_colour(); CUBE_SIZE]
        );
        assert_eq!(
            cube.get_face(FaceType::Left).get_row(2),
            [FaceType::<CUBE_SIZE>::Back.get_solved_colour(); CUBE_SIZE]
        );
        assert_eq!(
            cube.get_face(FaceType::Back).get_row(2),
            [FaceType::<CUBE_SIZE>::Right.get_solved_colour(); CUBE_SIZE]
        );
        assert_eq!(
            cube.get_face(FaceType::Right).get_row(2),
            [FaceType::<CUBE_SIZE>::Front.get_solved_colour(); CUBE_SIZE]
        );
    }

    #[test]
    fn make_turn_counterclockwise() {
        const CUBE_SIZE: usize = 3;
        let mut cube = Cube::default();
        cube.make_turn(Turn::new(FaceType::Bottom, TurnType::CounterClockwise));

        assert_eq!(
            cube.get_face(FaceType::Front).get_row(2),
            [FaceType::<CUBE_SIZE>::Right.get_solved_colour(); CUBE_SIZE]
        );
        assert_eq!(
            cube.get_face(FaceType::Right).get_row(2),
            [FaceType::<CUBE_SIZE>::Back.get_solved_colour(); CUBE_SIZE]
        );
        assert_eq!(
            cube.get_face(FaceType::Back).get_row(2),
            [FaceType::<CUBE_SIZE>::Left.get_solved_colour(); CUBE_SIZE]
        );
        assert_eq!(
            cube.get_face(FaceType::Left).get_row(2),
            [FaceType::<CUBE_SIZE>::Front.get_solved_colour(); CUBE_SIZE]
        );
    }

    #[test]
    fn make_turn_half() {
        const CUBE_SIZE: usize = 3;
        let mut cube = Cube::default();
        cube.make_turn(Turn::new(FaceType::Bottom, TurnType::Half));

        assert_eq!(
            cube.get_face(FaceType::Front).get_row(2),
            [FaceType::<CUBE_SIZE>::Back.get_solved_colour(); CUBE_SIZE]
        );
        assert_eq!(
            cube.get_face(FaceType::Right).get_row(2),
            [FaceType::<CUBE_SIZE>::Left.get_solved_colour(); CUBE_SIZE]
        );
        assert_eq!(
            cube.get_face(FaceType::Back).get_row(2),
            [FaceType::<CUBE_SIZE>::Front.get_solved_colour(); CUBE_SIZE]
        );
        assert_eq!(
            cube.get_face(FaceType::Left).get_row(2),
            [FaceType::<CUBE_SIZE>::Right.get_solved_colour(); CUBE_SIZE]
        );
    }

    #[test]
    fn four_clockwise_returns_to_solved() {
        let mut cube = Cube::<3>::default();
        for _ in 0..4 {
            cube.make_turn(Turn::new(FaceType::Right, TurnType::Clockwise));
        }
        assert!(cube.is_solved());
    }

    #[test]
    fn make_turn_clockwise_2x2() {
        const CUBE_SIZE: usize = 2;
        let mut cube = Cube::default();
        cube.make_turn(Turn::new(FaceType::Bottom, TurnType::Clockwise));

        assert_eq!(
            cube.get_face(FaceType::Front).get_row(1),
            [FaceType::<CUBE_SIZE>::Left.get_solved_colour(); CUBE_SIZE]
        );
        assert_eq!(
            cube.get_face(FaceType::Left).get_row(1),
            [FaceType::<CUBE_SIZE>::Back.get_solved_colour(); CUBE_SIZE]
        );
        assert_eq!(
            cube.get_face(FaceType::Back).get_row(1),
            [FaceType::<CUBE_SIZE>::Right.get_solved_colour(); CUBE_SIZE]
        );
        assert_eq!(
            cube.get_face(FaceType::Right).get_row(1),
            [FaceType::<CUBE_SIZE>::Front.get_solved_colour(); CUBE_SIZE]
        );
    }

    #[test]
    fn make_turn_counterclockwise_2x2() {
        const CUBE_SIZE: usize = 2;
        let mut cube = Cube::default();
        cube.make_turn(Turn::new(FaceType::Bottom, TurnType::CounterClockwise));

        assert_eq!(
            cube.get_face(FaceType::Front).get_row(1),
            [FaceType::<CUBE_SIZE>::Right.get_solved_colour(); CUBE_SIZE]
        );
        assert_eq!(
            cube.get_face(FaceType::Right).get_row(1),
            [FaceType::<CUBE_SIZE>::Back.get_solved_colour(); CUBE_SIZE]
        );
        assert_eq!(
            cube.get_face(FaceType::Back).get_row(1),
            [FaceType::<CUBE_SIZE>::Left.get_solved_colour(); CUBE_SIZE]
        );
        assert_eq!(
            cube.get_face(FaceType::Left).get_row(1),
            [FaceType::<CUBE_SIZE>::Front.get_solved_colour(); CUBE_SIZE]
        );
    }

    #[test]
    fn make_turn_half_2x2() {
        const CUBE_SIZE: usize = 2;
        let mut cube = Cube::default();
        cube.make_turn(Turn::new(FaceType::Bottom, TurnType::Half));

        assert_eq!(
            cube.get_face(FaceType::Front).get_row(1),
            [FaceType::<CUBE_SIZE>::Back.get_solved_colour(); CUBE_SIZE]
        );
        assert_eq!(
            cube.get_face(FaceType::Right).get_row(1),
            [FaceType::<CUBE_SIZE>::Left.get_solved_colour(); CUBE_SIZE]
        );
        assert_eq!(
            cube.get_face(FaceType::Back).get_row(1),
            [FaceType::<CUBE_SIZE>::Front.get_solved_colour(); CUBE_SIZE]
        );
        assert_eq!(
            cube.get_face(FaceType::Left).get_row(1),
            [FaceType::<CUBE_SIZE>::Right.get_solved_colour(); CUBE_SIZE]
        );
    }

    #[test]
    fn four_clockwise_returns_to_solved_2x2() {
        let mut cube = Cube::<2>::default();
        for _ in 0..4 {
            cube.make_turn(Turn::new(FaceType::Right, TurnType::Clockwise));
        }
        assert!(cube.is_solved());
    }

    #[test]
    fn make_turns_reversible() {
        let mut cube = Cube::<3>::default();

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
    fn make_turns_reversible_2x2() {
        let mut cube = Cube::<2>::default();

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
        let cube = Cube::<3>::default();
        assert!(cube.is_solved());
    }

    #[test]
    fn is_solved_false() {
        let mut cube = Cube::<3>::default();
        cube.get_face_mut(FaceType::Back)
            .set_tile_colour(0, 0, crate::face::Colour::Yellow);
        assert!(!cube.is_solved());
    }

    #[test]
    fn count_solved_all() {
        let cube = Cube::<3>::default();
        assert_eq!(cube.count_solved_faces(), 6);
    }

    #[test]
    fn count_solved_after_one_move() {
        let mut cube = Cube::<3>::default();
        cube.make_turn(Turn::new(FaceType::Bottom, TurnType::Clockwise));
        assert_eq!(cube.count_solved_faces(), 2);
    }
}
