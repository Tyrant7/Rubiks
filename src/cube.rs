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
    Seeded(u128),
}

impl Cube {
    pub fn new() -> Self {
        unimplemented!()
    }

    pub fn scramble(&mut self, moves: usize, scramble_type: ScrambleType) {
        unimplemented!()
    }

    pub fn make_turn(&mut self, turn: Turn) {
        unimplemented!()
    }

    pub fn get_face(&self, face_type: FaceType) -> &Face {
        unimplemented!()
    }

    pub fn get_face_mut(&mut self, face_type: FaceType) -> &mut Face {
        unimplemented!()
    }

    pub fn is_solved(&self) -> bool {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::face::Colour;

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
        cube.make_turn(Turn::new(FaceType::Right, TurnType::Half));
        assert!(!cube.is_solved());
    }
}
