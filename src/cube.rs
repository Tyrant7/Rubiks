use crate::{
    face::{Face, FaceType},
    turn::{Turn, TurnType},
};

pub struct Cube {
    faces: [Face; 6],
}

enum ScrambleType {
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

    fn cycle_edges(&mut self, face_type: FaceType, turn_type: TurnType) {
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
    fn scramble() {}

    #[test]
    fn make_turns_left_edge() {}

    #[test]
    fn make_turns_right_edge() {}

    #[test]
    fn make_turns_top_edge() {}

    #[test]
    fn make_turns_bottom_edge() {}

    #[test]
    fn is_solved_false() {
        let cube = Cube::new();
        assert!(cube.is_solved());
    }

    #[test]
    fn is_solved_true() {
        let mut cube = Cube::new();
        cube.get_face_mut(FaceType::Bottom)
            .set_tile_colour(0, 0, Colour::Yellow);
        assert!(!cube.is_solved());
    }
}
