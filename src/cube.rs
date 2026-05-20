use crate::{
    face::{Face, FaceType},
    turn::{Turn, TurnType},
};

pub struct Cube {
    faces: [Face; 6],
}

impl Cube {
    pub fn new() -> Self {
        unimplemented!()
    }

    pub fn scramble(&mut self, moves: usize) {
        unimplemented!()
    }

    pub fn make_turn(&mut self, turn: Turn) {
        unimplemented!()
    }

    pub fn get_face(&self, face_type: FaceType) -> &Face {
        unimplemented!()
    }

    pub fn is_solved(&self) -> bool {
        unimplemented!()
    }

    fn cycle_edges(&mut self, face_type: FaceType, turn_type: TurnType) {
        unimplemented!()
    }
}
