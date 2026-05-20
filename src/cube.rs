use crate::{
    face::Face,
    turn::{Turn, TurnType},
};

pub struct Cube {
    faces: [Face; 6],
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum Colour {
    White,
    Red,
    Green,
    Blue,
    Orange,
    Yellow,
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum FaceType {
    Top,
    Bottom,
    Front,
    Back,
    Left,
    Right,
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
