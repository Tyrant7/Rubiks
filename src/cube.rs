use crate::face::Face;

pub struct Cube {
    faces: [Face; 6],
}

pub struct Turn {
    face_index: usize,
    turn_type: TurnType,
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

pub enum TurnType {
    Clockwise,
    CounterClockwise,
    Half,
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

impl Turn {
    pub fn new(face_index: usize, turn_type: TurnType) -> Self {
        unimplemented!()
    }
}
