use crate::face::FaceType;

pub struct Turn {
    face_type: FaceType,
    turn_type: TurnType,
}

#[derive(Clone, Copy)]
pub enum TurnType {
    Clockwise,
    CounterClockwise,
    Half,
}

impl Turn {
    pub fn new(face_type: FaceType, turn_type: TurnType) -> Self {
        Turn {
            face_type,
            turn_type,
        }
    }
}
