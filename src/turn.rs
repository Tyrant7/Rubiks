pub struct Turn {
    face_index: usize,
    turn_type: TurnType,
}

pub enum TurnType {
    Clockwise,
    CounterClockwise,
    Half,
}

impl Turn {
    pub fn new(face_index: usize, turn_type: TurnType) -> Self {
        Turn {
            face_index,
            turn_type,
        }
    }
}
