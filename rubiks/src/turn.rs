use crate::face::FaceType;

/// Represents a single move on the cube, combining a face and a turn direction.
#[derive(Clone, Copy, Debug)]
pub struct Turn<const SIZE: usize> {
    /// The face to be turned.
    pub face_type: FaceType<SIZE>,
    /// The direction and magnitude of the turn.
    pub turn_type: TurnType,
}

/// Represents the direction and magnitude of a turn.
#[derive(Clone, Copy, Debug)]
pub enum TurnType {
    /// A 90 degree clockwise rotation.
    Clockwise,
    /// A 90 degree counterclockwise rotation.
    CounterClockwise,
    /// A 180 degree rotation.
    Half,
}

impl<const SIZE: usize> Turn<SIZE> {
    /// Creates a new turn with the given face and turn type.
    pub fn new(face_type: FaceType<SIZE>, turn_type: TurnType) -> Self {
        Turn {
            face_type,
            turn_type,
        }
    }
}
