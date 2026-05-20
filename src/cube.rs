struct Cube {
    faces: [Face; 6],
}

struct Face {
    tiles: [[u8; 3]; 3],
}

struct Turn {
    face_type: FaceType,
    turn_type: TurnType,
}

enum Colour {
    White,
    Red,
    Green,
    Blue,
    Orange,
    Yellow,
}

enum FaceType {
    Top,
    Bottom,
    Front,
    Back,
    Left,
    Right,
}

enum TurnType {
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
}

impl Face {
    pub fn new(face_type: FaceType) -> Self {
        unimplemented!()
    }

    pub fn get_tile_colour(&self, row: usize, col: usize) -> Colour {
        unimplemented!()
    }

    pub fn make_turn(&mut self, turn: TurnType) {
        unimplemented!()
    }
}

impl Turn {
    pub fn new(face_type: FaceType, turn_type: TurnType) -> Self {
        unimplemented!()
    }
}
