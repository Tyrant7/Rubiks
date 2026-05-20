const CUBE_SIZE: usize = 3;
const FACE_COLOURS: [Colour; 6] = [White, Red, Green, Blue, Orange, Yellow];

struct Cube {
    faces: [Face; 6],
}

struct Face {
    tiles: [[u8; CUBE_SIZE]; CUBE_SIZE],
}

struct Turn {
    face_index: usize,
    turn_type: TurnType,
}

#[derive(PartialEq, Debug, Clone, Copy)]
enum Colour {
    White,
    Red,
    Green,
    Blue,
    Orange,
    Yellow,
}

#[derive(PartialEq, Debug, Clone, Copy)]
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
    pub fn new(face_type: Colour) -> Self {
        unimplemented!()
    }

    pub fn make_turn(&mut self, turn: TurnType) {
        unimplemented!()
    }

    pub fn get_tile_colour(&self, row: usize, col: usize) -> Colour {
        unimplemented!()
    }

    pub fn is_solved(&self) -> bool {
        unimplemented!()
    }

    fn get_tile_raw(&self, row: usize, col: usize) -> u8 {
        unimplemented!()
    }
}

impl Turn {
    pub fn new(face_index: usize, turn_type: TurnType) -> Self {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn face_new() {
        let face = Face::new(Colour::Green);
        for i in 0..CUBE_SIZE {
            for j in 0..CUBE_SIZE {
                assert_eq!(face.get_tile_colour(i, j), Colour::Green);
            }
        }
    }

    #[test]
    fn cube_new() {}
}
