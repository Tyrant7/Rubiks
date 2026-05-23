use crate::turn::TurnType;

/// Used for mapping `FaceType` representation of tiles to their respective colours
const FACE_COLOURS: [Colour; 6] = [
    Colour::White,
    Colour::Red,
    Colour::Green,
    Colour::Blue,
    Colour::Orange,
    Colour::Yellow,
];

/// The size of the cube, i.e. a standard Rubik's cube is 3x3.
pub const CUBE_SIZE: usize = 3;

/// Represents a single face of the cube as a 2D grid of tiles.
#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
pub struct Face {
    tiles: [[Colour; CUBE_SIZE]; CUBE_SIZE],
}

/// Represents the orientation of a face.
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum FaceType {
    Top,
    Bottom,
    Front,
    Back,
    Left,
    Right,
}

#[derive(Clone, Debug)]
pub struct EdgeRef {
    pub face: FaceType,
    pub index: usize,
    pub is_row: bool,
    pub reversed: bool,
}

/// Represents the colour of a face.
/// Used for mapping orientations to their visual display.
#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
pub enum Colour {
    White,
    Red,
    Green,
    Blue,
    Orange,
    Yellow,
}

impl Face {
    /// Creates a new face with all tiles set to the given colour.
    pub fn new(colour: Colour) -> Self {
        Face {
            tiles: [[colour; CUBE_SIZE]; CUBE_SIZE],
        }
    }

    /// Rotates the tiles within this face according to the given turn type.
    /// Note: this only handles the internal tile rotation and not cycling edges
    /// when this face is adjacent to the rotated face.
    pub fn make_turn(&mut self, turn: TurnType) {
        match turn {
            TurnType::Clockwise => self.rotate_clockwise(),
            TurnType::CounterClockwise => self.rotate_counterclockwise(),
            TurnType::Half => {
                self.rotate_clockwise();
                self.rotate_clockwise();
            }
        };
    }

    /// Rotates the tile grid in place 90 degrees clockwise
    fn rotate_clockwise(&mut self) {
        // 1. Reverse columns
        self.tiles.reverse();
        // 2. Transpose
        for i in 0..CUBE_SIZE {
            for j in i + 1..CUBE_SIZE {
                let temp = self.tiles[i][j];
                self.tiles[i][j] = self.tiles[j][i];
                self.tiles[j][i] = temp;
            }
        }
    }

    /// Rotates the tile grid in place 90 degrees counterclockwise
    fn rotate_counterclockwise(&mut self) {
        // 1. Transpose
        for i in 0..CUBE_SIZE {
            for j in i + 1..CUBE_SIZE {
                let temp = self.tiles[i][j];
                self.tiles[i][j] = self.tiles[j][i];
                self.tiles[j][i] = temp;
            }
        }
        // 2. Reverse columns
        self.tiles.reverse();
    }

    /// Returns the colour of the tile at the given column and row.
    pub fn get_tile_colour(&self, row: usize, col: usize) -> Colour {
        self.tiles[row][col]
    }

    /// Sets the tile at the given column and row to the given colour.
    pub fn set_tile_colour(&mut self, row: usize, col: usize, tile: Colour) {
        self.tiles[row][col] = tile;
    }

    pub fn get_row(&self, row: usize) -> [Colour; CUBE_SIZE] {
        self.tiles[row]
    }

    pub fn get_col(&self, col: usize) -> [Colour; CUBE_SIZE] {
        std::array::from_fn(|i| self.tiles[i][col])
    }

    pub fn set_row(&mut self, row: usize, data: [Colour; CUBE_SIZE]) {
        self.tiles[row] = data;
    }

    pub fn set_col(&mut self, col: usize, data: [Colour; CUBE_SIZE]) {
        for (row, d) in self.tiles.iter_mut().zip(data) {
            row[col] = d;
        }
    }

    /// Returns true if all tiles on this face are the same colour.
    pub fn is_solved(&self) -> bool {
        let first = self.get_tile_colour(0, 0);
        self.tiles.iter().flatten().all(|&x| x == first)
    }
}

impl FaceType {
    /// Returns the corresponding colour of this face in its initial position
    pub fn get_solved_colour(&self) -> Colour {
        FACE_COLOURS[*self as usize]
    }

    #[rustfmt::skip]
    pub fn get_edges(&self) -> [EdgeRef; 4] {
        match self {
            FaceType::Top => [
                EdgeRef { face: FaceType::Front, index: 0, is_row: true,  reversed: false },
                EdgeRef { face: FaceType::Left,  index: 0, is_row: true,  reversed: false },
                EdgeRef { face: FaceType::Back,  index: 0, is_row: true,  reversed: true  },
                EdgeRef { face: FaceType::Right, index: 0, is_row: true,  reversed: false },
            ],
            FaceType::Bottom => [
                EdgeRef { face: FaceType::Front, index: CUBE_SIZE - 1, is_row: true,  reversed: false },
                EdgeRef { face: FaceType::Right, index: CUBE_SIZE - 1, is_row: true,  reversed: false },
                EdgeRef { face: FaceType::Back,  index: CUBE_SIZE - 1, is_row: true,  reversed: true  },
                EdgeRef { face: FaceType::Left,  index: CUBE_SIZE - 1, is_row: true,  reversed: false },
            ],
            FaceType::Left => [
                EdgeRef { face: FaceType::Front,  index: 0,             is_row: false, reversed: false },
                EdgeRef { face: FaceType::Top,    index: 0,             is_row: false, reversed: false },
                EdgeRef { face: FaceType::Back,   index: CUBE_SIZE - 1, is_row: false, reversed: true  },
                EdgeRef { face: FaceType::Bottom, index: 0,             is_row: false, reversed: false },
            ],
            FaceType::Right => [
                EdgeRef { face: FaceType::Front,  index: CUBE_SIZE - 1, is_row: false, reversed: false },
                EdgeRef { face: FaceType::Bottom, index: CUBE_SIZE - 1, is_row: false, reversed: false },
                EdgeRef { face: FaceType::Back,   index: 0,             is_row: false, reversed: true  },
                EdgeRef { face: FaceType::Top,    index: CUBE_SIZE - 1, is_row: false, reversed: false },
            ],
            FaceType::Front => [
                EdgeRef { face: FaceType::Top,    index: CUBE_SIZE - 1, is_row: true,  reversed: false },
                EdgeRef { face: FaceType::Right,  index: 0,             is_row: false, reversed: false },
                EdgeRef { face: FaceType::Bottom, index: 0,             is_row: true,  reversed: true  },
                EdgeRef { face: FaceType::Left,   index: CUBE_SIZE - 1, is_row: false, reversed: true  },
            ],
            FaceType::Back => [
                EdgeRef { face: FaceType::Top,    index: 0,             is_row: true,  reversed: true  },
                EdgeRef { face: FaceType::Left,   index: 0,             is_row: false, reversed: true  },
                EdgeRef { face: FaceType::Bottom, index: CUBE_SIZE - 1, is_row: false, reversed: false },
                EdgeRef { face: FaceType::Right,  index: CUBE_SIZE - 1, is_row: false, reversed: true  },
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_face() {
        let face = Face::new(Colour::Green);
        for i in 0..CUBE_SIZE {
            for j in 0..CUBE_SIZE {
                assert_eq!(face.get_tile_colour(i, j), Colour::Green);
            }
        }
    }

    #[test]
    fn make_turn_clockwise() {
        let mut face = Face::new(Colour::Blue);
        face.set_tile_colour(0, 0, Colour::Green);
        face.set_tile_colour(0, 1, Colour::White);
        face.set_tile_colour(2, 2, Colour::Red);
        // G W B
        // B B B
        // B B R

        face.make_turn(TurnType::Clockwise);
        // B B G
        // B B W
        // R B B

        let mut target_face = Face::new(Colour::Blue);
        target_face.set_tile_colour(0, 2, Colour::Green);
        target_face.set_tile_colour(1, 2, Colour::White);
        target_face.set_tile_colour(2, 0, Colour::Red);

        assert_eq!(face, target_face);
    }

    #[test]
    fn make_turn_counterclockwise() {
        let mut face = Face::new(Colour::Blue);
        face.set_tile_colour(0, 0, Colour::Green);
        face.set_tile_colour(0, 1, Colour::White);
        face.set_tile_colour(2, 2, Colour::Red);
        // G W B
        // B B B
        // B B R

        face.make_turn(TurnType::CounterClockwise);
        // B B R
        // W B B
        // G B B

        let mut target_face = Face::new(Colour::Blue);
        target_face.set_tile_colour(0, 2, Colour::Red);
        target_face.set_tile_colour(1, 0, Colour::White);
        target_face.set_tile_colour(2, 0, Colour::Green);

        assert_eq!(face, target_face);
    }

    #[test]
    fn make_half_turn() {
        let mut face = Face::new(Colour::Blue);
        face.set_tile_colour(0, 0, Colour::Green);
        face.set_tile_colour(0, 1, Colour::White);
        face.set_tile_colour(2, 2, Colour::Red);
        // G W B
        // B B B
        // B B R

        face.make_turn(TurnType::Half);
        // R B B
        // B B B
        // B W G

        let mut target_face = Face::new(Colour::Blue);
        target_face.set_tile_colour(0, 0, Colour::Red);
        target_face.set_tile_colour(2, 1, Colour::White);
        target_face.set_tile_colour(2, 2, Colour::Green);

        assert_eq!(face, target_face);
    }

    #[test]
    fn is_solved_false() {
        let mut face = Face::new(Colour::Yellow);
        face.set_tile_colour(0, 0, Colour::Blue);
        assert!(!face.is_solved());
    }

    #[test]
    fn is_solved_true() {
        let face = Face::new(Colour::Yellow);
        assert!(face.is_solved());
    }
}
