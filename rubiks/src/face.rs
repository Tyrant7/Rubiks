use crate::turn::TurnType;

/// Used for mapping `FaceType` representation of tiles to their respective colours.
const FACE_COLOURS: [Colour; 6] = [
    Colour::White,
    Colour::Red,
    Colour::Green,
    Colour::Blue,
    Colour::Orange,
    Colour::Yellow,
];

/// Represents a single face of the cube as a 2D grid of tiles.
/// Each tile stores a [`Colour`] directly, representing which face
/// the tile belongs to in the solved state.
#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
pub struct Face<const SIZE: usize> {
    tiles: [[Colour; SIZE]; SIZE],
}

/// Represents the orientation of a face on the cube.
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum FaceType<const SIZE: usize> {
    Top,
    Bottom,
    Front,
    Back,
    Left,
    Right,
}

/// Describes an edge of a face that participates in an edge cycle when a turn is made.
/// Used by [`FaceType::get_edges`] to generalize edge cycling across all faces.
#[derive(Clone, Debug)]
pub struct EdgeRef<const SIZE: usize> {
    /// The face this edge belongs to.
    pub face: FaceType<SIZE>,
    /// The row or column index of the edge on the face.
    pub index: usize,
    /// If `true`, the edge is a row; if `false`, it is a column.
    pub is_row: bool,
    /// If `true`, the edge tiles should be reversed before cycling.
    /// This accounts for edges that run in opposite directions on adjacent faces.
    pub reversed: bool,
}

/// Represents the colour of a tile.
/// Each variant corresponds to one of the six faces in the solved state.
#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
pub enum Colour {
    White,
    Red,
    Green,
    Blue,
    Orange,
    Yellow,
}

impl<const SIZE: usize> Face<SIZE> {
    /// Creates a new face with all tiles set to the given colour.
    pub fn new(colour: Colour) -> Self {
        Face {
            tiles: [[colour; SIZE]; SIZE],
        }
    }

    /// Rotates the tiles within this face according to the given turn type.
    /// Note: this only handles the internal tile rotation and not cycling edges
    /// with adjacent faces. See [`crate::cube::Cube::cycle_edges`] for that behaviour.
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

    /// Rotates the tile grid in place 90 degrees clockwise.
    fn rotate_clockwise(&mut self) {
        // 1. Reverse columns
        self.tiles.reverse();
        // 2. Transpose
        for i in 0..SIZE {
            for j in i + 1..SIZE {
                let temp = self.tiles[i][j];
                self.tiles[i][j] = self.tiles[j][i];
                self.tiles[j][i] = temp;
            }
        }
    }

    /// Rotates the tile grid in place 90 degrees counterclockwise.
    fn rotate_counterclockwise(&mut self) {
        // 1. Transpose
        for i in 0..SIZE {
            for j in i + 1..SIZE {
                let temp = self.tiles[i][j];
                self.tiles[i][j] = self.tiles[j][i];
                self.tiles[j][i] = temp;
            }
        }
        // 2. Reverse columns
        self.tiles.reverse();
    }

    /// Returns the colour of the tile at the given row and column.
    pub fn get_tile_colour(&self, row: usize, col: usize) -> Colour {
        self.tiles[row][col]
    }

    /// Sets the tile at the given row and column to the given colour.
    #[allow(dead_code)]
    pub fn set_tile_colour(&mut self, row: usize, col: usize, tile: Colour) {
        self.tiles[row][col] = tile;
    }

    /// Returns all tiles in the given row.
    pub fn get_row(&self, row: usize) -> [Colour; SIZE] {
        self.tiles[row]
    }

    /// Returns all tiles in the given column.
    pub fn get_col(&self, col: usize) -> [Colour; SIZE] {
        std::array::from_fn(|i| self.tiles[i][col])
    }

    /// Sets all tiles in the given row.
    pub fn set_row(&mut self, row: usize, data: [Colour; SIZE]) {
        self.tiles[row] = data;
    }

    /// Sets all tiles in the given column.
    pub fn set_col(&mut self, col: usize, data: [Colour; SIZE]) {
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

impl<const SIZE: usize> FaceType<SIZE> {
    /// Returns the colour this face has in the solved state.
    pub fn get_solved_colour(&self) -> Colour {
        FACE_COLOURS[*self as usize]
    }

    /// Returns the four edges that participate in a cycle when this face is turned,
    /// in clockwise order. Each [`EdgeRef`] describes which row or column of an
    /// adjacent face is involved and whether it needs to be reversed during cycling.
    #[rustfmt::skip]
    pub fn get_edges(&self) -> [EdgeRef<SIZE>; 4] {
        match self {
            FaceType::Top => [
                EdgeRef { face: FaceType::Front, index: 0, is_row: true,  reversed: false },
                EdgeRef { face: FaceType::Left,  index: 0, is_row: true,  reversed: false },
                EdgeRef { face: FaceType::Back,  index: 0, is_row: true,  reversed: true  },
                EdgeRef { face: FaceType::Right, index: 0, is_row: true,  reversed: false },
            ],
            FaceType::Bottom => [
                EdgeRef { face: FaceType::Front, index: SIZE - 1, is_row: true,  reversed: false },
                EdgeRef { face: FaceType::Right, index: SIZE - 1, is_row: true,  reversed: false },
                EdgeRef { face: FaceType::Back,  index: SIZE - 1, is_row: true,  reversed: true  },
                EdgeRef { face: FaceType::Left,  index: SIZE - 1, is_row: true,  reversed: false },
            ],
            FaceType::Left => [
                EdgeRef { face: FaceType::Front,  index: 0,             is_row: false, reversed: false },
                EdgeRef { face: FaceType::Top,    index: 0,             is_row: false, reversed: false },
                EdgeRef { face: FaceType::Back,   index: SIZE - 1, is_row: false, reversed: true  },
                EdgeRef { face: FaceType::Bottom, index: 0,             is_row: false, reversed: false },
            ],
            FaceType::Right => [
                EdgeRef { face: FaceType::Front,  index: SIZE - 1, is_row: false, reversed: false },
                EdgeRef { face: FaceType::Bottom, index: SIZE - 1, is_row: false, reversed: false },
                EdgeRef { face: FaceType::Back,   index: 0,             is_row: false, reversed: true  },
                EdgeRef { face: FaceType::Top,    index: SIZE - 1, is_row: false, reversed: false },
            ],
            FaceType::Front => [
                EdgeRef { face: FaceType::Top,    index: SIZE - 1, is_row: true,  reversed: false },
                EdgeRef { face: FaceType::Right,  index: 0,             is_row: false, reversed: false },
                EdgeRef { face: FaceType::Bottom, index: 0,             is_row: true,  reversed: true  },
                EdgeRef { face: FaceType::Left,   index: SIZE - 1, is_row: false, reversed: true  },
            ],
            FaceType::Back => [
                EdgeRef { face: FaceType::Top,    index: 0,             is_row: true,  reversed: true  },
                EdgeRef { face: FaceType::Left,   index: 0,             is_row: false, reversed: true  },
                EdgeRef { face: FaceType::Bottom, index: SIZE - 1, is_row: false, reversed: false },
                EdgeRef { face: FaceType::Right,  index: SIZE - 1, is_row: false, reversed: true  },
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_face() {
        let face = Face::<3>::new(Colour::Green);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(face.get_tile_colour(i, j), Colour::Green);
            }
        }
    }

    #[test]
    fn make_turn_clockwise() {
        let mut face = Face::<3>::new(Colour::Blue);
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
        let mut face = Face::<3>::new(Colour::Blue);
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
        let mut face = Face::<3>::new(Colour::Blue);
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
        let mut face = Face::<3>::new(Colour::Yellow);
        face.set_tile_colour(0, 0, Colour::Blue);
        assert!(!face.is_solved());
    }

    #[test]
    fn is_solved_true() {
        let face = Face::<3>::new(Colour::Yellow);
        assert!(face.is_solved());
    }
}
