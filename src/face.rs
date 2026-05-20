use crate::cube::{Colour, TurnType};

/// The size of the cube, i.e. a standard Rubik's cube is 3x3.
pub const CUBE_SIZE: usize = 3;

/// Represents a single face of the cube as a 2D grid of tiles.
/// Each tile value is a `u8` corresponding to a [`Colour`], representing
/// which face the tile belongs to in the solved state.
#[derive(PartialEq, Debug, Clone, Copy)]
pub struct Face {
    tiles: [[u8; CUBE_SIZE]; CUBE_SIZE],
}

impl Face {
    /// Creates a new face with all tiles set to the given colour.
    pub fn new(colour: Colour) -> Self {
        let fill = colour as u8;
        Face {
            tiles: [[fill; CUBE_SIZE]; CUBE_SIZE],
        }
    }

    /// Rotates the tiles within this face according to the given turn type.
    /// Note: this only handles the internal tile rotation and not cycling edges
    /// when this face is adjacent to the rotated face.
    pub fn make_turn(&mut self, turn: TurnType) {
        unimplemented!()
    }

    /// Returns the colour of the tile at the given column and row.
    pub fn get_tile_colour(&self, col: usize, row: usize) -> Colour {
        unimplemented!()
    }

    /// Sets the tile at the given column and row to the given colour.
    fn set_tile_colour(&mut self, col: usize, row: usize, tile: Colour) {
        unimplemented!()
    }

    /// Returns true if all tiles on this face are the same colour.
    pub fn is_solved(&self) -> bool {
        unimplemented!()
    }

    /// Returns the raw `u8` value of the tile at the given column and row.
    fn get_tile_raw(&self, col: usize, row: usize) -> u8 {
        unimplemented!()
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
        face.set_tile_colour(1, 0, Colour::White);
        face.set_tile_colour(2, 2, Colour::Red);
        // G W B
        // B B B
        // B B R

        face.make_turn(TurnType::Clockwise);
        // B B G
        // B B W
        // R B B

        let target_face = Face::new(Colour::Blue);
        face.set_tile_colour(2, 0, Colour::Green);
        face.set_tile_colour(2, 1, Colour::White);
        face.set_tile_colour(0, 2, Colour::Red);

        assert_eq!(face, target_face);
    }

    #[test]
    fn make_turn_counterclockwise() {
        let mut face = Face::new(Colour::Blue);
        face.set_tile_colour(0, 0, Colour::Green);
        face.set_tile_colour(1, 0, Colour::White);
        face.set_tile_colour(2, 2, Colour::Red);
        // G W B
        // B B B
        // B B R

        face.make_turn(TurnType::CounterClockwise);
        // B B R
        // W B B
        // G B B

        let target_face = Face::new(Colour::Blue);
        face.set_tile_colour(2, 0, Colour::Red);
        face.set_tile_colour(0, 1, Colour::White);
        face.set_tile_colour(0, 2, Colour::Green);

        assert_eq!(face, target_face);
    }

    #[test]
    fn make_half_turn() {
        let mut face = Face::new(Colour::Blue);
        face.set_tile_colour(0, 0, Colour::Green);
        face.set_tile_colour(1, 0, Colour::White);
        face.set_tile_colour(2, 2, Colour::Red);
        // G W B
        // B B B
        // B B R

        face.make_turn(TurnType::Half);
        // R B B
        // B B B
        // B W G

        let target_face = Face::new(Colour::Blue);
        face.set_tile_colour(0, 0, Colour::Red);
        face.set_tile_colour(1, 2, Colour::White);
        face.set_tile_colour(2, 2, Colour::Green);

        assert_eq!(face, target_face);
    }

    #[test]
    fn is_solved_false() {
        let face = Face::new(Colour::Yellow);
        face.set_tile_colour(0, 0, Colour::Blue);
        assert!(!face.is_solved());
    }

    #[test]
    fn is_solved_true() {
        let face = Face::new(Colour::Yellow);
        assert!(face.is_solved());
    }
}
