use crate::cube::{Colour, TurnType};

pub const CUBE_SIZE: usize = 3;

#[derive(PartialEq, Debug, Clone, Copy)]
pub struct Face {
    tiles: [[u8; CUBE_SIZE]; CUBE_SIZE],
}

impl Face {
    pub fn new(colour: Colour) -> Self {
        let fill = colour as u8;
        Face {
            tiles: [[fill; CUBE_SIZE]; CUBE_SIZE],
        }
    }

    pub fn make_turn(&mut self, turn: TurnType) {
        unimplemented!()
    }

    pub fn get_tile_colour(&self, col: usize, row: usize) -> Colour {
        unimplemented!()
    }

    fn set_tile_colour(&self, col: usize, row: usize, tile: Colour) {
        unimplemented!()
    }

    pub fn is_solved(&self) -> bool {
        unimplemented!()
    }

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

        face.make_turn(TurnType::CounterClockwise);
        // R B B
        // B B B
        // B W G

        let target_face = Face::new(Colour::Blue);
        face.set_tile_colour(0, 0, Colour::Red);
        face.set_tile_colour(1, 2, Colour::White);
        face.set_tile_colour(2, 2, Colour::Green);

        assert_eq!(face, target_face);
    }
}
