use rubiks::{Cube, FaceType, ScrambleType, Turn, TurnType};

#[test]
fn scramble_not_solved() {
    let mut cube = Cube::default();
    cube.scramble(30, ScrambleType::Seeded(42));
    assert!(!cube.is_solved());
}

#[test]
fn make_turns() {
    let mut cube = Cube::default();
    assert!(cube.is_solved());

    // Turn in
    cube.make_turn(Turn::new(FaceType::Bottom, TurnType::CounterClockwise));
    cube.make_turn(Turn::new(FaceType::Top, TurnType::Clockwise));
    cube.make_turn(Turn::new(FaceType::Right, TurnType::Half));
    cube.make_turn(Turn::new(FaceType::Left, TurnType::Clockwise));
    cube.make_turn(Turn::new(FaceType::Front, TurnType::Clockwise));
    cube.make_turn(Turn::new(FaceType::Back, TurnType::Half));

    // Turn out
    cube.make_turn(Turn::new(FaceType::Back, TurnType::Half));
    cube.make_turn(Turn::new(FaceType::Front, TurnType::CounterClockwise));
    cube.make_turn(Turn::new(FaceType::Left, TurnType::CounterClockwise));
    cube.make_turn(Turn::new(FaceType::Right, TurnType::Half));
    cube.make_turn(Turn::new(FaceType::Top, TurnType::CounterClockwise));
    cube.make_turn(Turn::new(FaceType::Bottom, TurnType::Clockwise));

    assert!(cube.is_solved());
}
