use rubiks::Cube;

fn main() {
    let cube = Cube::default();
    println!("{:#?}", cube);

    // Hooray!
    let t = tch::Tensor::from_slice(&[3, 1, 4, 1, 5]);
    let t = t * 2;
    t.print();
}
