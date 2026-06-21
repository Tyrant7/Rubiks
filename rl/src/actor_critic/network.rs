use crate::{ACTIONS, CUBE_SIZE, get_device};
use tch::{
    Tensor,
    nn::{self, Module},
};

const FACE_TILES: i64 = (CUBE_SIZE * CUBE_SIZE) as i64 / 6;
const INPUT_SIZE: i64 = FACE_TILES * 6;
const OUTPUT_SIZE: i64 = ACTIONS as i64;
const HIDDEN: i64 = 256;
const GROWTH: i64 = 128;
const EMBED_DIM: i64 = 4;
const LAYERS_PER_BLOCK: usize = 2;
const NUM_BLOCKS: usize = 3;

#[derive(Debug)]
pub struct PositionalEmbedding {
    shared_linear: nn::Linear,
}

impl PositionalEmbedding {
    pub fn new(path: &nn::Path, input_size: i64, output_size: i64) -> Self {
        PositionalEmbedding {
            shared_linear: nn::linear(path, input_size, output_size, Default::default()),
        }
    }

    pub fn forward(&self, xs: &Tensor) -> Tensor {
        let batch_size = xs.size()[0];

        // Reshape colour one-hots to per-tile
        let xs = xs.view([batch_size, FACE_TILES, 6]);

        // Build positional features [24, 3]
        let f = Tensor::arange(FACE_TILES, (tch::Kind::Float, get_device()));
        let normalized = &f / (FACE_TILES as f64 - 1.0);
        let sin = (&f * std::f64::consts::TAU / FACE_TILES as f64).sin();
        let cos = (&f * std::f64::consts::TAU / FACE_TILES as f64).cos();
        let pos = Tensor::stack(&[&normalized, &sin, &cos], 1); // [24, 3]

        // Tile pos across batch [1, 24, 3] -> [batch, 24, 3]
        let pos = pos.unsqueeze(0).expand([batch_size, -1, -1], false);

        // Concatenate per tile [batch, 24, 9]
        let xs = Tensor::cat(&[&xs, &pos], 2);

        // Shared linear [batch, 24, 9] -> [batch, 24, 16]
        let xs = self.shared_linear.forward(&xs);

        // flatten [batch, 24 * 16]
        xs.view([batch_size, -1])
    }
}

fn linear(vs: nn::Path, in_dim: i64, out_dim: i64, ws_init: nn::Init) -> nn::Linear {
    nn::linear(
        vs,
        in_dim,
        out_dim,
        nn::LinearConfig {
            ws_init,
            bs_init: Some(nn::Init::Const(0.0)),
            bias: true,
        },
    )
}

fn hidden_linear(vs: nn::Path, in_dim: i64, out_dim: i64) -> nn::Linear {
    linear(vs, in_dim, out_dim, nn::init::DEFAULT_KAIMING_NORMAL)
}

fn head_linear(vs: nn::Path, in_dim: i64, out_dim: i64) -> nn::Linear {
    linear(
        vs,
        in_dim,
        out_dim,
        nn::Init::Randn {
            mean: 0.0,
            stdev: 0.01,
        },
    )
}

#[derive(Debug)]
struct DenseLayer {
    norm: nn::LayerNorm,
    fc: nn::Linear,
}

impl DenseLayer {
    fn new(vs: &nn::Path, in_dim: i64) -> Self {
        Self {
            norm: nn::layer_norm(vs / "norm", vec![GROWTH], Default::default()),
            fc: hidden_linear(vs / "fc", in_dim, GROWTH),
        }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.fc).elu().apply(&self.norm)
    }
}

#[derive(Debug)]
struct DenseBlock {
    layers: Vec<DenseLayer>,
}

impl DenseBlock {
    fn new(vs: &nn::Path, in_dim: i64) -> Self {
        let layers = (0..LAYERS_PER_BLOCK)
            .map(|i| DenseLayer::new(&(vs / format!("layer{i}")), in_dim + i as i64 * GROWTH))
            .collect();
        Self { layers }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut outputs = vec![xs.shallow_clone()];
        for layer in &self.layers {
            let input = Tensor::cat(&outputs, 1);
            outputs.push(layer.forward(&input));
        }
        Tensor::cat(&outputs, 1)
    }

    fn out_dim(in_dim: i64) -> i64 {
        in_dim + LAYERS_PER_BLOCK as i64 * GROWTH
    }
}

#[derive(Debug)]
pub struct DenseNetwork {
    embedding: PositionalEmbedding,
    input: nn::Linear,
    blocks: Vec<DenseBlock>,
    transitions: Vec<nn::Linear>,
    head: nn::Linear,
}

impl Module for DenseNetwork {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = self.embedding.forward(xs);
        let mut xs = self.input.forward(&xs).elu();
        for (block, transition) in self.blocks.iter().zip(self.transitions.iter()) {
            xs = transition.forward(&block.forward(&xs)).elu();
        }
        self.head.forward(&xs)
    }
}

pub fn initialize_network(vs: &nn::Path) -> DenseNetwork {
    let block_out_dim = DenseBlock::out_dim(HIDDEN);

    let mut blocks = Vec::with_capacity(NUM_BLOCKS);
    let mut transitions = Vec::with_capacity(NUM_BLOCKS);
    for i in 0..NUM_BLOCKS {
        blocks.push(DenseBlock::new(&(vs / format!("block{i}")), HIDDEN));
        transitions.push(hidden_linear(
            vs / format!("transition{i}"),
            block_out_dim,
            HIDDEN,
        ));
    }

    DenseNetwork {
        embedding: PositionalEmbedding::new(vs, INPUT_SIZE, INPUT_SIZE * EMBED_DIM),
        input: hidden_linear(vs / "input", INPUT_SIZE + FACE_TILES * EMBED_DIM, HIDDEN),
        blocks,
        transitions,
        head: head_linear(vs / "head", HIDDEN, OUTPUT_SIZE),
    }
}
