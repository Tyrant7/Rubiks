use crate::{ACTIONS, CUBE_SIZE, get_device};
use tch::{
    Tensor,
    nn::{self, Module},
};

const FACE_TILES: i64 = (CUBE_SIZE * CUBE_SIZE) as i64 * 6;
const OUTPUT_SIZE: i64 = ACTIONS as i64;
const HIDDEN: i64 = 256;
const GROWTH: i64 = 128;
const EMBED_DIM: i64 = 16;
const LAYERS_PER_BLOCK: usize = 2;
const NUM_BLOCKS: usize = 3;

#[derive(Debug)]
pub struct TileEmbedding {
    shared_linear: nn::Linear,
    pos_features: Tensor, // [24, 5]
}

impl TileEmbedding {
    pub fn new(path: &nn::Path, embed_dim: i64) -> Self {
        let mut features = Vec::new();
        for face_idx in 0..6i32 {
            for row in 0..CUBE_SIZE as i32 {
                for col in 0..CUBE_SIZE as i32 {
                    features.push(face_idx as f32 / 5.0);
                    features.push((face_idx as f32 * std::f32::consts::TAU / 6.0).sin());
                    features.push((face_idx as f32 * std::f32::consts::TAU / 6.0).cos());
                    features.push(row as f32);
                    features.push(col as f32);
                }
            }
        }
        let pos_features = Tensor::from_slice(&features)
            .view([FACE_TILES, 5])
            .to_device(get_device());

        TileEmbedding {
            shared_linear: nn::linear(path, 11, embed_dim, Default::default()),
            pos_features,
        }
    }

    pub fn forward(&self, xs: &Tensor) -> Tensor {
        let batch_size = xs.size()[0];

        // [batch, 144] → [batch, 24, 6]
        let xs = xs.view([batch_size, FACE_TILES, 6]);

        // [24, 5] → [1, 24, 5] → [batch, 24, 5]
        let pos = self
            .pos_features
            .unsqueeze(0)
            .expand([batch_size, -1, -1], false);

        // [batch, 24, 11]
        let xs = Tensor::cat(&[&xs, &pos], 2);

        // shared linear [batch, 24, 11] → [batch, 24, embed_dim]
        let xs = self.shared_linear.forward(&xs);

        // flatten [batch, 24 * embed_dim]
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
    embedding: TileEmbedding,
    input: nn::Linear,
    blocks: Vec<DenseBlock>,
    transitions: Vec<nn::Linear>,
    head: nn::Linear,
}

impl Module for DenseNetwork {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = self.embedding.forward(xs).elu();
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
        embedding: TileEmbedding::new(vs, EMBED_DIM),
        input: hidden_linear(vs / "input", FACE_TILES * EMBED_DIM, HIDDEN),
        blocks,
        transitions,
        head: head_linear(vs / "head", HIDDEN, OUTPUT_SIZE),
    }
}
