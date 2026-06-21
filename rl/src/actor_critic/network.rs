use crate::{INPUT_SIZE, OUTPUT_SIZE, get_device};
use tch::{
    Tensor,
    nn::{self, Module},
};

const HIDDEN: i64 = 256;
const GROWTH: i64 = 128;
const EMBED_DIM: i64 = 4;
const FACE_TILES: i64 = INPUT_SIZE as i64 / 6;
const LAYERS_PER_BLOCK: usize = 2;
const NUM_BLOCKS: usize = 3;

#[derive(Debug)]
pub struct PositionalEmbedding {
    embedding: nn::Embedding,
    embed_dim: i64,
}

impl PositionalEmbedding {
    pub fn new(vs: &nn::Path, embed_dim: i64) -> Self {
        Self {
            embedding: nn::embedding(vs / "pos_embed", FACE_TILES, embed_dim, Default::default()),
            embed_dim,
        }
    }

    pub fn forward(&self, batch_size: i64) -> Tensor {
        // indices 0..FACE_TILES-1, tiled across batch
        let indices = Tensor::arange(FACE_TILES, (tch::Kind::Int64, get_device()));
        let embedded = self.embedding.forward(&indices);
        // [FACE_TILES, embed_dim] -> [1, FACE_TILES * embed_dim] -> [batch, FACE_TILES * embed_dim]
        embedded
            .view([1, FACE_TILES * self.embed_dim])
            .expand([batch_size, -1], false)
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
        let batch_size = xs.size()[0];
        let pos = self.embedding.forward(batch_size);
        let xs = Tensor::cat(&[xs, &pos], 1); // [batch, 144 + INPUT_SIZE * embed_dim]
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
        embedding: PositionalEmbedding::new(vs, EMBED_DIM),
        input: hidden_linear(
            vs / "input",
            INPUT_SIZE as i64 + FACE_TILES * EMBED_DIM,
            HIDDEN,
        ),
        blocks,
        transitions,
        head: head_linear(vs / "head", HIDDEN, OUTPUT_SIZE as i64),
    }
}
