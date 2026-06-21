use crate::{INPUT_SIZE, OUTPUT_SIZE};
use tch::{
    Tensor,
    nn::{self, Module},
};

const HIDDEN: i64 = 256;
const GROWTH: i64 = 128;
const EMBED_DIM: i64 = 16;
const LAYERS_PER_BLOCK: usize = 2;
const NUM_BLOCKS: usize = 3;

#[derive(Debug)]
pub struct ColourEmbedding {
    embedding: nn::Embedding,
    embed_dim: i64,
}

impl ColourEmbedding {
    pub fn new(vs: &nn::Path, embed_dim: i64) -> Self {
        Self {
            embedding: nn::embedding(vs / "colour_embed", 6, embed_dim, Default::default()),
            embed_dim,
        }
    }

    pub fn forward(&self, colour_indices: &Tensor) -> Tensor {
        // colour_indices: [batch, INPUT_SIZE] integer indices
        // output: [batch, INPUT_SIZE * embed_dim]
        let embedded = self.embedding.forward(colour_indices);
        embedded.view([-1, INPUT_SIZE as i64 * self.embed_dim])
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
    embedding: ColourEmbedding,
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
        embedding: ColourEmbedding::new(vs, EMBED_DIM),
        input: hidden_linear(vs / "input", INPUT_SIZE as i64 * EMBED_DIM, HIDDEN),
        blocks,
        transitions,
        head: head_linear(vs / "head", HIDDEN, OUTPUT_SIZE as i64),
    }
}
