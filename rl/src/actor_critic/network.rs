use crate::{ACTIONS, CUBE_SIZE, get_device};
use tch::{
    Tensor,
    nn::{self, Module},
};

// TODO:
// Multi-head attention
// Learned positional embeddings instead of hand-crafted coordinates

const FACE_TILES: i64 = (CUBE_SIZE * CUBE_SIZE) as i64 * 6;
const OUTPUT_SIZE: i64 = ACTIONS as i64;
const D_MODEL: i64 = 128;
const BLOCK_HIDDEN: i64 = 512;
const NUM_BLOCKS: usize = 3;

#[derive(Debug)]
pub struct TileEmbedding {
    shared_linear: nn::Linear,
    norm: nn::LayerNorm,
    pos_features: Tensor, // [24, 5]
}

impl TileEmbedding {
    pub fn new(vs: nn::Path, embed_dim: i64) -> Self {
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
            shared_linear: nn::linear(&vs / "linear", 11, embed_dim, Default::default()),
            norm: nn::layer_norm(&vs / "norm", vec![embed_dim], Default::default()),
            pos_features,
        }
    }

    pub fn forward(&self, xs: &Tensor) -> Tensor {
        let batch_size = xs.size()[0];

        // [batch, 144] -> [batch, 24, 6]
        let xs = xs.view([batch_size, FACE_TILES, 6]);

        // [24, 5] -> [1, 24, 5] -> [batch, 24, 5]
        let pos = self
            .pos_features
            .unsqueeze(0)
            .expand([batch_size, -1, -1], false);

        // [batch, 24, 11]
        let xs = Tensor::cat(&[&xs, &pos], 2);

        // shared linear [batch, 24, 11] -> [batch, 24, embed_dim]
        self.shared_linear.forward(&xs).apply(&self.norm)
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
pub struct SelfAttention {
    q_linear: nn::Linear,
    k_linear: nn::Linear,
    v_linear: nn::Linear,
    embed_dim: i64,
    out_linear: nn::Linear,
}

impl SelfAttention {
    pub fn new(vs: nn::Path, embed_dim: i64) -> Self {
        let config = nn::LinearConfig::default();
        Self {
            q_linear: nn::linear(&vs / "q", embed_dim, embed_dim, config),
            k_linear: nn::linear(&vs / "k", embed_dim, embed_dim, config),
            v_linear: nn::linear(&vs / "v", embed_dim, embed_dim, config),
            embed_dim,
            out_linear: nn::linear(&vs / "out", embed_dim, embed_dim, config),
        }
    }
}

impl Module for SelfAttention {
    fn forward(&self, x: &Tensor) -> Tensor {
        // x shape: [batch_size, seq_len, embed_dim]
        let q = x.apply(&self.q_linear);
        let k = x.apply(&self.k_linear);
        let v = x.apply(&self.v_linear);

        // Calculate attention scores: (Q * K^T)
        let scores = q.matmul(&k.transpose(-2, -1));

        // Scale scores by sqrt(embed_dim)
        let scaling_factor = (self.embed_dim as f64).sqrt();
        let scaled_scores = scores / scaling_factor;

        // Apply softmax to get attention weights
        let attention_weights = scaled_scores.softmax(-1, tch::Kind::Float);

        // Multiply weights by values: (Weights * V)
        attention_weights.matmul(&v).apply(&self.out_linear)
    }
}

#[derive(Debug)]
struct TransformerLayer {
    attention: SelfAttention,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl TransformerLayer {
    fn new(vs: &nn::Path, in_dim: i64) -> Self {
        Self {
            attention: SelfAttention::new(vs / "self_attention", in_dim),
            norm1: nn::layer_norm(vs / "norm1", vec![D_MODEL], Default::default()),
            norm2: nn::layer_norm(vs / "norm2", vec![D_MODEL], Default::default()),
            fc1: hidden_linear(vs / "fc1", D_MODEL, BLOCK_HIDDEN),
            fc2: hidden_linear(vs / "fc2", BLOCK_HIDDEN, D_MODEL),
        }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        let attn = self.attention.forward(&xs.apply(&self.norm1));
        let xs = xs + attn;

        let mlp = xs
            .apply(&self.norm2)
            .apply(&self.fc1)
            .elu()
            .apply(&self.fc2);
        xs + mlp
    }
}

#[derive(Debug)]
pub struct DenseNetwork {
    embedding: TileEmbedding,
    blocks: Vec<TransformerLayer>,
    head: nn::Sequential,
}

impl Module for DenseNetwork {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut xs = self.embedding.forward(xs);
        for block in self.blocks.iter() {
            xs = block.forward(&xs);
        }
        xs = xs.mean_dim(&[1i64][..], false, tch::Kind::Float);
        self.head.forward(&xs)
    }
}

pub fn initialize_network(vs: &nn::Path) -> DenseNetwork {
    let mut blocks = Vec::with_capacity(NUM_BLOCKS);
    for i in 0..NUM_BLOCKS {
        blocks.push(TransformerLayer::new(&(vs / format!("block{i}")), D_MODEL));
    }

    let head = nn::seq()
        .add(head_linear(vs / "head1", D_MODEL, BLOCK_HIDDEN))
        .add_fn(|xs| xs.elu())
        .add(head_linear(vs / "head2", BLOCK_HIDDEN, OUTPUT_SIZE));

    DenseNetwork {
        embedding: TileEmbedding::new(vs / "embed", D_MODEL),
        blocks,
        head,
    }
}
