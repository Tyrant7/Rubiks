use tch::{
    Tensor,
    nn::{self, Module},
};

use crate::{INPUT_SIZE, OUTPUT_SIZE};

pub fn relu_sq(xs: &Tensor) -> Tensor {
    let activated = xs.relu();
    &activated * &activated
}

pub fn linear(vs: nn::Path, in_dim: i64, out_dim: i64, ws_init: nn::Init) -> nn::Linear {
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

pub fn hidden_linear(vs: nn::Path, in_dim: i64, out_dim: i64) -> nn::Linear {
    linear(vs, in_dim, out_dim, nn::init::DEFAULT_KAIMING_NORMAL)
}

pub fn residual_output_linear(vs: nn::Path, in_dim: i64, out_dim: i64) -> nn::Linear {
    linear(vs, in_dim, out_dim, nn::Init::Const(0.0))
}

pub fn head_linear(vs: nn::Path, in_dim: i64, out_dim: i64) -> nn::Linear {
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
pub struct ResidualBlock {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

#[derive(Debug)]
pub struct ResidualNetwork {
    input: nn::Linear,
    blocks: [ResidualBlock; 3],
    head: nn::Linear,
}

impl Module for ResidualNetwork {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut xs = relu_sq(&self.input.forward(xs));
        for block in &self.blocks {
            let residual = xs.shallow_clone();
            let hidden = relu_sq(&block.fc1.forward(&xs));
            xs = residual + block.fc2.forward(&hidden);
        }
        self.head.forward(&xs)
    }
}

pub fn initialize_network(vs: &nn::Path) -> ResidualNetwork {
    ResidualNetwork {
        input: hidden_linear(vs / "input", INPUT_SIZE as i64, 256),
        blocks: std::array::from_fn(|idx| ResidualBlock {
            fc1: hidden_linear(vs / format!("block{idx}_fc1"), 256, 256),
            fc2: residual_output_linear(vs / format!("block{idx}_fc2"), 256, 256),
        }),
        head: head_linear(vs / "head", 256, OUTPUT_SIZE as i64),
    }
}
