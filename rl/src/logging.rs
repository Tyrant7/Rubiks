use tensorboard_rs::summary_writer::SummaryWriter;

use crate::TrainingConfig;

pub trait Loggable {
    fn scalars(&self) -> Vec<(&'static str, f32)>;
}

pub fn write_scalars(writer: &mut SummaryWriter, items: &[(&str, f32)], step: usize) {
    for (key, val) in items {
        writer.add_scalar(key, *val, step);
    }
}

#[rustfmt::skip]
impl Loggable for TrainingConfig {
    fn scalars(&self) -> Vec<(&'static str, f32)> {
        vec![
            ("config/num_envs",                    self.num_envs as f32),
            ("config/learning_starts",             self.learning_starts as f32),
            ("config/eval_episodes",               self.eval_episodes as f32),
            ("config/eval_every",                  self.eval_every as f32),
        ]
    }
}
