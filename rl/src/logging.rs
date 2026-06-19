use tensorboard_rs::summary_writer::SummaryWriter;

pub trait Loggable {
    fn scalars(&self) -> Vec<(&'static str, f32)>;
}

pub fn write_scalars(writer: &mut SummaryWriter, items: &[(&str, f32)], step: usize) {
    for (key, val) in items {
        writer.add_scalar(key, *val, step);
    }
}

// --- Eval metrics ------------------------------------

pub struct EvalMetrics {
    pub(crate) solve_rate: f32,
    pub(crate) average_reward: f32,
    pub(crate) average_steps: f32,
}

impl Loggable for EvalMetrics {
    fn scalars(&self) -> Vec<(&'static str, f32)> {
        vec![
            ("eval/greedy_solve_rate", self.solve_rate),
            ("eval/greedy_average_reward", self.average_reward),
            ("eval/greedy_average_steps", self.average_steps),
        ]
    }
}
