use tensorboard_rs::summary_writer::SummaryWriter;

pub trait Loggable {
    fn scalars(&self) -> Vec<(&'static str, f32)>;
}

pub fn write_scalars(writer: &mut SummaryWriter, items: &[(&str, f32)], step: usize) {
    for (key, val) in items {
        writer.add_scalar(key, *val, step);
    }
}
