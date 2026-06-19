use crate::{logging::Loggable, sac::SacConfig};

// --- Config ------------------------------------

#[rustfmt::skip]
impl Loggable for SacConfig {
    fn scalars(&self) -> Vec<(&'static str, f32)> {
        vec![
            ("sac/update_every",                self.update_every as f32),
            ("sac/curriculum_threshold",        self.curriculum_threshold as f32),
            ("sac/curriculum_min_episodes",     self.curriculum_min_episodes as f32),
            ("sac/clear_replay_on_advance",     self.clear_replay_on_advance as u8 as f32),
            ("sac/adam_eps",                    self.adam_eps as f32),
            ("sac/target_entropy_scale",        self.target_entropy_scale as f32),
            ("sac/log_alpha_init",              self.log_alpha_init as f32),
            ("sac/target_network_frequency",    self.target_network_frequency as f32),
        ]
    }
}

// --- Episode metrics ------------------------------------

pub struct EpisodeMetrics {
    pub(crate) reward: f32,
    pub(crate) solved: bool,
    pub(crate) truncated: bool,
    pub(crate) max_solved_faces: usize,
    pub(crate) recent_solve_rate: f32,
}

impl Loggable for EpisodeMetrics {
    fn scalars(&self) -> Vec<(&'static str, f32)> {
        vec![
            ("episode/reward", self.reward),
            ("episode/solve", self.solved as u8 as f32),
            ("episode/truncated", self.truncated as u8 as f32),
            ("episode/max_solved_faces", self.max_solved_faces as f32),
            ("episode/recent_solve_rate", self.recent_solve_rate),
        ]
    }
}

// --- Curriculum metrics ------------------------------------

pub struct CurriculumMetrics {
    pub(crate) scramble_depth: usize,
    pub(crate) max_steps: usize,
    pub(crate) replay_size: usize,
}

impl Loggable for CurriculumMetrics {
    fn scalars(&self) -> Vec<(&'static str, f32)> {
        vec![
            ("curriculum/scramble_depth", self.scramble_depth as f32),
            ("curriculum/max_steps", self.max_steps as f32),
            ("curriculum/replay_size", self.replay_size as f32),
        ]
    }
}

// --- Performance metrics ------------------------------------

pub struct PerformanceMetrics {
    pub(crate) sps: f32,
    pub(crate) recent_sps: f32,
    pub(crate) env_steps: usize,
    pub(crate) learner_updates: usize,
}

impl Loggable for PerformanceMetrics {
    fn scalars(&self) -> Vec<(&'static str, f32)> {
        vec![
            ("performance/sps", self.sps),
            ("performance/recent_sps", self.recent_sps),
            ("performance/env_steps", self.env_steps as f32),
            ("performance/learner_updates", self.learner_updates as f32),
        ]
    }
}

// --- Update metrics ------------------------------------

#[derive(Default)]
pub struct UpdateMetricTotals {
    pub(crate) actor_loss: f32,
    pub(crate) critic_loss: f32,
    pub(crate) alpha_loss: f32,
    pub(crate) entropy: f32,
    pub(crate) entropy_error: f32,
    pub(crate) policy_max_prob: f32,
    pub(crate) target_q: f32,
    pub(crate) q1: f32,
    pub(crate) q2: f32,
    pub(crate) replay_truncation: f32,
    pub(crate) steps: usize,
}

pub struct UpdateMetrics {
    pub(crate) actor_loss: f32,
    pub(crate) critic_loss: f32,
    pub(crate) alpha_loss: f32,
    pub(crate) entropy: f32,
    pub(crate) entropy_error: f32,
    pub(crate) policy_max_prob: f32,
    pub(crate) target_q: f32,
    pub(crate) q1: f32,
    pub(crate) q2: f32,
    pub(crate) replay_truncation: f32,
}

impl UpdateMetricTotals {
    pub fn add(&mut self, metrics: UpdateMetrics) {
        self.actor_loss += metrics.actor_loss;
        self.critic_loss += metrics.critic_loss;
        self.alpha_loss += metrics.alpha_loss;
        self.entropy += metrics.entropy;
        self.entropy_error += metrics.entropy_error;
        self.policy_max_prob += metrics.policy_max_prob;
        self.target_q += metrics.target_q;
        self.q1 += metrics.q1;
        self.q2 += metrics.q2;
        self.replay_truncation += metrics.replay_truncation;
        self.steps += 1;
    }

    pub fn average(&self, value: f32) -> f32 {
        value / self.steps as f32
    }
}

impl Loggable for UpdateMetricTotals {
    fn scalars(&self) -> Vec<(&'static str, f32)> {
        vec![
            ("loss/critic_avg", self.average(self.critic_loss) / 2.),
            ("loss/actor", self.average(self.actor_loss)),
            ("loss/alpha", self.average(self.alpha_loss)),
            ("entropy/policy", self.average(self.entropy)),
            (
                "entropy/error_policy_minus_target",
                self.average(self.entropy_error),
            ),
            ("policy/max_probability", self.average(self.policy_max_prob)),
            ("q/target", self.average(self.target_q)),
            ("q/q1_taken", self.average(self.q1)),
            ("q/q2_taken", self.average(self.q2)),
            (
                "replay/truncation_rate",
                self.average(self.replay_truncation),
            ),
        ]
    }
}

// --- Alpha metrics ------------------------------------

pub struct AlphaMetrics {
    pub(crate) value: f32,
    pub(crate) log_value: f32,
    pub(crate) target_entropy: f32,
}

impl Loggable for AlphaMetrics {
    fn scalars(&self) -> Vec<(&'static str, f32)> {
        vec![
            ("alpha/value", self.value),
            ("alpha/log_value", self.log_value),
            ("entropy/target", self.target_entropy),
        ]
    }
}
