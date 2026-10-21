use rayon::prelude::*;

use crate::model::{GameState, JointAction};
use crate::rules::step;

#[derive(Clone, Debug)]
pub struct BatchResult {
    pub states: Vec<GameState>,
    pub transitions: u64,
}

pub fn run_batch(mut states: Vec<GameState>, action_streams: &[Vec<JointAction>]) -> BatchResult {
    assert_eq!(states.len(), action_streams.len());
    states
        .par_iter_mut()
        .zip(action_streams.par_iter())
        .for_each(|(state, actions)| {
            for action in actions {
                if state
                    .statuses
                    .iter()
                    .all(|status| *status == crate::model::Status::Done)
                {
                    break;
                }
                step(state, action);
            }
        });
    BatchResult {
        transitions: action_streams
            .iter()
            .map(|actions| actions.len() as u64)
            .sum(),
        states,
    }
}
