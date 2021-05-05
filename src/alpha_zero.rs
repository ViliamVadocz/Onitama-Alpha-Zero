use crate::{mcts::Node, rand_game::random_game};
use crate::network::Network;
use onitama_move_gen::gen::Game;
use rand::random;
use tensor::*;

// Self-play
const GAMES_PER_BATCH: u32 = 500;
const ROLLOUTS_PER_MOVE: u32 = 100;

struct IncompleteTrainingExample {
    game: Game,
    improved_policy: Vector<f64, 625>,
}

impl IncompleteTrainingExample {
    pub fn update_result(self, result: f64) -> TrainingExample {
        TrainingExample {
            game: self.game,
            improved_policy: self.improved_policy,
            result
        }
    }
}

struct TrainingExample {
    game: Game,
    improved_policy: Vector<f64, 625>,
    result: f64,
}

fn self_play(network: &Network) -> Vec<TrainingExample> {
    let mut training = Vec::new();

    // Run multiple games against self.
    for _ in 0..GAMES_PER_BATCH {
        let mut game_training = Vec::new();
        // Create and play out a game while keeping track of examples.
        let mut node = Node::random();
        loop {
            if node.game.is_loss() { break; }
            for _ in 0..ROLLOUTS_PER_MOVE {
                node.rollout(network);
            }
            game_training.push(IncompleteTrainingExample {
                game: node.game,
                improved_policy: Vector::new(node.improved_policy()),
            });
            let move_index = node.pick_move();
            node = node.step(move_index);
        };

        // Go through incomplete examples and fill in the game result.
        let mut val = 1.;
        for example in game_training.into_iter().rev() {
            training.push(example.update_result(val));
            // Changing player perspective.
            val = -val;
        } 
    }

    training
}

