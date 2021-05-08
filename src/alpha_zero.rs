use crate::{convert::game_to_input, mcts::Node, network::Network, rand_game::random_game};
use onitama_move_gen::gen::Game;
use rand::random;
use tensor::*;

// Self-play
const GAMES_PER_BATCH: u32 = 500;
const ROLLOUTS_PER_MOVE: u32 = 100;
// Training
const PIT_GAMES: u32 = 100;
const WIN_RATE_THRESHOLD: f64 = 0.55;

struct IncompleteTrainingExample {
    game: Game,
    improved_policy: Vector<f64, 625>,
}

impl IncompleteTrainingExample {
    pub fn update_result(self, result: f64) -> TrainingExample {
        TrainingExample {
            game: self.game,
            improved_policy: self.improved_policy,
            result,
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
            if node.game.is_loss() {
                break;
            }
            for _ in 0..ROLLOUTS_PER_MOVE {
                node.rollout(network);
            }
            game_training.push(IncompleteTrainingExample {
                game: node.game,
                improved_policy: Vector::new(node.improved_policy()),
            });
            let move_index = node.pick_move();
            node = node.step(move_index);
        }

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

struct PitResult {
    wins: u32,
    losses: u32,
}

impl PitResult {
    pub fn win_rate(&self) -> f64 {
        self.wins as f64 / (self.wins + self.losses) as f64
    }
}

/// Pits two networks against each other.
/// Counts wins and losses of the new network.
fn pit(new: &Network, old: &Network) -> PitResult {
    let mut wins = 0;
    let mut losses = 0;

    for _ in 0..PIT_GAMES {
        let mut game = random_game();
        let mut my_turn = random();
        let mut my_node = Node::from(game);
        let mut opp_node = Node::from(game);
        while !game.is_loss() {
            if my_turn {
                for _ in 0..ROLLOUTS_PER_MOVE {
                    my_node.rollout(new);
                }
                let move_index = my_node.pick_move();
                my_node = my_node.step(move_index);
                opp_node = opp_node.step(move_index);
                game = my_node.game;
            } else {
                for _ in 0..ROLLOUTS_PER_MOVE {
                    opp_node.rollout(old);
                }
                let move_index = opp_node.pick_move();
                opp_node = opp_node.step(move_index);
                my_node = my_node.step(move_index);
                game = opp_node.game;
            }
            my_turn = !my_turn;
        }
        if my_turn {
            losses += 1;
        } else {
            wins += 1;
        }
    }

    PitResult { wins, losses }
}

pub fn train_network(network: &mut Network) {
    loop {
        let training = self_play(&network);
        let mut new_network = *network;
        for example in training.into_iter() {
            new_network.back_prop(
                game_to_input(&example.game),
                example.improved_policy,
                example.result,
            );
        }
        if pit(&new_network, &network).win_rate() > WIN_RATE_THRESHOLD {
            *network = new_network;
            return;
        }
    }
}
