use crate::convert::game_to_input;
use crate::network::Network;
use crate::rand_game::random_game;
use std::collections::HashMap;
use rand::{thread_rng, distributions::{WeightedIndex, Distribution}};
use onitama_move_gen::gen::Game;
use tensor::*;

const EXPLORATION: f64 = 1.0;

pub struct Node {
    game: Game,
    policy: f64,
    expected_reward: f64,
    visited_count: u32,
    children: Option<HashMap<usize, Node>>,
}

impl Node {
    /// Create a root node with a random game.
    pub fn init() -> Node {
        Node {
            game: random_game(),
            expected_reward: 0.,
            policy: 1.,
            visited_count: 0,
            children: None,
        }
    }

    /// Get the improved policy after MCTS.
    pub fn improved_policy(&self) -> [f64; 625] {
        let mut policy = [0.; 625];
        for (&move_index, child) in self.children.as_ref().unwrap() {
            policy[move_index] = child.expected_reward;
        }
        policy
    }

    /// Pick a random action based on policy acquired from MCTS.
    pub fn step(self) -> Node {
        let improved_policy = self.improved_policy();

        let mut rng = thread_rng();
        let distr = WeightedIndex::new(&improved_policy).unwrap();
        let selected_index = distr.sample(&mut rng);

        self.children.unwrap().remove(&selected_index).unwrap()
    }

    /// Use neural network to guide Monte Carlo tree search.
    pub fn rollout(&mut self, network: &Network) -> f64 {
        self.visited_count += 1;

        // Leaf node.
        if self.game.is_loss() {
            return -1.;
        } else if self.game.is_win() {
            return 1.;
        }

        // This is the first time we are visiting this node,
        // initialize all child states.
        if self.children.is_none() {
            // Use the neural network to get initial policy for children and eval for this board.
            let (probability_vec, eval) = network.feed_forward(game_to_input(&self.game));
            let policy = probability_vec.get_data();

            let mut children = HashMap::new();
            for game in self.game.forward() {
                let from = game.my & !game.other;
                let to = game.other & !game.my;
                let move_index = (from * 25 + to) as usize;

                let node = Node {
                    game,
                    policy: policy[move_index],
                    expected_reward: 0.0,
                    visited_count: 0,
                    children: None,
                };

                children.insert(move_index, node);
            }

            self.expected_reward = eval;
            self.children = Some(children);
            return -eval;
        }

        // We have been at this node before.
        // Pick which node to rollout.
        let mut children = self.children.take().unwrap();
        let mut max_upper_bound = f64::NEG_INFINITY;
        let mut next_node = None;
        for (_move_index, node) in children.iter_mut() {
            let upper_confidence_bound = node.expected_reward
                + EXPLORATION * node.policy * (self.visited_count as f64).sqrt()
                    / (1.0 + node.visited_count as f64);
            if upper_confidence_bound > max_upper_bound {
                max_upper_bound = upper_confidence_bound;
                next_node = Some(node);
            }
        }
        // Rollout next node.
        let eval = next_node.unwrap().rollout(&network);
        self.children = Some(children);

        // Take the mean of the expected reward and eval.
        self.expected_reward = ((self.visited_count - 1) as f64 * self.expected_reward + eval)
            / (self.visited_count as f64);

        return -eval;
    }
}
