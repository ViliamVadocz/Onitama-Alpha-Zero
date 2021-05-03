use onitama_move_gen::gen::Game;
use crate::network::Network;
use tensor::*;

struct Node {
    expected_reward: f64,
    visited_count: u32,
    policy: Vector<f64, 625>,
    children: Vec<Node>,
}

// fn perft(game: Game, depth: u8) -> u64 {
//     let mut total = 0;
//     for new_game in game.forward() {
//         if new_game.is_loss() {
//             total += 1;
//         } else if depth == 2 {
//             total += new_game.count_moves();
//         } else {
//             total += perft(new_game, depth - 1);
//         }
//     }
//     total
// }

impl Node {
    pub fn search(&mut self, game: Game, network: &Network) {
        for new_game in game.forward() {
            let from = game.my & !new_game.other;
            let to = new_game.other & !game.my;
            let move_index = from * 25 + to;
        }

    }   
}


/*
def search(s, game, nnet):
    if game.gameEnded(s): return -game.gameReward(s)

    if s not in visited:
        visited.add(s)
        P[s], v = nnet.predict(s)
        return -v
  
    max_u, best_a = -float("inf"), -1
    for a in game.getValidActions(s):
        u = Q[s][a] + c_puct*P[s][a]*sqrt(sum(N[s]))/(1+N[s][a])
        if u>max_u:
            max_u = u
            best_a = a
    a = best_a
    
    sp = game.nextState(s, a)
    v = search(sp, game, nnet)

    Q[s][a] = (N[s][a]*Q[s][a] + v)/(N[s][a]+1)
    N[s][a] += 1
    return -v
*/