use onitama_move_gen::gen::Game;
use rand::{
    distributions::{Distribution, Uniform},
    thread_rng,
};

pub fn random_game() -> Game {
    // Get five random cards.
    let mut rng = thread_rng();
    let distr = Uniform::new(0, 16);
    let mut cards = Vec::new();
    while cards.len() < 5 {
        let n = distr.sample(&mut rng);
        if !cards.contains(&n) {
            cards.push(n);
        }
    }
    let mut cards = cards.into_iter();
    let my_cards = 1 << cards.next().unwrap() | 1 << cards.next().unwrap();
    let other_cards = 1 << cards.next().unwrap() | 1 << cards.next().unwrap();

    Game {
        my: 0b11111 | 2 << 25,
        other: 0b11111 | 2 << 25,
        cards: my_cards | other_cards << 16,
        table: cards.next().unwrap(),
    }
}
