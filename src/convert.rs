use onitama_move_gen::{
    gen::{Game, PIECE_MASK},
    ops::CardIter,
    SHIFTED,
};
use tensor::*;

const fn card_bitmap(x: u32) -> u32 {
    SHIFTED[x as usize][12]
}

/// Convert game struct to input tensor for nn.
pub fn game_to_input(game: &Game) -> Tensor3<f64, 5, 5, 8> {
    let mut data = [0.; 5 * 5 * 8];
    // Write pieces.
    write_bitmap_into_data(&mut data, game.my & PIECE_MASK, 0);
    write_bitmap_into_data(&mut data, (game.other & PIECE_MASK).reverse_bits() >> 7, 25);
    // Write kings.
    let kings = (1 << game.my.wrapping_shr(25)) | (1 << 24 >> game.other.wrapping_shr(25));
    write_bitmap_into_data(&mut data, kings, 50);
    // Write cards.
    let mut cards = CardIter::new(game.cards);
    write_bitmap_into_data(&mut data, card_bitmap(cards.next().unwrap()), 75);
    write_bitmap_into_data(&mut data, card_bitmap(cards.next().unwrap()), 100);
    cards = CardIter::new(game.cards.wrapping_shr(16));
    write_bitmap_into_data(&mut data, card_bitmap(cards.next().unwrap()), 125);
    write_bitmap_into_data(&mut data, card_bitmap(cards.next().unwrap()), 150);
    write_bitmap_into_data(&mut data, card_bitmap(game.table), 175);

    Tensor3::new(data)
}

fn write_bitmap_into_data<const L: usize>(data: &mut [f64; L], bitmap: u32, start: usize) {
    for i in 7..32 {
        if bitmap & (1 << 31 >> i) != 0 {
            data[start + i as usize - 7] = 1.;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_bitmap() {
        let mut data = [0.; 30];
        write_bitmap_into_data(&mut data, 0b10011011, 5);
        #[rustfmt::skip]
        assert_eq!(
            data,
            [
                0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0.,
                0., 0., 1., 0., 0.,
                1., 1., 0., 1., 1.,
            ]
        );
    }

    #[test]
    fn convert() {
        let game = Game {
            my: 0b11111 | 2 << 25,
            other: 0b11111 | 2 << 25,
            cards: 0b00011 | 0b01100 << 16,
            table: 4,
        };
        let input = game_to_input(&game);
        #[rustfmt::skip]
        assert_eq!(
            input,
            Tensor3::new([
                0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0.,
                1., 1., 1., 1., 1.,

                1., 1., 1., 1., 1.,
                0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0.,

                0., 0., 1., 0., 0.,
                0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0.,
                0., 0., 1., 0., 0.,

                0., 0., 0., 0., 0.,
                0., 0., 1., 0., 0.,
                0., 0., 0., 1., 0.,
                0., 0., 1., 0., 0.,
                0., 0., 0., 0., 0.,
                
                0., 0., 0., 0., 0.,
                0., 0., 1., 0., 0.,
                0., 1., 0., 1., 0.,
                0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0.,

                0., 0., 0., 0., 0.,
                0., 0., 1., 0., 0.,
                0., 1., 0., 0., 0.,
                0., 0., 1., 0., 0.,
                0., 0., 0., 0., 0.,

                0., 0., 0., 0., 0.,
                0., 1., 0., 1., 0.,
                0., 1., 0., 1., 0.,
                0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0.,

                0., 0., 0., 0., 0.,
                0., 0., 1., 0., 0.,
                1., 0., 0., 0., 1.,
                0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0.,
            ])
        );
    }
}
