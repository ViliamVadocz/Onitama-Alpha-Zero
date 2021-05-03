use tensor::*;
use onitama_move_gen::{SHIFTED, gen::{PIECE_MASK, Game}, ops::CardIter};

pub fn game_to_input(game: &Game) -> Tensor3<f64, 5, 5, 8> {
    let mut data = [0.; 5 * 5 * 8];

    write_bitmap_into_data(&mut data, game.my & PIECE_MASK, 0);
    write_bitmap_into_data(&mut data, game.other & PIECE_MASK, 25);
    let kings = (1 << 25 >> game.my.wrapping_shr(25)) | (1 << 25 >> game.other.wrapping_shr(25));
    write_bitmap_into_data(&mut data, kings, 50);
    for (card, &start) in CardIter::new(game.cards).map(|x| SHIFTED[x as usize][0]).zip([75, 100].iter()) {
        write_bitmap_into_data(&mut data, card, start);
    }
    for (card, &start) in CardIter::new(game.cards.wrapping_shr(16)).map(|x| SHIFTED[x as usize][0]).zip([125, 150].iter()) {
        write_bitmap_into_data(&mut data, card, start);
    }
    write_bitmap_into_data(&mut data, SHIFTED[game.table as usize][0], 175);

    Tensor3::new(data)
}

fn write_bitmap_into_data<const L: usize>(data: &mut [f64; L], bitmap: u32, start: usize) {
    for i in 0..25 {
        if (bitmap & (1 << i)) == 1 {
            data[start + i] = 1.;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

                0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0.,
                1., 1., 1., 1., 1.,

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
