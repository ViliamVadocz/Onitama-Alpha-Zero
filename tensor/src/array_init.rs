pub(crate) fn array_init<T, F, const N: usize>(mut f: F) -> [T; N]
where
    F: FnMut(usize) -> T,
{
    let mut i = 0;
    [(); N].map(|()| {
        let elem = f(i);
        i += 1;
        elem
    })
}
