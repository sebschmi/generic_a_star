pub trait Reset {
    fn reset(&mut self);
}

impl Reset for () {
    #[inline]
    fn reset(&mut self) {}
}

impl<T> Reset for Option<T> {
    #[inline]
    fn reset(&mut self) {
        *self = None;
    }
}

impl<A: Reset, B: Reset> Reset for (A, B) {
    #[inline]
    fn reset(&mut self) {
        self.0.reset();
        self.1.reset();
    }
}

impl<const N: usize, T: Reset> Reset for [T; N] {
    #[inline]
    fn reset(&mut self) {
        self.iter_mut().for_each(Reset::reset);
    }
}

impl<Key, Value> Reset for rustc_hash::FxHashMapSeed<Key, Value> {
    #[inline]
    fn reset(&mut self) {
        self.clear();
    }
}
