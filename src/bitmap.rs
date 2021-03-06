use std::fmt::{Display, Formatter};
use std::ops::Range;

const ADDRESS_BITS_PER_WORD: usize = 6;

const WORD_MASK_U64: u64 = u64::MAX;
const WORD_MASK_I64: i64 = u64::MAX as i64;

pub struct Bitmap {
    words: Vec<i64>,
}

impl Clone for Bitmap {
    fn clone(&self) -> Self {
        Self {
            words: self.words.clone(),
        }
    }
}

impl Display for Bitmap {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Bitmap: {:?}", self.iter().collect::<Vec<u32>>())
    }
}

impl Bitmap {
    pub fn new() -> Bitmap {
        Bitmap { words: Vec::new() }
    }

    pub fn insert(&mut self, bit: u32) {
        let word_index = index_word(bit);
        self.grow(word_index + 1);

        if let Some(w) = self.words.get_mut(word_index) {
            *w |= 1i64 << offset_word(bit);
        }
    }

    fn grow(&mut self, new_len: usize) {
        if new_len > self.words.len() {
            self.words.resize(new_len, 0);
        }
    }

    pub fn insert_many(&mut self, bits: &[u32]) {
        for &bit in bits {
            self.insert(bit);
        }
    }

    pub fn insert_range(&mut self, range: Range<u32>) {
        if range.is_empty() {
            return;
        }

        let new_len = index_word(range.end - 1) + 1;
        self.grow(new_len);

        self.transform_range(range, |_| WORD_MASK_I64)
    }

    fn transform_range(&mut self, range: Range<u32>, transformer: fn(i64) -> i64) {
        let start_index = index_word(range.start);
        let end_index = index_word(range.end - 1);

        if start_index == end_index {
            let word = self.words[start_index];
            self.words[start_index] = merge_origin_with_transformed(
                word,
                transformer(word),
                single_word_range_mask(range),
            );
        } else {
            let len = self.words.len();
            let (end_index, end_mask) = if end_index < len {
                (end_index, high_bit_mask(range.end))
            } else {
                (len - 1, WORD_MASK_I64)
            };

            let first = self.words[start_index];
            self.words[start_index] =
                merge_origin_with_transformed(first, transformer(first), low_bit_mask(range.start));

            for i in (start_index + 1)..=(end_index - 1) {
                self.words[i] = transformer(self.words[i]);
            }

            let last = self.words[end_index];
            self.words[end_index] =
                merge_origin_with_transformed(last, transformer(last), end_mask);
        }
    }

    pub fn remove(&mut self, bit: u32) {
        let word_index = index_word(bit);
        if let Some(w) = self.words.get_mut(word_index) {
            *w &= !(1i64 << offset_word(bit));
            self.truncate_word_array();
        }
    }

    fn truncate_word_array(&mut self) {
        for i in (0..self.words.len()).rev() {
            let w = self.words[i];
            if w != 0 {
                self.words.truncate(i + 1);
                break;
            }
        }
    }

    pub fn remove_many(&mut self, bits: &[u32]) {
        for &bit in bits {
            self.remove(bit);
        }
    }

    pub fn remove_range(&mut self, range: Range<u32>) {
        if range.is_empty() || self.words.len() == 0 {
            return;
        }

        self.transform_range(range, |_| 0);
        self.truncate_word_array();
    }

    pub fn flip(&mut self, bit: u32) {
        let word_index = index_word(bit);
        self.grow(word_index + 1);

        if let Some(w) = self.words.get_mut(word_index) {
            *w ^= 1i64 << offset_word(bit);
            self.truncate_word_array();
        }
    }

    pub fn flip_range(&mut self, range: Range<u32>) {
        if range.is_empty() {
            return;
        }

        let new_len = index_word(range.end - 1) + 1;
        self.grow(new_len);

        self.transform_range(range, |w| !w);
        self.truncate_word_array();
    }

    pub fn contains(&self, bit: u32) -> bool {
        self.words
            .get(index_word(bit))
            .map_or(false, |&w| (w & (1i64 << offset_word(bit))) != 0)
    }

    pub fn contains_binary(&self, bit: u32) -> u32 {
        self.words
            .get(index_word(bit))
            .map_or(0, |&w| ((w >> offset_word(bit)) & 1) as u32)
    }

    pub fn first(&self) -> Option<u32> {
        for (i, &w) in self.words.iter().enumerate() {
            if w != 0 {
                return Some((i as u32) * i64::BITS + w.trailing_zeros());
            }
        }

        None
    }

    pub fn last(&self) -> Option<u32> {
        for i in (0..self.words.len()).rev() {
            let w = self.words[i];
            if w != 0 {
                return Some((i as u32) * i64::BITS + (i64::BITS - w.leading_zeros() - 1));
            }
        }

        None
    }

    pub fn next_set_bit(&self, from: u32) -> Option<u32> {
        let word_index = index_word(from);
        self.words
            .get(word_index)
            .map(|&w| w & low_bit_mask(from))
            .filter(|&w| w != 0)
            .map(|w| (word_index as u32) * i64::BITS + w.trailing_zeros())
            .or_else(|| {
                for i in (word_index + 1)..self.words.len() {
                    let w = self.words[i];
                    if w != 0 {
                        return Some((i as u32) * i64::BITS + w.trailing_zeros());
                    }
                }

                None
            })
    }

    pub fn previous_set_bit(&self, from: u32) -> Option<u32> {
        let word_index = index_word(from);
        self.words
            .get(word_index)
            .map(|&w| w & high_bit_mask(from + 1))
            .filter(|&w| w != 0)
            .map(|w| (word_index as u32) * i64::BITS + (i64::BITS - w.leading_zeros() - 1))
            .or_else(|| {
                let upper_bound = word_index.min(self.words.len());
                for i in (0..upper_bound).rev() {
                    let w = self.words[i];
                    if w != 0 {
                        return Some((i as u32) * i64::BITS + (i64::BITS - w.leading_zeros() - 1));
                    }
                }

                None
            })
    }

    pub fn next_clear_bit(&self, from: u32) -> Option<u32> {
        let word_index = index_word(from);
        if word_index >= self.words.len() {
            Some(from)
        } else {
            let mut i = word_index;
            let mut w = !self.words[word_index] & low_bit_mask(from);

            loop {
                if w != 0 {
                    return Some((i as u32) * i64::BITS + w.trailing_zeros());
                } else if i == self.words.len() {
                    return Some((i as u32) * i64::BITS);
                }

                i += 1;
                w = !self.words[i];
            }
        }
    }

    pub fn previous_clear_bit(&self, from: u32) -> Option<u32> {
        let word_index = index_word(from);
        if word_index >= self.words.len() {
            Some(from)
        } else {
            let mut i = word_index;
            let mut w = !self.words[word_index] & high_bit_mask(from + 1);

            loop {
                if w != 0 {
                    return Some((i as u32) * i64::BITS + (i64::BITS - w.leading_zeros() - 1));
                } else if i == 0 {
                    return None;
                }

                i -= 1;
                w = !self.words[i];
            }
        }
    }

    pub fn clear(&mut self) {
        self.words.clear();
    }

    pub fn trim(&mut self) {
        self.words.shrink_to_fit();
    }

    pub fn and(&mut self, other: &Bitmap) {
        let intersection = self.words.len().min(other.words.len());
        self.words.resize(intersection, 0);

        for i in 0..intersection {
            self.words[i] &= other.words[i];
        }

        self.truncate_word_array();
    }

    pub fn and_not(&mut self, other: &Bitmap) {
        let intersection = self.words.len().min(other.words.len());
        self.words.resize(intersection, 0);

        for i in 0..intersection {
            self.words[i] &= !other.words[i];
        }

        self.truncate_word_array();
    }

    pub fn or(&mut self, other: &Bitmap) {
        let intersection = self.words.len().min(other.words.len());
        for i in 0..intersection {
            self.words[i] |= other.words[i];
        }

        let new_len = self.words.len().max(other.words.len());
        self.grow(new_len);

        for i in intersection..other.words.len() {
            self.words[i] = other.words[i];
        }
    }

    pub fn or_not(&mut self, other_bitmap: &Bitmap, range_end: u32) {
        if range_end == 0 {
            return;
        }

        let end = index_word(range_end - 1) + 1;
        self.grow(end);

        let upper = end - 1;

        for i in 0..upper {
            let this = self.words.get_mut(i);
            let other = other_bitmap.words.get(i);

            match (this, other) {
                (Some(a), Some(&b)) => {
                    *a |= !b;
                }
                (None, Some(_)) => {
                    self.panic_on_or_not_operation(other_bitmap, upper);
                }
                _ => {
                    let start = i as u32 * i64::BITS;
                    let end = upper as u32 * i64::BITS;
                    self.insert_range(start..end);
                    break;
                }
            }
        }

        let last = upper;
        match (self.words.get_mut(last), other_bitmap.words.get(last)) {
            (Some(a), Some(&b)) => {
                *a |= !b & high_bit_mask(range_end);
            }
            (None, Some(_)) => {
                self.panic_on_or_not_operation(other_bitmap, upper);
            }
            _ => {
                let start = upper as u32 * i64::BITS;
                let end = start + (i64::BITS - range_end.leading_zeros() - 1);
                self.insert_range(start..end);
            }
        }

        self.truncate_word_array();
    }

    fn panic_on_or_not_operation(&self, other: &Bitmap, upper_boundary: usize) {
        panic!(
            "unreachable, calling grow before iteration should ensure correctness \
                                {{ self_len: {}, other_len: {}, upper: {} }}",
            self.words.len(),
            other.words.len(),
            upper_boundary
        );
    }

    pub fn xor(&mut self, other: &Bitmap) {
        let intersection = self.words.len().min(other.words.len());
        for i in 0..intersection {
            self.words[i] ^= other.words[i];
        }

        let new_len = self.words.len().max(other.words.len());
        self.grow(new_len);

        for i in intersection..other.words.len() {
            self.words[i] = other.words[i];
        }

        self.truncate_word_array();
    }

    pub fn cardinality(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    pub fn cardinality_in_range(&self, range: Range<u32>) -> u32 {
        if range.is_empty() {
            return 0;
        }

        let len = self.words.len();
        if len == 0 {
            return 0;
        }

        let start_index = index_word(range.start);
        let end_index = index_word(range.end - 1);

        if start_index == end_index {
            (self.words[start_index] & single_word_range_mask(range)).count_ones()
        } else {
            let (end_index, end_mask) = if end_index < len {
                (end_index, high_bit_mask(range.end))
            } else {
                (len - 1, WORD_MASK_I64)
            };

            let mut cardinality: u32 =
                (self.words[start_index] & low_bit_mask(range.start)).count_ones();

            for i in (start_index + 1)..=(end_index - 1) {
                cardinality += self.words[i].count_ones();
            }

            cardinality += (self.words[end_index] & end_mask).count_ones();
            cardinality
        }
    }

    pub fn and_cardinality(&self, other: &Bitmap) -> u32 {
        let intersection = self.words.len().min(other.words.len());
        let mut cardinality = 0;

        for i in 0..intersection {
            cardinality += (self.words[i] & other.words[i]).count_ones()
        }

        cardinality
    }

    pub fn and_not_cardinality(&self, other: &Bitmap) -> u32 {
        let intersection = self.words.len().min(other.words.len());
        let mut cardinality = 0;

        for i in 0..intersection {
            cardinality += (self.words[i] & (!other.words[i])).count_ones()
        }

        cardinality
    }

    pub fn or_cardinality(&self, other: &Bitmap) -> u32 {
        let intersection = self.words.len().min(other.words.len());
        let mut cardinality = 0;

        for i in 0..intersection {
            cardinality += (self.words[i] | other.words[i]).count_ones()
        }

        for i in intersection..self.words.len() {
            cardinality += self.words[i].count_ones();
        }

        for i in intersection..other.words.len() {
            cardinality += other.words[i].count_ones();
        }

        cardinality
    }

    pub fn or_not_cardinality(&self, other: &Bitmap, range_end: u32) -> u32 {
        if range_end == 0 {
            return 0;
        }

        let end = index_word(range_end - 1) + 1;
        let upper = end - 1;

        let mut cardinality = 0;

        for i in 0..upper {
            let this = self.words.get(i);
            let other = other.words.get(i);

            match (this, other) {
                (Some(&a), Some(&b)) => {
                    cardinality += (a | !b).count_ones();
                }
                (None, Some(&b)) => {
                    cardinality += (!b).count_ones();
                }
                _ => {
                    cardinality += ((upper - i) as u32) * i64::BITS;
                    break;
                }
            }
        }

        let last = upper;
        match (self.words.get(last), other.words.get(last)) {
            (Some(&a), Some(&b)) => {
                cardinality += ((a | !b) & high_bit_mask(range_end)).count_ones();
            }
            (None, Some(&b)) => {
                cardinality += (!b & high_bit_mask(range_end)).count_ones();
            }
            _ => {
                cardinality += i64::BITS - range_end.leading_zeros() - 1;
            }
        }

        cardinality
    }

    pub fn xor_cardinality(&self, other: &Bitmap) -> u32 {
        let intersection = self.words.len().min(other.words.len());
        let mut cardinality = 0;

        for i in 0..intersection {
            cardinality += (self.words[i] ^ other.words[i]).count_ones()
        }

        for i in intersection..self.words.len() {
            cardinality += self.words[i].count_ones();
        }

        for i in intersection..other.words.len() {
            cardinality += other.words[i].count_ones();
        }

        cardinality
    }

    pub fn intersects(&self, other: &Bitmap) -> bool {
        let intersection = self.words.len().min(other.words.len());

        for i in 0..intersection {
            if (self.words[i] & other.words[i]) != 0 {
                return true;
            }
        }

        return false;
    }

    pub fn is_subset_of(&self, other: &Bitmap) -> bool {
        if self.words.len() > other.words.len() {
            return false;
        }

        let mut self_cardinality = 0;
        let mut and_cardinality = 0;

        let intersection = self.words.len().min(other.words.len());
        for i in 0..intersection {
            self_cardinality += self.words[i].count_ones();
            and_cardinality += (self.words[i] & other.words[i]).count_ones();
        }

        self_cardinality == and_cardinality
    }

    pub fn is_superset_of(&self, other: &Bitmap) -> bool {
        other.is_subset_of(self)
    }

    pub fn is_empty(&self) -> bool {
        self.cardinality() == 0
    }

    pub fn serialized_size_bytes(&self) -> u32 {
        (i64::BITS / i8::BITS) * (self.words.len() as u32)
    }

    pub fn rank(&self, range_end: u32) -> u32 {
        self.cardinality_in_range(0..range_end)
    }

    pub fn iter(&self) -> impl BatchIterator + '_ {
        let word_iter = self
            .words
            .iter()
            .enumerate()
            .map(|(p, &w)| (p, w))
            .filter(|&(_, w)| w != 0);

        Iter::new(word_iter)
    }

    pub fn iter_rev(&self) -> impl BatchIterator + '_ {
        let word_iter = self
            .words
            .iter()
            .rev()
            .enumerate()
            .map(|(p, &w)| (p, w))
            .filter(|&(_, w)| w != 0);

        IterRev::new(word_iter, self.words.len())
    }

    pub fn iter_rank(&self) -> impl Iterator<Item = Rank> + '_ {
        IterRank::new(self.iter())
    }
}

pub trait BatchIterator: Iterator<Item = u32> {
    fn next_batch(&mut self, dst: &mut [u32]) -> u32;
}

pub struct Iter<I> {
    iter: I,
    word: Option<(usize, i64)>,
}

impl<I> Iter<I> {
    pub fn new(iter: I) -> Iter<I> {
        Iter { iter, word: None }
    }
}

impl<I> BatchIterator for Iter<I>
where
    I: Iterator<Item = (usize, i64)>,
{
    fn next_batch(&mut self, dst: &mut [u32]) -> u32 {
        let mut off = 0;

        while let Some((i, w)) = self.word.take().or_else(|| self.iter.next()) {
            let curr_cap = w.count_ones().min((dst.len() - off) as u32);
            let word_shift = i as u32 * i64::BITS;

            let mut word = w;

            for _ in 0..curr_cap {
                dst[off] = word_shift + word.trailing_zeros();
                off += 1;
                word &= word - 1;
            }

            if word != 0 {
                self.word.insert((i, word));
            }

            if off == dst.len() {
                break;
            }
        }

        off as u32
    }
}

impl<I> Iterator for Iter<I>
where
    I: Iterator<Item = (usize, i64)>,
{
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        let word = self
            .word
            .filter(|&(_, w)| w != 0)
            .or_else(|| self.iter.next());

        if let Some((i, w)) = word {
            self.word.insert((i, w & (w - 1)));
            Some(i as u32 * i64::BITS + w.trailing_zeros())
        } else {
            None
        }
    }
}

pub struct IterRev<I> {
    iter: I,
    word: Option<(usize, i64)>,
    len: usize,
}

impl<I> IterRev<I> {
    pub fn new(iter: I, len: usize) -> IterRev<I> {
        IterRev {
            iter,
            word: None,
            len,
        }
    }
}

impl<I> BatchIterator for IterRev<I>
where
    I: Iterator<Item = (usize, i64)>,
{
    fn next_batch(&mut self, dst: &mut [u32]) -> u32 {
        let mut off = 0;

        while let Some(b) = self.next() {
            if off < dst.len() {
                dst[off] = b;
                off += 1;
            } else {
                break;
            }
        }

        off as u32
    }
}

impl<I> Iterator for IterRev<I>
where
    I: Iterator<Item = (usize, i64)>,
{
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        let word = self
            .word
            .filter(|&(_, w)| w != 0)
            .or_else(|| self.iter.next());

        if let Some((i, w)) = word {
            let offset = i64::BITS - w.leading_zeros() - 1;
            self.word.insert((i, w ^ (1i64 << offset)));
            Some((self.len - i - 1) as u32 * i64::BITS + offset)
        } else {
            None
        }
    }
}

pub struct IterRank<I> {
    iter: I,
    cardinality: u32,
}

impl<I> IterRank<I> {
    pub fn new(iter: I) -> IterRank<I> {
        IterRank {
            iter,
            cardinality: 0,
        }
    }

    pub fn rank(&self) -> u32 {
        self.cardinality
    }
}

pub struct Rank {
    bit: u32,
    rank: u32,
}

impl<I> Iterator for IterRank<I>
where
    I: BatchIterator,
{
    type Item = Rank;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(next_bit) = self.iter.next() {
            self.cardinality += 1;
            Some(Rank {
                bit: next_bit,
                rank: self.cardinality,
            })
        } else {
            None
        }
    }
}

#[inline]
fn index_word(bit: u32) -> usize {
    (bit >> ADDRESS_BITS_PER_WORD) as usize
}

#[inline]
fn offset_word(bit: u32) -> u32 {
    bit & 0x3F
}

#[inline]
fn single_word_range_mask(range: Range<u32>) -> i64 {
    low_bit_mask(range.start) & high_bit_mask(range.end)
}

#[inline]
fn low_bit_mask(start: u32) -> i64 {
    (WORD_MASK_U64 << (start & 0x3F)) as i64
}

#[inline]
fn high_bit_mask(end: u32) -> i64 {
    (WORD_MASK_U64 >> ((i64::BITS - offset_word(end)) & 0x3F)) as i64
}

#[inline]
fn merge_origin_with_transformed(origin: i64, transformed: i64, mask: i64) -> i64 {
    let mut res = transformed;
    res &= mask;
    res |= origin & !mask;
    res
}

fn print_words(words: &[i64]) {
    for &w in words {
        println!("{}", binary_to_string(w));
    }
}

fn binary_to_string(word: i64) -> String {
    let mut buff = String::with_capacity(64);

    for i in (0..64).rev() {
        if (word & 1i64 << i) != 0 {
            buff.push('1');
        } else {
            buff.push('0');
        }
    }

    buff
}

#[cfg(test)]
mod tests {
    use crate::bitmap::{binary_to_string, BatchIterator, Bitmap, Rank};

    #[test]
    fn print_word() {
        let mut bitmap = Bitmap::new();

        bitmap.insert(1);
        bitmap.insert_range(10..15);
        bitmap.insert(63);

        let words = bitmap.words;
        assert_eq!(
            "1000000000000000000000000000000000000000000000000111110000000010",
            binary_to_string(words[0])
        );
    }

    #[test]
    fn should_clone_bitmap() {
        let mut bitmap = Bitmap::new();
        bitmap.insert(8);

        let clone = bitmap.clone();
        assert!(clone.contains(8));
        assert_eq!(bitmap.cardinality(), clone.cardinality());
    }

    #[test]
    fn should_estimate_cardinality_of_single_bit() {
        //given
        let mut bitmap = Bitmap::new();

        //when
        bitmap.insert(500);

        //then
        assert_eq!(1, bitmap.cardinality());
    }

    #[test]
    fn should_estimate_cardinality_of_many_bits() {
        //given
        let mut bitmap = Bitmap::new();

        //when
        bitmap.insert_many(&vec![3, 63, 64, 100, 700, 2000]);

        //then
        assert_eq!(6, bitmap.cardinality());
    }

    #[test]
    fn new_bitmap_should_be_empty() {
        //given
        let bitmap = Bitmap::new();

        //when
        let is_empty = bitmap.is_empty();

        //then
        assert!(is_empty);
    }

    #[test]
    fn should_properly_clear_bitmap() {
        //given
        let mut bitmap = Bitmap::new();
        bitmap.insert_many(&vec![1, 2, 5, 7]);

        //when
        bitmap.clear();

        //then
        assert!(bitmap.is_empty());
    }

    #[test]
    fn should_identify_first_set_bit() {
        let mut bitmap = Bitmap::new();

        bitmap.insert_many(&vec![5, 63, 64, 100, 1024]);

        assert_eq!(5, bitmap.first().unwrap());
        bitmap.remove(5);

        assert_eq!(63, bitmap.first().unwrap());
        bitmap.remove(63);

        assert_eq!(64, bitmap.first().unwrap());
        bitmap.remove(64);

        assert_eq!(100, bitmap.first().unwrap());
        bitmap.remove(100);

        assert_eq!(1024, bitmap.first().unwrap());
        bitmap.remove(1024);

        assert!(bitmap.first().is_none());
    }

    #[test]
    fn should_identify_last_set_bit() {
        let mut bitmap = Bitmap::new();

        bitmap.insert_many(&vec![5, 63, 64, 100, 1024]);

        assert_eq!(1024, bitmap.last().unwrap());
        bitmap.remove(1024);

        assert_eq!(100, bitmap.last().unwrap());
        bitmap.remove(100);

        assert_eq!(64, bitmap.last().unwrap());
        bitmap.remove(64);

        assert_eq!(63, bitmap.last().unwrap());
        bitmap.remove(63);

        assert_eq!(5, bitmap.last().unwrap());
        bitmap.remove(5);

        assert!(bitmap.last().is_none());
    }

    #[test]
    fn should_remove_bit() {
        let mut bitmap = Bitmap::new();

        bitmap.insert_many(&vec![5, 63, 64]);
        bitmap.remove(64);

        assert!(!bitmap.contains(64));
    }

    #[test]
    fn should_remove_many_bits() {
        let mut bitmap = Bitmap::new();

        bitmap.insert_many(&vec![5, 63, 64, 100, 1024]);

        let removed_bits = vec![63, 100, 5, 1024];
        bitmap.remove_many(&removed_bits);

        for bit in removed_bits {
            assert!(!bitmap.contains(bit));
        }
    }

    #[test]
    fn should_insert_bit() {
        //given
        let mut bitmap = Bitmap::new();

        //when
        bitmap.insert(50);

        //then
        assert!(bitmap.contains(50));
    }

    #[test]
    fn should_insert_many_bits() {
        //given
        let mut bitmap = Bitmap::new();
        let bits = vec![5, 63, 64, 100, 1024];

        //when
        bitmap.insert_many(&bits);

        //then
        for bit in bits {
            assert!(bitmap.contains(bit));
        }
    }

    #[test]
    fn contains_binary_gives_1_when_bit_is_set() {
        //given
        let mut bitmap = Bitmap::new();

        //when
        bitmap.insert_range(0..1000);

        //then
        assert_eq!(1, bitmap.contains_binary(32));
    }

    #[test]
    fn contains_binary_gives_0_when_bit_is_not_set() {
        //given
        let bitmap = Bitmap::new();

        //then
        assert_eq!(0, bitmap.contains_binary(100));
    }

    #[test]
    fn contains_gives_true_when_bit_is_set() {
        //given
        let mut bitmap = Bitmap::new();

        //when
        bitmap.insert(100);

        //then
        assert!(bitmap.contains(100));
    }

    #[test]
    fn contains_gives_false_when_bit_is_not_set() {
        //given
        let bitmap = Bitmap::new();

        //when
        let contains = bitmap.contains(100);

        //then
        assert!(!contains);
    }

    #[test]
    fn flip_clear_bit() {
        //given
        let mut bitmap = Bitmap::new();
        bitmap.insert(100);

        //when
        bitmap.flip(100);

        //then
        assert!(!bitmap.contains(100));
    }

    #[test]
    fn flip_set_bit() {
        //given
        let mut bitmap = Bitmap::new();

        //when
        bitmap.flip(100);

        //then
        assert!(bitmap.contains(100));
    }

    #[test]
    fn intersects_gives_true_if_two_sets_overlap() {
        //given
        let mut b1 = Bitmap::new();
        b1.insert(512);

        let mut b2 = Bitmap::new();
        b2.insert(512);

        //when
        let intersects = b1.intersects(&b2);

        //then
        assert!(intersects);
    }

    #[test]
    fn intersects_gives_false_if_two_sets_dont_overlap() {
        //given
        let mut b1 = Bitmap::new();
        let mut b2 = Bitmap::new();

        b1.insert(100);
        b2.insert(99);

        //when
        let intersects = b1.intersects(&b2);

        //then
        assert!(!intersects);
    }

    #[test]
    fn evaluate_enclosing_set_as_superset() {
        //given
        let mut b1 = Bitmap::new();
        let mut b2 = Bitmap::new();

        b1.insert(1);
        b1.insert(2);

        b2.insert(1);

        //when
        let superset = b1.is_superset_of(&b2);

        //then
        assert!(superset);
    }

    #[test]
    fn evaluate_inner_set_as_subset() {
        //given
        let mut b1 = Bitmap::new();
        let mut b2 = Bitmap::new();

        b1.insert(1);
        b1.insert(2);

        b2.insert(1);

        //when
        let subset = b2.is_subset_of(&b1);

        //then
        assert!(subset);
    }

    #[test]
    fn superset_must_completely_enclose_subset() {
        //given
        let mut b1 = Bitmap::new();
        let mut b2 = Bitmap::new();

        b1.insert(1);
        b1.insert(2);

        b2.insert(1);
        b2.insert(100);

        //when
        let superset = b1.is_superset_of(&b2);

        //then
        assert!(!superset);
    }

    #[test]
    fn subset_must_completely_be_enclosed_within_superset() {
        //given
        let mut b1 = Bitmap::new();
        let mut b2 = Bitmap::new();

        b1.insert(1);
        b1.insert(2);

        b2.insert(1);
        b2.insert(100);

        //when
        let subset = b2.is_subset_of(&b1);

        //then
        assert!(!subset);
    }

    #[test]
    fn flip_range() {
        let mut bitmap = Bitmap::new();

        bitmap.insert(63);
        bitmap.flip_range(63..65);

        assert!(!bitmap.contains(63));
        assert!(bitmap.contains(64));

        assert_eq!(1, bitmap.cardinality());
    }

    #[test]
    fn flip_empty_range_is_nop() {
        let mut bitmap = Bitmap::new();
        bitmap.flip_range(512..0);
        assert_eq!(0, bitmap.cardinality());
    }

    #[test]
    fn insert_range() {
        let mut bitmap = Bitmap::new();

        bitmap.insert_range(0..70);

        let mut cardinality_in_range = 0;
        for i in 0..70 {
            cardinality_in_range += bitmap.contains_binary(i);
        }

        assert_eq!(70, cardinality_in_range);
        assert_eq!(70, bitmap.cardinality());
    }

    #[test]
    fn insert_empty_range_is_nop() {
        let mut bitmap = Bitmap::new();
        bitmap.insert_range(512..0);
        assert_eq!(0, bitmap.cardinality());
    }

    #[test]
    fn remove_range() {
        let mut bitmap = Bitmap::new();

        bitmap.insert_range(13..20);
        bitmap.insert_range(40..70);

        bitmap.remove_range(15..69);

        assert!(bitmap.contains(13));
        assert!(bitmap.contains(14));
        assert!(bitmap.contains(69));

        assert_eq!(3, bitmap.cardinality());
    }

    #[test]
    fn remove_range_beyond_last_set_bit() {
        let mut bitmap = Bitmap::new();

        bitmap.insert_range(64..192);
        bitmap.remove_range(0..300);

        assert_eq!(0, bitmap.cardinality());
    }

    #[test]
    fn remove_range_on_empty_bitmap_is_nop() {
        let mut bitmap = Bitmap::new();
        bitmap.remove_range(100..700);
    }

    #[test]
    fn remove_empty_range_is_nop() {
        let mut bitmap = Bitmap::new();

        bitmap.insert_range(0..64);
        bitmap.remove_range(64..0); // 64..0 is an empty range

        assert_eq!(64, bitmap.cardinality());
    }

    #[test]
    fn iter_empty() {
        //given
        let bitmap = Bitmap::new();
        let mut iter = bitmap.iter();

        //when
        let result = iter.next();

        //then
        assert!(result.is_none());
    }

    #[test]
    fn iter_single() {
        //given
        let mut bitmap = Bitmap::new();
        bitmap.insert(100);
        let mut iter = bitmap.iter();

        //when
        let result = iter.next();

        //then
        assert_eq!(100, result.unwrap());
    }

    #[test]
    fn iter_multi_word() {
        //given
        let mut bitmap = Bitmap::new();

        bitmap.insert(0);
        bitmap.insert(100);
        bitmap.insert(101);
        bitmap.insert(1024);

        let mut iter = bitmap.iter();

        //when
        assert_eq!(0, iter.next().unwrap());
        assert_eq!(100, iter.next().unwrap());
        assert_eq!(101, iter.next().unwrap());
        assert_eq!(1024, iter.next().unwrap());

        assert!(iter.next().is_none());
    }

    #[test]
    fn iter_batch_empty() {
        //given
        let bitmap = Bitmap::new();
        let mut iter = bitmap.iter();

        let mut buff: Vec<u32> = vec![0; 8];

        //when
        let result = iter.next_batch(&mut buff);

        //then
        assert_eq!(0, result);
        assert_eq!(8, buff.into_iter().filter(|&x| x == 0).count());
    }

    #[test]
    fn iter_batch_over_empty_bitmap_with_empty_buffer() {
        //given
        let bitmap = Bitmap::new();
        let mut iter = bitmap.iter();

        let mut buff: Vec<u32> = vec![0; 0];

        //when
        let result = iter.next_batch(&mut buff);

        //then
        assert_eq!(0, result);
    }

    #[test]
    fn iter_batch_read_one_by_one() {
        //given
        let mut bitmap = Bitmap::new();
        bitmap.insert(5);
        bitmap.insert(100);
        bitmap.insert(500);

        let mut iter = bitmap.iter();
        let mut buff: Vec<u32> = vec![0; 1];

        //when
        let result = iter.next_batch(&mut buff);

        //then
        assert_eq!(1, result);
        assert_eq!(5, buff[0]);

        //when
        let result = iter.next_batch(&mut buff);

        //then
        assert_eq!(1, result);
        assert_eq!(100, buff[0]);

        //when
        let result = iter.next_batch(&mut buff);

        //then
        assert_eq!(1, result);
        assert_eq!(500, buff[0]);
    }

    #[test]
    fn iter_batch_multi_word() {
        //given
        let mut bitmap = Bitmap::new();

        bitmap.insert(0);
        bitmap.insert(100);
        bitmap.insert(101);
        bitmap.insert(1024);

        let mut iter = bitmap.iter();
        let mut buff: Vec<u32> = vec![0; 3];

        //when
        let read = iter.next_batch(&mut buff);

        //then
        assert_eq!(3, read);
        assert_eq!(0, buff[0]);
        assert_eq!(100, buff[1]);
        assert_eq!(101, buff[2]);

        //when
        let read = iter.next_batch(&mut buff);

        //then
        assert_eq!(1, read);
        assert_eq!(1024, buff[0]);
    }

    #[test]
    fn iter_rev_empty() {
        //given
        let bitmap = Bitmap::new();
        let mut iter = bitmap.iter_rev();

        //when
        let result = iter.next();

        //then
        assert!(result.is_none());
    }

    #[test]
    fn iter_rev_single() {
        //given
        let mut bitmap = Bitmap::new();
        bitmap.insert(100);
        let mut iter = bitmap.iter_rev();

        //when
        let result = iter.next();

        //then
        assert_eq!(100, result.unwrap());
    }

    #[test]
    fn iter_rev_multi_word() {
        //given
        let mut bitmap = Bitmap::new();

        bitmap.insert(0);
        bitmap.insert(100);
        bitmap.insert(101);
        bitmap.insert(1024);

        //when
        let mut iter = bitmap.iter_rev();

        //then
        assert_eq!(1024, iter.next().unwrap());
        assert_eq!(101, iter.next().unwrap());
        assert_eq!(100, iter.next().unwrap());
        assert_eq!(0, iter.next().unwrap());

        assert!(iter.next().is_none());
    }

    #[test]
    fn iter_rank_empty() {
        //given
        let bitmap = Bitmap::new();
        let mut iter = bitmap.iter_rank();

        //when
        let result = iter.next();

        //then
        assert!(result.is_none());
    }

    #[test]
    fn iter_rank_single() {
        //given
        let mut bitmap = Bitmap::new();
        bitmap.insert(100);
        let mut iter = bitmap.iter_rank();

        //when
        let result = iter.next();

        //then
        assert_rank(result, 100, 1);
    }

    #[test]
    fn iter_rank_multi_word() {
        //given
        let mut bitmap = Bitmap::new();

        bitmap.insert(0);
        bitmap.insert(100);
        bitmap.insert(101);
        bitmap.insert(1024);

        let mut iter = bitmap.iter_rank();

        //when
        assert_rank(iter.next(), 0, 1);
        assert_rank(iter.next(), 100, 2);
        assert_rank(iter.next(), 101, 3);
        assert_rank(iter.next(), 1024, 4);

        assert!(iter.next().is_none());
    }

    fn assert_rank(next: Option<Rank>, bit: u32, rank: u32) {
        let rankstruct = next.unwrap();
        assert_eq!(bit, rankstruct.bit);
        assert_eq!(rank, rankstruct.rank);
    }

    #[test]
    fn should_shrink_backing_array() {
        let mut bitmap = Bitmap::new();

        bitmap.insert(128);
        bitmap.insert(1024);

        bitmap.remove(1024);

        bitmap.trim();

        assert_eq!(bitmap.words.len(), bitmap.words.capacity());
    }

    #[test]
    fn should_properly_calculate_serialized_size() {
        let mut bitmap = Bitmap::new();

        bitmap.insert(1024);

        assert_eq!(
            (bitmap.words.len() * 8) as u32,
            bitmap.serialized_size_bytes()
        );
    }

    #[test]
    fn rank() {
        let mut bitmap = Bitmap::new();

        bitmap.insert(1);
        bitmap.insert(3);
        bitmap.insert(5);

        bitmap.insert(150);

        assert_eq!(0, bitmap.rank(1));
        assert_eq!(1, bitmap.rank(2));
        assert_eq!(1, bitmap.rank(3));
        assert_eq!(2, bitmap.rank(4));
        assert_eq!(2, bitmap.rank(5));

        assert_eq!(3, bitmap.rank(150));
        assert_eq!(4, bitmap.rank(151));
        assert_eq!(4, bitmap.rank(500));
    }

    #[test]
    fn cardinality_in_range() {
        let mut bitmap = Bitmap::new();

        bitmap.insert_range(127..260);
        bitmap.insert(266);
        bitmap.insert(340);
        bitmap.insert_range(344..420);
        bitmap.insert(500);

        assert_eq!(23, bitmap.cardinality_in_range(0..150));
        assert_eq!(23, bitmap.cardinality_in_range(127..150));

        assert_eq!(133, bitmap.cardinality_in_range(127..266));
        assert_eq!(134, bitmap.cardinality_in_range(127..267));

        assert_eq!(168, bitmap.cardinality_in_range(150..400));

        assert_eq!(79, bitmap.cardinality_in_range(266..800));
    }

    #[test]
    fn cardinality_in_range_over_empty_bitmap() {
        let mut bitmap = Bitmap::new();
        assert_eq!(0, bitmap.cardinality_in_range(0..u32::MAX))
    }

    #[test]
    fn cardinality_in_empty_range() {
        let mut bitmap = Bitmap::new();
        bitmap.insert_range(0..1024);
        assert_eq!(0, bitmap.cardinality_in_range(u32::MAX..0))
    }
}

#[cfg(test)]
mod test_jump_to_next_or_previous_bit {
    use crate::bitmap::Bitmap;

    #[test]
    fn find_closest_next_set_bit() {
        //given
        let mut bitmap = Bitmap::new();
        let set_bits = vec![30, 40, 63, 64, 66, 70, 260];
        bitmap.insert_many(&set_bits);

        for i in (0..512).rev() {
            let expected = set_bits.iter().copied().find(|&x| x >= i);

            if let Some(exp) = expected {
                assert_eq!(exp, bitmap.next_set_bit(i).unwrap());
            } else {
                assert!(bitmap.next_set_bit(i).is_none());
            }
        }
    }

    #[test]
    fn find_closest_previous_set_bit() {
        //given
        let mut bitmap = Bitmap::new();
        let set_bits = vec![30, 40, 63, 64, 66, 70, 260];
        bitmap.insert_many(&set_bits);

        for i in (0..512).rev() {
            let expected = set_bits.iter().copied().rev().find(|&x| x <= i);

            if let Some(exp) = expected {
                assert_eq!(exp, bitmap.previous_set_bit(i).unwrap());
            } else {
                assert!(bitmap.previous_set_bit(i).is_none());
            }
        }
    }

    #[test]
    fn find_closest_next_clear_bit() {
        //given
        let mut bitmap = Bitmap::new();
        let set_bits = vec![30, 40, 62, 64, 66, 70, 260];
        bitmap.insert_many(&set_bits);

        for i in (0..512).rev() {
            if set_bits.contains(&i) {
                assert_eq!(i + 1, bitmap.next_clear_bit(i).unwrap())
            } else {
                assert_eq!(i, bitmap.next_clear_bit(i).unwrap())
            }
        }
    }

    #[test]
    fn find_closest_previous_clear_bit() {
        //given
        let mut bitmap = Bitmap::new();
        let set_bits = vec![30, 40, 62, 64, 66, 70, 260];
        bitmap.insert_many(&set_bits);

        for i in (0..512).rev() {
            if set_bits.contains(&i) {
                assert_eq!(i - 1, bitmap.previous_clear_bit(i).unwrap())
            } else {
                assert_eq!(i, bitmap.previous_clear_bit(i).unwrap())
            }
        }
    }
}

#[cfg(test)]
mod test_base_ops {
    use crate::Bitmap;

    #[test]
    fn test_in_place_and() {
        let mut b1 = Bitmap::new();

        b1.insert(150);
        b1.insert(400);
        b1.insert_range(10..14);

        let mut b2 = Bitmap::new();

        b2.insert(151);
        b2.insert(400);
        b2.insert(700);
        b2.insert_range(13..14);

        b1.and(&b2);

        assert!(b1.contains(13));
        assert!(b1.contains(400));

        assert_eq!(2, b1.cardinality());
    }

    #[test]
    fn test_in_place_or() {
        let mut b1 = Bitmap::new();

        b1.insert(150);
        b1.insert(400);
        b1.insert_range(10..14);

        let mut b2 = Bitmap::new();

        b2.insert(151);
        b2.insert(400);
        b2.insert(700);
        b2.insert_range(13..14);

        b1.or(&b2);

        assert!(b1.contains(10));
        assert!(b1.contains(11));
        assert!(b1.contains(12));
        assert!(b1.contains(13));

        assert!(b1.contains(150));
        assert!(b1.contains(151));
        assert!(b1.contains(400));
        assert!(b1.contains(700));

        assert_eq!(8, b1.cardinality());
    }

    #[test]
    fn test_in_place_xor() {
        let mut b1 = Bitmap::new();

        b1.insert(150);
        b1.insert(400);
        b1.insert_range(10..14);

        let mut b2 = Bitmap::new();

        b2.insert(151);
        b2.insert(400);
        b2.insert(700);
        b2.insert_range(13..14);

        b1.xor(&b2);

        assert!(b1.contains(10));
        assert!(b1.contains(11));
        assert!(b1.contains(12));

        assert!(b1.contains(150));
        assert!(b1.contains(151));
        assert!(b1.contains(700));

        assert_eq!(6, b1.cardinality());
    }

    #[test]
    fn test_in_place_and_not() {
        let mut b1 = Bitmap::new();

        b1.insert(150);
        b1.insert(400);
        b1.insert_range(10..14);

        let mut b2 = Bitmap::new();

        b2.insert(151);
        b2.insert(400);
        b2.insert(700);
        b2.insert_range(13..14);

        b1.and_not(&b2);

        assert!(b1.contains(10));
        assert!(b1.contains(11));
        assert!(b1.contains(12));
        assert!(b1.contains(150));

        assert_eq!(4, b1.cardinality());
    }

    #[test]
    fn test_in_place_or_not() {
        // TODO
    }

    #[test]
    fn test_and_cardinality() {
        let mut b1 = Bitmap::new();

        b1.insert(150);
        b1.insert(400);
        b1.insert_range(10..14);

        let mut b2 = Bitmap::new();

        b2.insert(151);
        b2.insert(400);
        b2.insert(700);
        b2.insert_range(13..14);

        assert_eq!(2, b1.and_cardinality(&b2));
    }

    #[test]
    fn test_or_cardinality() {
        let mut b1 = Bitmap::new();

        b1.insert(150);
        b1.insert(400);
        b1.insert_range(10..14);

        let mut b2 = Bitmap::new();

        b2.insert(151);
        b2.insert(400);
        b2.insert(700);
        b2.insert_range(13..14);

        assert_eq!(8, b1.or_cardinality(&b2));
    }

    #[test]
    fn test_xor_cardinality() {
        let mut b1 = Bitmap::new();

        b1.insert(150);
        b1.insert(400);
        b1.insert_range(10..14);

        let mut b2 = Bitmap::new();

        b2.insert(151);
        b2.insert(400);
        b2.insert(700);
        b2.insert_range(13..14);

        assert_eq!(6, b1.xor_cardinality(&b2));
    }

    #[test]
    fn test_and_not_cardinality() {
        let mut b1 = Bitmap::new();

        b1.insert(150);
        b1.insert(400);
        b1.insert_range(10..14);

        let mut b2 = Bitmap::new();

        b2.insert(151);
        b2.insert(400);
        b2.insert(700);
        b2.insert_range(13..14);

        assert_eq!(4, b1.and_not_cardinality(&b2));
    }

    #[test]
    fn test_or_not_cardinality() {
        // TODO
    }
}
