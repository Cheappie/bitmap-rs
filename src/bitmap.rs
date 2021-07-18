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
        self.transform_range(range, |_| WORD_MASK_I64)
    }

    fn transform_range(&mut self, range: Range<u32>, transformer: fn(i64) -> i64) {
        if range.is_empty() {
            return;
        }

        let start_index = index_word(range.start);
        let end_index = index_word(range.end - 1);

        let new_len = end_index + 1;
        self.grow(new_len);

        if start_index == end_index {
            let word = self.words[start_index];
            self.words[start_index] = merge_origin_with_transformed(
                word,
                transformer(word),
                single_word_range_mask(range),
            );
        } else {
            let first = self.words[start_index];
            self.words[start_index] =
                merge_origin_with_transformed(first, transformer(first), low_bit_mask(range.start));

            for i in (start_index + 1)..=(end_index - 1) {
                self.words[i] = transformer(self.words[i]);
            }

            let last = self.words[end_index];
            self.words[end_index] =
                merge_origin_with_transformed(last, transformer(last), high_bit_mask(range.end));
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
        self.transform_range(range, |_| 0);
        self.truncate_word_array();
    }

    pub fn flip(&mut self, bit: u32) {
        let word_index = index_word(bit);
        self.grow(word_index + 1);

        if let Some(w) = self.words.get_mut(word_index) {
            *w ^= 1i64 << offset_word(bit);
        }

        self.truncate_word_array();
    }

    pub fn flip_range(&mut self, range: Range<u32>) {
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
                return Some((i as u32) * i64::BITS + (i64::BITS - (w.leading_zeros() + 1)));
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

        let start_index = index_word(range.start);
        let end_index = index_word(range.end - 1);

        if start_index == end_index {
            (self.words[start_index] & single_word_range_mask(range)).count_ones()
        } else {
            let mut cardinality: u32 =
                (self.words[start_index] & low_bit_mask(range.start)).count_ones();

            for i in (start_index + 1)..=(end_index - 1) {
                cardinality += self.words[i].count_ones();
            }

            cardinality += (self.words[end_index] & high_bit_mask(range.end)).count_ones();
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

    pub fn iter(&self) -> Iter {
        Iter::new(&self)
    }

    pub fn iter_rev(&self) -> IterRev {
        IterRev::new(&self)
    }

    pub fn iter_rank(&self) -> IterRank {
        IterRank::new(self.iter())
    }
}

pub struct Iter<'a> {
    bitmap: &'a Bitmap,
    forward_bit: Option<u32>,
}

impl<'a> Iter<'a> {
    pub fn new(bitmap: &'a Bitmap) -> Iter<'a> {
        Iter {
            bitmap,
            forward_bit: None,
        }
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        let from = self.forward_bit.map(|p| p + 1).unwrap_or(u32::MIN);
        if let Some(next) = self.bitmap.next_set_bit(from) {
            self.forward_bit.insert(next);
            Some(next)
        } else {
            None
        }
    }

    // optimize ?
    // fn advance_by(&mut self, n: usize) -> Result<(), usize> {
    //     todo!()
    // }

    // optimize
    // fn skip_while<P>(self, predicate: P) -> SkipWhile<Self, P>
    // where
    //     Self: Sized,
    //     P: FnMut(&Self::Item) -> bool,
    // {
    //     Iterator::skip_while(self, predicate)
    // }
    //
    // // optimize
    // fn skip(self, n: usize) -> Skip<Self>
    // where
    //     Self: Sized,
    // {
    //     Iterator::skip(self, n)
    // }
}

pub struct IterRev<'a> {
    bitmap: &'a Bitmap,
    backward_bit: Option<u32>,
}

impl<'a> IterRev<'a> {
    pub fn new(bitmap: &'a Bitmap) -> IterRev<'a> {
        IterRev {
            bitmap,
            backward_bit: Some(u32::MAX),
        }
    }
}

impl<'a> Iterator for IterRev<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        let from = self.backward_bit.filter(|&x| x > 0).map(|x| x - 1);
        if let Some(next) = from.and_then(|x| self.bitmap.previous_set_bit(x)) {
            self.backward_bit.insert(next);
            Some(next)
        } else {
            None
        }
    }

    // optimize ?
    // fn advance_by(&mut self, n: usize) -> Result<(), usize> {
    //     todo!()
    // }

    // // optimize
    // fn skip_while<P>(self, predicate: P) -> SkipWhile<Self, P>
    // where
    //     Self: Sized,
    //     P: FnMut(&Self::Item) -> bool,
    // {
    //     Iterator::skip_while(self, predicate)
    // }
    //
    // // optimize
    // fn skip(self, n: usize) -> Skip<Self>
    // where
    //     Self: Sized,
    // {
    //     Iterator::skip(self, n)
    // }
}

pub struct IterRank<'a> {
    iter: Iter<'a>,
    cardinality: u32,
}

impl<'a> IterRank<'a> {
    pub fn new(iter: Iter<'a>) -> IterRank {
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

impl Iterator for IterRank<'_> {
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

/// start inclusive
///
///
#[inline]
fn low_bit_mask(start: u32) -> i64 {
    (WORD_MASK_U64 << offset_word(start)) as i64
}

/// end exclusive
///
///
#[inline]
fn high_bit_mask(end: u32) -> i64 {
    (WORD_MASK_U64 >> ((64 - offset_word(end)) & 0x3F)) as i64
}

#[inline]
fn merge_origin_with_transformed(origin: i64, transformed: i64, mask: i64) -> i64 {
    let mut res = transformed;
    res &= mask;
    res |= origin & !mask;
    res
}

#[cfg(test)]
mod tests {
    use crate::bitmap::{Bitmap, Rank};

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
    fn flip_range() {}

    #[test]
    fn insert_range() {}

    #[test]
    fn remove_range() {}

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

        let mut iter = bitmap.iter_rev();

        //when
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
    fn iter_rank_ddmulti_word() {
        //given
        let mut bitmap = Bitmap::new();

        bitmap.insert(0);
        bitmap.insert(100);
        bitmap.insert(101);
        bitmap.insert(1024);

        println!("{}", bitmap);
    }
}

#[cfg(test)]
mod test_previous_next_jumps {
    use crate::bitmap::Bitmap;

    #[test]
    fn find_closest_next_set_bit() {
        //given
        let mut bitmap = Bitmap::new();

        //when
        bitmap.insert_many(&vec![30, 40, 65, 70]);

        //then
        assert_eq!(30, bitmap.next_set_bit(0).unwrap());
        assert_eq!(30, bitmap.next_set_bit(29).unwrap());
        assert_eq!(30, bitmap.next_set_bit(30).unwrap());

        assert_eq!(40, bitmap.next_set_bit(31).unwrap());
        assert_eq!(40, bitmap.next_set_bit(40).unwrap());

        assert_eq!(65, bitmap.next_set_bit(63).unwrap());
        assert_eq!(65, bitmap.next_set_bit(64).unwrap());
        assert_eq!(65, bitmap.next_set_bit(65).unwrap());

        assert_eq!(70, bitmap.next_set_bit(66).unwrap());
        assert_eq!(70, bitmap.next_set_bit(70).unwrap());

        assert!(bitmap.next_set_bit(71).is_none());
        assert!(bitmap.next_set_bit(1024).is_none());
    }

    #[test]
    fn find_closest_previous_set_bit() {
        //given
        let mut bitmap = Bitmap::new();

        //when
        bitmap.insert_many(&vec![30, 40, 65, 70]);

        //then
        assert!(bitmap.previous_set_bit(0).is_none());
        assert!(bitmap.previous_set_bit(29).is_none());

        assert_eq!(30, bitmap.previous_set_bit(30).unwrap());
        assert_eq!(30, bitmap.previous_set_bit(39).unwrap());

        assert_eq!(40, bitmap.previous_set_bit(40).unwrap());
        assert_eq!(40, bitmap.previous_set_bit(63).unwrap());
        assert_eq!(40, bitmap.previous_set_bit(64).unwrap());

        assert_eq!(65, bitmap.previous_set_bit(65).unwrap());
        assert_eq!(65, bitmap.previous_set_bit(69).unwrap());

        assert_eq!(70, bitmap.previous_set_bit(70).unwrap());
        assert_eq!(70, bitmap.previous_set_bit(71).unwrap());
        assert_eq!(70, bitmap.previous_set_bit(1024).unwrap());
    }

    #[test]
    fn find_closest_next_clear_bit() {
        let mut bitmap = Bitmap::new();
        bitmap.insert_many(&vec![30, 40, 64, 65, 70]);

        assert_eq!(0, bitmap.next_clear_bit(0).unwrap());
        assert_eq!(29, bitmap.next_clear_bit(29).unwrap());
        assert_eq!(31, bitmap.next_clear_bit(30).unwrap());
        assert_eq!(66, bitmap.next_clear_bit(64).unwrap());
        assert_eq!(69, bitmap.next_clear_bit(69).unwrap());
        assert_eq!(71, bitmap.next_clear_bit(70).unwrap());
        assert_eq!(120, bitmap.next_clear_bit(120).unwrap());
        assert_eq!(1024, bitmap.next_clear_bit(1024).unwrap());
    }

    #[test]
    fn find_closest_previous_clear_bit() {
        let mut bitmap = Bitmap::new();
        bitmap.insert_many(&vec![30, 40, 64, 65, 70]);
        bitmap.insert_range(0..30);

        assert_eq!(1024, bitmap.previous_clear_bit(1024).unwrap());
        assert_eq!(71, bitmap.previous_clear_bit(71).unwrap());
        assert_eq!(69, bitmap.previous_clear_bit(70).unwrap());
        assert_eq!(66, bitmap.previous_clear_bit(66).unwrap());
        assert_eq!(63, bitmap.previous_clear_bit(65).unwrap());
        assert_eq!(31, bitmap.previous_clear_bit(31).unwrap());

        assert!(bitmap.previous_clear_bit(30).is_none());
        assert!(bitmap.previous_clear_bit(0).is_none());
    }
}
