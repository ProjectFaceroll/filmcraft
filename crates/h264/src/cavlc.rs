//! CAVLC residual block parsing (9.2).

use crate::cavlc_tables::{COEFF_TOKEN, RUN_BEFORE, TOTAL_ZEROS_2X2, TOTAL_ZEROS_2X4, TOTAL_ZEROS_4X4};
use crate::error::{Result, ensure, invalid};
use filmcraft_bitstream::BitReader;
use std::sync::OnceLock;

/// Prefix-code lookup keyed by (number of leading zeros, next 5 bits after the first one bit).
pub(crate) struct VlcTable {
    /// entry = (len << 8) | value, 0 = invalid.
    entries: Box<[u16; 17 * 32]>,
}

impl VlcTable {
    pub fn build(codes: &[(&str, u8)]) -> Self {
        let mut entries = Box::new([0u16; 17 * 32]);
        for &(code, value) in codes {
            if code.is_empty() {
                continue;
            }
            let len = code.len();
            let lz = code.bytes().take_while(|&b| b == b'0').count();
            let val = ((len as u16) << 8) | value as u16;
            if lz == len {
                // all-zero code: any continuation with at least `len` zeros
                for l in lz..=16 {
                    for s in 0..32 {
                        entries[l * 32 + s] = val;
                    }
                }
                continue;
            }
            let suffix = &code[lz + 1..];
            assert!(suffix.len() <= 5, "suffix too long in {code}");
            let sbits = suffix.bytes().fold(0u32, |a, b| (a << 1) | u32::from(b == b'1'));
            let free = 5 - suffix.len();
            for ext in 0..(1u32 << free) {
                let s = (sbits << free) | ext;
                entries[lz * 32 + s as usize] = val;
            }
        }
        Self { entries }
    }
}

/// A bit cursor over the slice data for the residual blocks of one macroblock: [`BitReader`]
/// without a bounds check per syntax element. The next bits wait in a 64-bit cache, so a
/// syntax element costs shifts on a register rather than a load at a bit position, and the
/// cache is topped up a whole word at a time. Reads past the end of the data see zero bits,
/// which no valid block ends in; [`Cursor::commit`] refuses a position past the data, so a
/// truncated macroblock fails there instead. A block reads at most about 1100 bits (coeff_token
/// 16, trailing ones 3, 16 levels of a prefix of at most 26 bits and a suffix of at most 22,
/// total_zeros 9 and 15 run_before codes of 11) and a macroblock has at most 48 of them, so
/// the position cannot overflow.
pub(crate) struct Cursor<'a> {
    data: &'a [u8],
    /// The next unread bits, first bit on top, `count` of them valid and the rest zero.
    cache: u64,
    /// Valid bits in `cache`: at least 32 whenever a syntax element is read.
    count: u32,
    /// Byte position of the next word to go into the cache.
    next: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(r: &BitReader<'a>) -> Result<Self> {
        let data = r.data();
        let pos = r.position();
        ensure!(pos <= data.len() * 8, "unexpected end of bitstream");
        let mut c = Self { data, cache: 0, count: 0, next: pos >> 3 };
        c.refill();
        c.skip((pos & 7) as u32);
        Ok(c)
    }

    /// Top the cache up to more than 32 bits with the next word (zero bytes past the end).
    #[inline(always)]
    fn refill(&mut self) {
        if self.count <= 32 {
            let w = match self.data.get(self.next..).and_then(<[u8]>::first_chunk::<4>) {
                Some(w) => u32::from_be_bytes(*w),
                None => {
                    let mut v = 0u32;
                    for i in 0..4 {
                        v = (v << 8) | u32::from(*self.data.get(self.next.saturating_add(i)).unwrap_or(&0));
                    }
                    v
                }
            };
            self.cache |= u64::from(w) << (32 - self.count);
            self.count += 32;
            self.next = self.next.saturating_add(4);
        }
    }

    /// Position in bits.
    #[inline(always)]
    fn pos(&self) -> usize {
        (self.next * 8).saturating_sub(self.count as usize)
    }

    /// The next `n` (1..=32) bits, zero-padded past the end.
    #[inline(always)]
    fn peek(&self, n: u32) -> u32 {
        (self.cache >> (64 - n)) as u32
    }

    /// Consume `n` (0..=32) bits.
    #[inline(always)]
    fn skip(&mut self, n: u32) {
        self.cache <<= n;
        self.count -= n.min(self.count);
        self.refill();
    }

    /// Read `n` (1..=32) bits.
    #[inline(always)]
    fn bits(&mut self, n: u32) -> u32 {
        let v = self.peek(n);
        self.skip(n);
        v
    }

    /// Decode one code of `t`.
    #[inline(always)]
    fn vlc(&mut self, t: &VlcTable) -> Result<u8> {
        let bits = self.peek(32);
        let lz = bits.leading_zeros().min(16);
        let s = if lz >= 16 { 0 } else { ((bits << lz) << 1) >> 27 };
        let e = t.entries[(lz * 32 + s) as usize];
        ensure!(e != 0, "invalid VLC code");
        self.skip(u32::from(e >> 8));
        Ok(e as u8)
    }

    /// Move `r` to this position; fails when the block read past the end of the data.
    #[inline(always)]
    pub(crate) fn commit(self, r: &mut BitReader<'a>) -> Result<()> {
        let pos = self.pos();
        ensure!(pos <= self.data.len() * 8, "unexpected end of bitstream");
        r.seek_bits(pos);
        Ok(())
    }
}

pub(crate) struct Tables {
    /// coeff_token per nC class; value = (TrailingOnes << 5) | TotalCoeff
    coeff_token: [VlcTable; 6],
    total_zeros_4x4: [VlcTable; 15],
    total_zeros_2x2: [VlcTable; 3],
    total_zeros_2x4: [VlcTable; 7],
    run_before: [VlcTable; 7],
}

pub(crate) fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let coeff_token = std::array::from_fn(|col| {
            let codes: Vec<(&str, u8)> = COEFF_TOKEN.iter().map(|&(t1, tc, ref c)| (c[col], (t1 << 5) | tc)).collect();
            VlcTable::build(&codes)
        });
        fn simple<const N: usize>(row: &[&'static str; N]) -> VlcTable {
            let codes: Vec<(&str, u8)> = row.iter().enumerate().map(|(i, &c)| (c, i as u8)).collect();
            VlcTable::build(&codes)
        }
        Tables {
            coeff_token,
            total_zeros_4x4: std::array::from_fn(|i| simple(&TOTAL_ZEROS_4X4[i])),
            total_zeros_2x2: std::array::from_fn(|i| simple(&TOTAL_ZEROS_2X2[i])),
            total_zeros_2x4: std::array::from_fn(|i| simple(&TOTAL_ZEROS_2X4[i])),
            run_before: std::array::from_fn(|i| simple(&RUN_BEFORE[i])),
        }
    })
}

/// Force table construction (so the first slice does not pay for it).
pub(crate) fn init_tables() {
    let _ = tables();
}

/// Where the coefficients of a residual block go, by scan position.
pub(crate) trait Sink {
    fn put(&mut self, scan: usize, level: i32);
}

/// Levels in scan order (blocks whose DC is dequantised separately).
impl Sink for [i32] {
    #[inline(always)]
    fn put(&mut self, scan: usize, level: i32) {
        if let Some(c) = self.get_mut(scan) {
            *c = level;
        }
    }
}

/// Dequantised coefficients written straight into the raster-order `N` x `N` block `out`
/// (8.5.12.1 / 8.5.13.1): the level at scan position `k` of the parsed block is coefficient
/// `stride * k + phase` of the block's scan `scan` (1 / 0 for a 4x4 block, 1 / 1 for the AC of
/// an Intra_16x16 or chroma block, 4 / `i4` for the four interleaved 4x4 parses of an 8x8
/// block), scaled by `ls[pos]` for the block's QP. `ac` records a coefficient past the DC.
pub(crate) struct Dequant<'a, const N: usize> {
    pub out: &'a mut [i32],
    pub scan: &'a [u8],
    pub ls: &'a [i32],
    pub qp: i32,
    pub stride: usize,
    pub phase: usize,
    pub ac: bool,
}

impl<const N: usize> Sink for Dequant<'_, N> {
    #[inline(always)]
    fn put(&mut self, k: usize, level: i32) {
        let idx = self.stride * k + self.phase;
        let (Some(&p), true) = (self.scan.get(idx), idx < N * N) else { return };
        let p = p as usize;
        let Some(&ls) = self.ls.get(p) else { return };
        let v = if N == 8 { crate::transform::scale8(level, ls, self.qp) } else { crate::transform::scale4(level, ls, self.qp) };
        if let Some(o) = self.out.get_mut(p) {
            *o = v;
        }
        self.ac |= idx > 0;
    }
}

/// Parse residual_block_cavlc(). Coefficients at scan positions `start_idx..=end_idx` go to
/// `sink` (positions it does not get stay as they are: the caller zeroes them). `nc` is the nC
/// value (-1 for 4:2:0 chroma DC, -2 for 4:2:2 chroma DC). Returns TotalCoeff.
#[inline]
pub(crate) fn residual_block<K: Sink + ?Sized>(
    t: &Tables,
    c: &mut Cursor,
    sink: &mut K,
    start_idx: usize,
    end_idx: usize,
    max_num_coeff: usize,
    nc: i32,
) -> Result<u8> {
    let col = match nc {
        -1 => 4,
        -2 => 5,
        0..=1 => 0,
        2..=3 => 1,
        4..=7 => 2,
        _ => 3,
    };
    let tok = c.vlc(&t.coeff_token[col])?;
    let total_coeff = (tok & 31) as usize;
    let trailing_ones = (tok >> 5) as usize;
    if total_coeff == 0 {
        return Ok(0);
    }
    ensure!(total_coeff <= max_num_coeff, "TotalCoeff exceeds maxNumCoeff");
    let mut level = [0i32; 16];
    if trailing_ones > 0 {
        // the sign bits of the trailing ones, first one in the top bit
        let signs = c.bits(trailing_ones as u32);
        for (i, lv) in level.iter_mut().enumerate().take(trailing_ones) {
            *lv = if (signs >> (trailing_ones - 1 - i)) & 1 == 1 { -1 } else { 1 };
        }
    }
    let mut suffix_length: u32 = if total_coeff > 10 && trailing_ones < 3 { 1 } else { 0 };
    let Some(rest) = level.get_mut(trailing_ones..total_coeff) else { return invalid("TrailingOnes exceeds TotalCoeff") };
    for (i, lv) in rest.iter_mut().enumerate() {
        let i = i + trailing_ones;
        // level_prefix, then the suffix: out of the same 32-bit word when both fit in it
        let w = c.peek(32);
        let level_prefix = w.leading_zeros();
        ensure!(level_prefix <= 25, "level_prefix too large");
        let level_suffix_size = if level_prefix == 14 && suffix_length == 0 {
            4
        } else if level_prefix >= 15 {
            level_prefix - 3
        } else {
            suffix_length
        };
        let level_suffix = if level_suffix_size == 0 {
            c.skip(level_prefix + 1);
            0
        } else if level_prefix + 1 + level_suffix_size <= 32 {
            c.skip(level_prefix + 1 + level_suffix_size);
            ((w << (level_prefix + 1)) >> (32 - level_suffix_size)) as i32
        } else {
            c.skip(level_prefix + 1);
            c.bits(level_suffix_size) as i32
        };
        let mut level_code = ((level_prefix.min(15) as i32) << suffix_length) + level_suffix;
        if level_prefix >= 15 && suffix_length == 0 {
            level_code += 15;
        }
        if level_prefix >= 16 {
            level_code += (1 << (level_prefix - 3)) - 4096;
        }
        if i == trailing_ones && trailing_ones < 3 {
            level_code += 2;
        }
        *lv = if level_code % 2 == 0 { (level_code + 2) >> 1 } else { (-level_code - 1) >> 1 };
        if suffix_length == 0 {
            suffix_length = 1;
        }
        if lv.abs() > (3 << (suffix_length - 1)) && suffix_length < 6 {
            suffix_length += 1;
        }
    }
    let mut zeros_left = if total_coeff < end_idx - start_idx + 1 {
        let tz_idx = total_coeff - 1;
        let tz = match max_num_coeff {
            4 => c.vlc(&t.total_zeros_2x2[tz_idx])?,
            8 => c.vlc(&t.total_zeros_2x4[tz_idx])?,
            _ => c.vlc(&t.total_zeros_4x4[tz_idx])?,
        } as usize;
        ensure!(tz + total_coeff <= end_idx - start_idx + 1, "total_zeros out of range");
        tz
    } else {
        0
    };
    // runs and placement: highest-frequency coefficient first
    let mut pos = start_idx + total_coeff + zeros_left - 1;
    for (i, &lv) in level.iter().enumerate().take(total_coeff) {
        sink.put(pos, lv);
        if i + 1 == total_coeff {
            break;
        }
        let run = if zeros_left > 0 {
            let run = c.vlc(&t.run_before[zeros_left.min(7) - 1])? as usize;
            ensure!(run <= zeros_left, "run_before out of range");
            zeros_left -= run;
            run
        } else {
            0
        };
        pos -= 1 + run;
    }
    Ok(total_coeff as u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_bitstream::BitWriter;

    fn kraft(codes: &[&str]) -> f64 {
        codes.iter().filter(|c| !c.is_empty()).map(|c| 0.5f64.powi(c.len() as i32)).sum()
    }

    fn prefix_free(codes: &[&str]) -> bool {
        let c: Vec<&&str> = codes.iter().filter(|c| !c.is_empty()).collect();
        for (i, a) in c.iter().enumerate() {
            for (j, b) in c.iter().enumerate() {
                if i != j && b.starts_with(**a) {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn coeff_token_tables_are_prefix_codes() {
        for col in 0..6 {
            let codes: Vec<&str> = COEFF_TOKEN.iter().map(|e| e.2[col]).collect();
            assert!(prefix_free(&codes), "column {col}");
            let k = kraft(&codes);
            assert!(k <= 1.0 + 1e-12, "column {col} kraft {k}");
        }
        // nC>=8 is a complete 6-bit fixed code except the unused 000011 pattern
        let codes: Vec<&str> = COEFF_TOKEN.iter().map(|e| e.2[3]).collect();
        assert!(codes.iter().all(|c| c.len() == 6));
    }

    #[test]
    fn total_zeros_and_run_before_are_complete_prefix_codes() {
        for (i, row) in TOTAL_ZEROS_4X4.iter().enumerate() {
            assert!(prefix_free(row));
            // tzVlcIndex 1 leaves the all-zero 9-bit code unused; all others are complete
            let k = kraft(row);
            if i == 0 {
                assert!((k - (1.0 - 1.0 / 512.0)).abs() < 1e-12, "{row:?}");
            } else {
                assert!((k - 1.0).abs() < 1e-12, "{row:?}");
            }
        }
        for row in TOTAL_ZEROS_2X2.iter() {
            assert!(prefix_free(row));
            assert!((kraft(row) - 1.0).abs() < 1e-12);
        }
        for row in TOTAL_ZEROS_2X4.iter() {
            assert!(prefix_free(row));
            assert!((kraft(row) - 1.0).abs() < 1e-12);
        }
        for row in RUN_BEFORE.iter() {
            assert!(prefix_free(row));
        }
    }

    fn write_code(w: &mut BitWriter, code: &str) {
        for b in code.bytes() {
            w.write_bit(b == b'1');
        }
    }

    #[test]
    fn decodes_spec_style_example() {
        // Block: 0 3 -1 0 | 0 -1 1 0 | 1 0 0 0 | 0 0 0 0 (zig-zag order), nC = 0.
        // TotalCoeff 5, TrailingOnes 3: coeff_token 0000 100.
        // trailing ones (reverse order): +1, -1, -1  -> sign bits 0 1 1
        // level -1? No: remaining levels (reverse): 1 (idx 4?) ... use the classic example:
        // coefficients in scan order: [0, 3, -1, 0, 0, -1, 1, 0, 1, 0, ...]
        // reverse order non-zero: 1 (pos 8), 1 (pos 6), -1 (pos 5), -1 (pos 2), 3 (pos 1)
        // TotalCoeff 5, TrailingOnes 3 (1, 1, -1).
        let mut w = BitWriter::new();
        write_code(&mut w, "0000100"); // coeff_token TC=5 T1=3, nC 0..2
        write_code(&mut w, "0"); // +1
        write_code(&mut w, "0"); // +1
        write_code(&mut w, "1"); // -1
        // level -1 with suffixLength 0: levelCode = 1 (T1 == 3 so no +2) -> prefix 1 => "01"
        write_code(&mut w, "01");
        // level 3 with suffixLength 1: levelCode = 4 -> prefix 2, suffix 0 => "001" "0"
        write_code(&mut w, "0010");
        // total_zeros = 4 (positions 0, 3, 4, 7) with tzVlcIndex 5 -> "110"
        write_code(&mut w, "110");
        // run_before: zerosLeft 4 at pos 8: run 1 -> "10"; zerosLeft 3 at pos 6: run 0 -> "11";
        // zerosLeft 3 at pos 5: run 2 -> "01"; zerosLeft 1 at pos 2: run 0 -> "1"
        write_code(&mut w, "10");
        write_code(&mut w, "11");
        write_code(&mut w, "01");
        write_code(&mut w, "1");
        w.rbsp_trailing();
        let b = w.finish();
        let mut r = BitReader::new(&b);
        let mut c = [0i32; 16];
        let mut cur = Cursor::new(&r).unwrap();
        let tc = residual_block(tables(), &mut cur, &mut c[..], 0, 15, 16, 0).unwrap();
        cur.commit(&mut r).unwrap();
        assert_eq!(tc, 5);
        assert_eq!(&c[..10], &[0, 3, -1, 0, 0, -1, 1, 0, 1, 0]);
        assert!(!r.more_rbsp_data());
    }
}
