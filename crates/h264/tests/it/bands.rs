//! Pictures whose slices decode as parallel bands (all intra, row-aligned slices, no deblocking)
//! come out the same as with the slices decoded one after another, with and without the
//! decoder's own thread pool, and bit-exact with ffmpeg.

use crate::common;
use filmcraft_h264::{Decoder, Picture};

fn decode_all(mut dec: Decoder, data: &[u8]) -> Vec<Picture> {
    let mut out = Vec::new();
    for (i, au) in common::split_access_units(data).into_iter().enumerate() {
        out.extend(dec.decode(au, i as i64).unwrap_or_else(|e| panic!("access unit {i}: {e}")));
    }
    out.extend(dec.flush());
    assert!(dec.take_error().is_none());
    out
}

#[test]
fn band_decoding_matches_serial_slices() {
    for name in ["intra_slices_nodeblock", "intra_slices_nodeblock_cabac"] {
        let f = common::fixture(name);
        let Some((h264, yuv)) = common::ensure(f) else { return };
        let data = std::fs::read(&h264).unwrap();
        let reference = std::fs::read(&yuv).unwrap();
        for threads in [1, 3, 0] {
            let make = || if threads == 0 { Decoder::new() } else { Decoder::with_threads(threads) };
            let mut serial = make();
            serial.set_parallel_slices(false);
            let serial = decode_all(serial, &data);
            let bands = decode_all(make(), &data);
            assert_eq!(serial.len(), bands.len(), "{name} (threads {threads}): picture count");
            for (i, (a, b)) in serial.iter().zip(&bands).enumerate() {
                assert!(
                    a.y == b.y && a.u == b.u && a.v == b.v && a.wide.is_none() && b.wide.is_none(),
                    "{name} (threads {threads}): picture {i} differs between bands and serial slices"
                );
            }
            common::compare(&bands, &reference, f.width as usize, f.height as usize).unwrap_or_else(|e| panic!("{name} (threads {threads}): {e}"));
        }
    }
}
