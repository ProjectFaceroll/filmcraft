//! Decode an Annex-B H.264 file to raw planar YUV (yuv420p / yuv422p, 16-bit little-endian
//! samples above 8 bits): `h264dec in.h264 [out.yuv]`.
//! Prints timing information to stderr. `H264_BENCH_ITERS=n` (with `H264_THREADS=t`,
//! `H264_SLICES=0` to decode the slices of a picture one after the other, and `H264_DRAFT=1` for
//! draft mode) turns it into a benchmark.

use filmcraft_h264::Decoder;
use std::io::Write;
use std::sync::Mutex;

/// Output planes of pictures the benchmark is done with, handed back to the decoder as the app's
/// frame pool does (fresh 33 MB allocations per 4K picture would measure page faults, not decoding).
static PLANES_U8: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());
static PLANES_U16: Mutex<Vec<Vec<u16>>> = Mutex::new(Vec::new());

fn take_u8(len: usize) -> Vec<u8> {
    let mut v = PLANES_U8.lock().ok().and_then(|mut p| p.iter().position(|v| v.capacity() >= len).map(|i| p.swap_remove(i))).unwrap_or_default();
    v.clear();
    v.reserve(len);
    v
}

fn take_u16(len: usize) -> Vec<u16> {
    let mut v = PLANES_U16.lock().ok().and_then(|mut p| p.iter().position(|v| v.capacity() >= len).map(|i| p.swap_remove(i))).unwrap_or_default();
    v.clear();
    v.reserve(len);
    v
}

fn recycle(pics: Vec<filmcraft_h264::Picture>) -> usize {
    let n = pics.len();
    for p in pics {
        if let Ok(mut pool) = PLANES_U8.lock() {
            pool.extend([p.y, p.u, p.v]);
            pool.truncate(64);
        }
        if let (Some(w), Ok(mut pool)) = (p.wide, PLANES_U16.lock()) {
            pool.extend([w.y, w.u, w.v]);
            pool.truncate(64);
        }
    }
    n
}

/// The stream cut into access units (at an access unit delimiter, at a parameter set or SEI
/// that follows a slice, or at a slice with `first_mb_in_slice` 0 that follows a slice), so a
/// benchmark feeds the decoder one picture at a time as a player does, and finished pictures
/// leave the decoder as it runs.
fn access_units(data: &[u8]) -> Vec<&[u8]> {
    let base = data.as_ptr() as usize;
    let mut cuts = Vec::new();
    let mut prev_slice = false;
    for nal in filmcraft_bitstream::annexb_nals(data) {
        let off = nal.as_ptr() as usize - base;
        let t = nal[0] & 0x1f;
        let is_slice = matches!(t, 1 | 5);
        let first_mb0 = is_slice && nal.get(1).is_some_and(|b| b & 0x80 != 0);
        if cuts.is_empty() || t == 9 || (prev_slice && (matches!(t, 6 | 7 | 8 | 14 | 15) || first_mb0)) {
            // back to the start code (3 or 4 bytes)
            let mut sc = off.saturating_sub(3);
            if sc > 0 && data[sc - 1] == 0 {
                sc -= 1;
            }
            cuts.push(sc);
        }
        prev_slice = is_slice;
    }
    cuts.push(data.len());
    cuts.windows(2).map(|w| &data[w[0]..w[1]]).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: h264dec in.h264 [out.yuv]");
        std::process::exit(2);
    }
    let data = std::fs::read(&args[1]).expect("read input");
    // H264_BENCH_ITERS=n: decode n times without output and report the best run
    if let Some(n) = std::env::var("H264_BENCH_ITERS").ok().and_then(|v| v.parse::<usize>().ok()) {
        let mut best = f64::MAX;
        let mut frames = 0;
        let threads = std::env::var("H264_THREADS").ok().and_then(|v| v.parse::<usize>().ok());
        let aus = access_units(&data);
        filmcraft_h264::set_plane_allocator(take_u8);
        filmcraft_h264::set_plane_allocator_u16(take_u16);
        for _ in 0..n {
            let mut dec = threads.map(Decoder::with_threads).unwrap_or_default();
            dec.set_draft(std::env::var_os("H264_DRAFT").is_some());
            dec.set_parallel_slices(std::env::var("H264_SLICES").ok().is_none_or(|v| v != "0"));
            let t0 = std::time::Instant::now();
            frames = 0;
            for au in &aus {
                frames += recycle(dec.decode(au, 0).expect("decode"));
            }
            frames += recycle(dec.flush());
            best = best.min(t0.elapsed().as_secs_f64());
        }
        eprintln!("best of {n}: {frames} frames in {best:.3}s ({:.1} fps)", frames as f64 / best);
        let prof = filmcraft_h264::prof::take();
        if prof.iter().any(|&v| v > 0) {
            eprintln!("stage time over all runs and threads (ms, per picture):");
            let pics = (frames * n).max(1) as f64;
            for (name, ns) in filmcraft_h264::prof::STAGES.iter().zip(prof) {
                eprintln!("  {name:16} {:8.2}", ns as f64 / 1e6 / pics);
            }
        }
        return;
    }
    let mut out = args.get(2).map(|p| std::io::BufWriter::new(std::fs::File::create(p).expect("create output")));
    let mut dec = Decoder::new();
    let t0 = std::time::Instant::now();
    let mut n = 0usize;
    let mut write = |pics: Vec<filmcraft_h264::Picture>| {
        for p in pics {
            n += 1;
            if let Some(o) = out.as_mut() {
                if let Some(w) = &p.wide {
                    // little-endian 16-bit samples (yuv420p10le / yuv422p10le)
                    for plane in [&w.y, &w.u, &w.v] {
                        let bytes: Vec<u8> = plane.iter().flat_map(|v| v.to_le_bytes()).collect();
                        o.write_all(&bytes).unwrap();
                    }
                } else {
                    o.write_all(&p.y).unwrap();
                    o.write_all(&p.u).unwrap();
                    o.write_all(&p.v).unwrap();
                }
            }
        }
    };
    match dec.decode(&data, 0) {
        Ok(p) => write(p),
        Err(e) => eprintln!("decode error: {e}"),
    }
    write(dec.flush());
    let dt = t0.elapsed().as_secs_f64();
    if let Some(e) = dec.take_error() {
        eprintln!("decode error: {e}");
    }
    if std::env::var("H264_STATS").is_ok() {
        eprintln!("{:#?}", dec.stats());
    }
    eprintln!("{n} frames in {dt:.3}s ({:.1} fps)", n as f64 / dt);
}
