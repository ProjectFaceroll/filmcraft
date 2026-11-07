//! High 10 / High 4:2:2 streams (intra and long-GOP inter): bit-exactness against ffmpeg's decoder
//! on libx264 fixtures with 4:2:2 and/or 10-bit samples (skipped when ffmpeg is absent).

use crate::common;

use filmcraft_h264::{Decoder, Picture};
use std::path::PathBuf;
use std::process::Command;

/// A libx264 fixture generated with ffmpeg and checked against ffmpeg's decode.
struct HbdFixture {
    name: &'static str,
    width: u32,
    height: u32,
    frames: u32,
    /// libx264 / ffmpeg pixel format (yuv420p10le, yuv422p10le, yuv422p).
    pix_fmt: &'static str,
    args: &'static [&'static str],
    /// Filters between the test source and the pixel format conversion: `zoompan` gives every
    /// block sub-sample motion (fractional chroma vectors) and vectors pointing outside the picture.
    motion: &'static str,
}

/// Slow zoom and diagonal pan for [`HbdFixture::motion`].
/// The output size is appended from the fixture (`zoompan` defaults to 1280x720).
const ZOOM_PAN: &str = "zoompan=z='1+0.006*on':x='iw/2-iw/zoom/2+on*0.37':y='ih/2-ih/zoom/2+on*0.61':d=1:fps=25";

const FIXTURES: &[HbdFixture] = &[
    // XAVC Intra-like: High 4:2:2 Intra, 10-bit, CABAC, 8x8 transform, several slices
    HbdFixture {
        name: "hbd_422p10_intra_cabac",
        width: 352,
        height: 288,
        frames: 6,
        pix_fmt: "yuv422p10le",
        args: &["-profile:v", "high422", "-x264-params", "keyint=1:slices=4"],
        motion: "",
    },
    HbdFixture {
        name: "hbd_422p10_intra_cavlc",
        width: 352,
        height: 288,
        frames: 6,
        pix_fmt: "yuv422p10le",
        args: &["-profile:v", "high422", "-x264-params", "keyint=1:cabac=0"],
        motion: "",
    },
    HbdFixture {
        name: "hbd_422p10_intra_qp_low",
        width: 176,
        height: 144,
        frames: 4,
        pix_fmt: "yuv422p10le",
        args: &["-profile:v", "high422", "-qp", "1", "-x264-params", "keyint=1"],
        motion: "",
    },
    HbdFixture {
        name: "hbd_422p10_intra_qp_high",
        width: 176,
        height: 144,
        frames: 4,
        pix_fmt: "yuv422p10le",
        args: &["-profile:v", "high422", "-qp", "60", "-x264-params", "keyint=1:chroma-qp-offset=4"],
        motion: "",
    },
    HbdFixture {
        name: "hbd_422p10_intra_no8x8",
        width: 176,
        height: 144,
        frames: 4,
        pix_fmt: "yuv422p10le",
        args: &["-profile:v", "high422", "-x264-params", "keyint=1:8x8dct=0:cqm=jvt"],
        motion: "",
    },
    HbdFixture {
        name: "hbd_422p8_intra",
        width: 352,
        height: 288,
        frames: 4,
        pix_fmt: "yuv422p",
        args: &["-profile:v", "high422", "-x264-params", "keyint=1"],
        motion: "",
    },
    HbdFixture {
        name: "hbd_422p8_intra_cavlc",
        width: 176,
        height: 144,
        frames: 4,
        pix_fmt: "yuv422p",
        args: &["-profile:v", "high422", "-x264-params", "keyint=1:cabac=0"],
        motion: "",
    },
    // High 10: inter prediction with B-frames, weighted prediction
    HbdFixture {
        name: "hbd_420p10_inter",
        width: 352,
        height: 288,
        frames: 20,
        pix_fmt: "yuv420p10le",
        args: &["-profile:v", "high10", "-x264-params", "bframes=3:weightp=2:ref=3"],
        motion: "",
    },
    HbdFixture {
        name: "hbd_420p10_inter_cavlc",
        width: 176,
        height: 144,
        frames: 12,
        pix_fmt: "yuv420p10le",
        args: &["-profile:v", "high10", "-x264-params", "cabac=0:bframes=2"],
        motion: "",
    },
    // High 4:2:2 long-GOP (Sony XAVC S 4:2:2 10-bit and the like): P and B pictures with 4:2:2 chroma
    // motion compensation, weighted prediction and inter deblocking of the 4:2:2 chroma edges
    HbdFixture {
        name: "hbd_422p10_inter",
        width: 352,
        height: 288,
        frames: 20,
        pix_fmt: "yuv422p10le",
        args: &["-profile:v", "high422", "-x264-params", "bframes=3:weightp=2:ref=3"],
        motion: "",
    },
    HbdFixture {
        name: "hbd_422p10_inter_cavlc",
        width: 176,
        height: 144,
        frames: 12,
        pix_fmt: "yuv422p10le",
        args: &["-profile:v", "high422", "-x264-params", "cabac=0:bframes=2"],
        motion: "",
    },
    HbdFixture {
        name: "hbd_422p8_inter",
        width: 352,
        height: 288,
        frames: 16,
        pix_fmt: "yuv422p",
        args: &["-profile:v", "high422", "-x264-params", "bframes=2:ref=2"],
        motion: ZOOM_PAN,
    },
    HbdFixture {
        name: "hbd_422p10_inter_motion",
        width: 352,
        height: 288,
        frames: 24,
        pix_fmt: "yuv422p10le",
        args: &["-profile:v", "high422", "-x264-params", "bframes=3:ref=3:subme=9:me=umh:merange=32:weightp=2"],
        motion: ZOOM_PAN,
    },
    HbdFixture {
        name: "hbd_422p10_inter_temporal_direct",
        width: 176,
        height: 144,
        frames: 16,
        pix_fmt: "yuv422p10le",
        args: &["-profile:v", "high422", "-x264-params", "bframes=3:direct=temporal:weightb=1:b-pyramid=normal:ref=4"],
        motion: ZOOM_PAN,
    },
    HbdFixture {
        name: "hbd_422p10_inter_no8x8_cqm",
        width: 176,
        height: 144,
        frames: 12,
        pix_fmt: "yuv422p10le",
        args: &["-profile:v", "high422", "-x264-params", "bframes=2:8x8dct=0:cqm=jvt"],
        motion: ZOOM_PAN,
    },
    HbdFixture {
        name: "hbd_422p10_inter_qp_low",
        width: 176,
        height: 144,
        frames: 10,
        pix_fmt: "yuv422p10le",
        args: &["-profile:v", "high422", "-qp", "4", "-x264-params", "bframes=2:chroma-qp-offset=-6"],
        motion: ZOOM_PAN,
    },
    HbdFixture {
        name: "hbd_422p10_inter_qp_high",
        width: 176,
        height: 144,
        frames: 10,
        pix_fmt: "yuv422p10le",
        args: &["-profile:v", "high422", "-qp", "56", "-x264-params", "bframes=2:chroma-qp-offset=6"],
        motion: ZOOM_PAN,
    },
    // XAVC S-like structure: 24-picture GOP, two B-pictures, eight slices per picture
    HbdFixture {
        name: "hbd_422p10_inter_xavcs",
        width: 352,
        height: 288,
        frames: 30,
        pix_fmt: "yuv422p10le",
        args: &["-profile:v", "high422", "-x264-params", "keyint=24:bframes=2:slices=8"],
        motion: ZOOM_PAN,
    },
];

/// The fixture called `name`.
fn fixture(name: &str) -> &'static HbdFixture {
    FIXTURES.iter().find(|f| f.name == name).unwrap_or_else(|| panic!("unknown fixture {name}"))
}

/// Generate (if needed) the stream and its ffmpeg reference decode (16-bit LE above 8 bits).
fn ensure(f: &HbdFixture) -> Option<(PathBuf, PathBuf)> {
    let ff = filmcraft_testkit::ffmpeg_or_skip(f.name)?;
    let dir = common::fixtures_dir();
    let h264 = dir.join(format!("{}.h264", f.name));
    let yuv = dir.join(format!("{}.yuv", f.name));
    if !h264.exists() {
        let tmp = filmcraft_testkit::temp_path(&h264);
        let motion = if f.motion.is_empty() { String::new() } else { format!("{}:s={}x{},", f.motion, f.width, f.height) };
        let vf = format!("testsrc2=size={}x{}:rate=25,{motion}format={},noise=alls=10:allf=t+u", f.width, f.height, f.pix_fmt);
        let ok = Command::new(&ff)
            .args(["-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", &vf, "-frames:v", &f.frames.to_string(), "-c:v", "libx264"])
            .args(["-pix_fmt", f.pix_fmt])
            .args(f.args)
            .args(["-f", "h264"])
            .arg(&tmp)
            .status()
            .is_ok_and(|s| s.success());
        assert!(ok, "fixture generation failed for {}", f.name);
        std::fs::rename(&tmp, &h264).unwrap();
    }
    if !yuv.exists() {
        let tmp = filmcraft_testkit::temp_path(&yuv);
        let ok = Command::new(&ff)
            .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
            .arg(&h264)
            .args(["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", f.pix_fmt])
            .arg(&tmp)
            .status()
            .is_ok_and(|s| s.success());
        assert!(ok, "reference decode failed for {}", f.name);
        std::fs::rename(&tmp, &yuv).unwrap();
    }
    Some((h264, yuv))
}

/// The picture as the bytes ffmpeg writes for its pixel format.
fn raw(p: &Picture) -> Vec<u8> {
    match &p.wide {
        Some(w) => [&w.y, &w.u, &w.v].into_iter().flat_map(|pl| pl.iter().flat_map(|v| v.to_le_bytes())).collect(),
        None => [&p.y, &p.u, &p.v].into_iter().flat_map(|pl| pl.iter().copied()).collect(),
    }
}

/// Decode fixture `name` with `threads` threads and compare every frame byte-for-byte with
/// ffmpeg's raw output.
fn check(name: &str, threads: usize) {
    let f = fixture(name);
    let Some((h264, yuv)) = ensure(f) else { return };
    let data = std::fs::read(h264).unwrap();
    let want = std::fs::read(yuv).unwrap();
    let mut dec = Decoder::with_threads(threads);
    let mut pics = Vec::new();
    for au in common::split_access_units(&data) {
        pics.extend(dec.decode(au, 0).unwrap_or_else(|e| panic!("{name}: {e}")));
    }
    pics.extend(dec.flush());
    assert!(dec.take_error().is_none(), "{name}: decode error");
    assert_eq!(pics.len(), f.frames as usize, "{name}: frame count");
    let wide = f.pix_fmt.ends_with("10le");
    let c422 = f.pix_fmt.starts_with("yuv422");
    let frame_bytes = (f.width * f.height) as usize * if c422 { 2 } else { 3 } / if c422 { 1 } else { 2 } * if wide { 2 } else { 1 };
    for (i, p) in pics.iter().enumerate() {
        assert_eq!((p.width, p.height), (f.width, f.height));
        assert_eq!(p.bit_depth, if wide { 10 } else { 8 }, "{name}");
        assert_eq!(p.chroma_format_idc, if c422 { 2 } else { 1 }, "{name}");
        assert_eq!(p.wide.is_some(), wide, "{name}");
        let got = raw(p);
        let exp = &want[i * frame_bytes..(i + 1) * frame_bytes];
        if got != exp {
            let first = got.iter().zip(exp).position(|(a, b)| a != b).unwrap_or(0);
            panic!("{name} ({threads} threads): frame {i} differs from ffmpeg (first byte {first} of {frame_bytes})");
        }
    }
}

macro_rules! hbd_tests {
    ($($name:ident),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                check(stringify!($name), 1);
                check(stringify!($name), 4);
            }
        )*
    };
}

hbd_tests!(
    hbd_422p10_intra_cabac,
    hbd_422p10_intra_cavlc,
    hbd_422p10_intra_qp_low,
    hbd_422p10_intra_qp_high,
    hbd_422p10_intra_no8x8,
    hbd_422p8_intra,
    hbd_422p8_intra_cavlc,
    hbd_420p10_inter,
    hbd_420p10_inter_cavlc,
    hbd_422p10_inter,
    hbd_422p10_inter_cavlc,
    hbd_422p8_inter,
    hbd_422p10_inter_motion,
    hbd_422p10_inter_temporal_direct,
    hbd_422p10_inter_no8x8_cqm,
    hbd_422p10_inter_qp_low,
    hbd_422p10_inter_qp_high,
    hbd_422p10_inter_xavcs,
);
