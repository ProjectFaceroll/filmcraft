//! H.264 intra-profile streams (High 4:2:2 Intra and the other intra profiles: XAVC Intra,
//! AVC-Intra) decode a picture at a time: the decoder reports `intra_only`, every access unit
//! comes out of its own `decode` call, in any order, bit-exact with ffmpeg, from an `avcC`
//! record (MP4) and from Annex B samples (MXF). A long-GOP stream of the same profile family
//! keeps the frame-threaded decoder.

use crate::common;

use common::*;
use filmcraft_codecs::video::{H264Decoder, VideoDecoder};
use filmcraft_media::FrameRequest;

const SIZE: (usize, usize) = (320, 240);

/// x264 signals the intra profile (`constraint_set3_flag`) for a keyframe-only stream.
fn args(gop: &'static str, bframes: &'static str) -> Vec<&'static str> {
    vec![
        "-f",
        "lavfi",
        "-i",
        "testsrc2=s=320x240:r=25:d=1,noise=alls=10:allf=t",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-g",
        gop,
        "-bf",
        bframes,
        "-pix_fmt",
        "yuv422p10le",
    ]
}

/// ffmpeg's frames of `file` as planes (4:2:2, 10-bit).
fn reference(ff: &std::path::Path, file: &std::path::Path) -> Vec<Vec<Vec<u16>>> {
    let (w, h) = SIZE;
    let (cw, ch) = (w / 2, h);
    let raw = ffmpeg_frames(ff, file, "yuv422p10le");
    let fsize = (w * h + 2 * cw * ch) * 2;
    (0..raw.len() / fsize).map(|i| raw_planes(&raw[i * fsize..(i + 1) * fsize], w, h, cw, ch, 2)).collect()
}

/// Every requested frame of `src`, out of order and on a cold cache, equals ffmpeg's.
fn random_access_is_exact(src: &filmcraft_media::SharedSource, frames: &[Vec<Vec<u16>>], order: &[usize]) {
    let rate = src.info().frame_rate();
    for &i in order {
        let f = src.video_frame(FrameRequest::full(rate.tick_of(i as i64))).expect("frame");
        assert_eq!(max_diff(&planes(&f), &frames[i]), 0, "frame {i}");
    }
}

#[test]
fn intra_profile_mp4_decodes_each_access_unit_alone() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let Some(file) = fixture(&ff, "intra_profile_x264_422p10.mp4", &args("1", "0")) else { return };
    let frames = reference(&ff, &file);
    let n = frames.len();
    assert!(n >= 20, "{n} frames");
    let bytes = bytes(&file);
    let mp4 = filmcraft_isobmff::open(&bytes[..]).expect("parse");
    let track = mp4.tracks.iter().position(|t| t.kind == filmcraft_isobmff::TrackKind::Video).expect("video track");
    let filmcraft_isobmff::CodecConfig::Avc(avcc) = &mp4.tracks[track].entries[0].codec else { panic!("not AVC") };
    let mut dec = H264Decoder::new(avcc.to_bytes()).expect("decoder");
    assert!(dec.intra_only(), "High 4:2:2 Intra is an intra profile");
    assert!(!dec.thread_limited(), "one thread is all an intra decoder gets; it is not a cut-down one");
    // Out of order, each access unit on its own: the picture comes out of the call that fed it.
    for i in [n - 1, 7, 0, 3, n / 2, 1] {
        let sample = mp4.read_sample(&bytes[..], track, i).expect("sample");
        let out = dec.decode(&sample, i as i64).expect("decode");
        assert_eq!(out.len(), 1, "sample {i}: one picture per call");
        assert_eq!(out[0].pts, i as i64);
        assert_eq!(max_diff(&planes(&out[0].frame), &frames[i]), 0, "sample {i}");
    }
    // Through the media stack (the GOP cache's intra path) too.
    let src = filmcraft_codecs::open_bytes("intra_profile_x264_422p10.mp4", bytes.clone()).expect("open");
    random_access_is_exact(&src, &frames, &[n - 2, 4, 11, 0, n / 2, 4]);
}

#[test]
fn intra_profile_mxf_is_told_by_the_first_samples_sps() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let Some(file) = fixture(&ff, "intra_profile_x264_422p10.mxf", &args("1", "0")) else { return };
    let frames = reference(&ff, &file);
    let n = frames.len();
    assert!(n >= 20, "{n} frames");
    let bytes = bytes(&file);
    let mxf = filmcraft_mxf::open(&bytes[..]).expect("parse");
    let track = mxf.tracks.iter().position(|t| t.kind == filmcraft_mxf::TrackKind::Picture).expect("picture track");
    let first = mxf.read_sample(&bytes[..], track, 0).expect("sample 0");
    let mut dec = H264Decoder::annexb_for(&first);
    assert!(dec.intra_only(), "the in-band SPS declares High 4:2:2 Intra");
    assert!(!H264Decoder::annexb().intra_only(), "without an SPS the stream is taken for long-GOP");
    for i in [n - 1, 5, 0, 9] {
        let sample = mxf.read_sample(&bytes[..], track, i).expect("sample");
        let out = dec.decode(&sample, i as i64).expect("decode");
        assert_eq!(out.len(), 1, "sample {i}: one picture per call");
        assert_eq!(max_diff(&planes(&out[0].frame), &frames[i]), 0, "sample {i}");
    }
    let src = filmcraft_codecs::open_bytes("intra_profile_x264_422p10.mxf", bytes.clone()).expect("open");
    random_access_is_exact(&src, &frames, &[n - 2, 4, 11, 0, n / 2, 4]);
}

#[test]
fn long_gop_high_422_keeps_the_frame_threaded_decoder() {
    let ff = filmcraft_testkit::require_ffmpeg!();
    let Some(file) = fixture(&ff, "long_gop_x264_422p10.mp4", &args("12", "2")) else { return };
    let bytes = bytes(&file);
    let mp4 = filmcraft_isobmff::open(&bytes[..]).expect("parse");
    let track = mp4.tracks.iter().position(|t| t.kind == filmcraft_isobmff::TrackKind::Video).expect("video track");
    let filmcraft_isobmff::CodecConfig::Avc(avcc) = &mp4.tracks[track].entries[0].codec else { panic!("not AVC") };
    let dec = H264Decoder::new(avcc.to_bytes()).expect("decoder");
    assert!(!dec.intra_only(), "High 4:2:2 without constraint_set3_flag is not an intra profile");
    let frames = reference(&ff, &file);
    let src = filmcraft_codecs::open_bytes("long_gop_x264_422p10.mp4", bytes.clone()).expect("open");
    random_access_is_exact(&src, &frames, &[frames.len() - 1, 6, 0, 13]);
}
