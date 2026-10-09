//! The .rktap file and the MSG_DEPTH_FRAME_V2 header written by link_spike must be byte
//! compatible with tools/realkk/rktap.py and depth_listener.parse_depth_v2 (offsets mirror
//! alvr/server_core/src/xr_data_relay.rs).

use link_spike::tap::{
    DEPTH_FRAME_V2_HEADER_SIZE, DepthFrame, FORMAT_RAW_D16, MSG_DEPTH_FRAME_V2, TapWriter,
    encode_depth_frame_v2, fov_to_pinhole_intrinsics,
};

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}
fn f32_at(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

fn sample_frame() -> DepthFrame {
    let a = std::f32::consts::FRAC_PI_4;
    DepthFrame {
        client_timestamp_ns: 123_456_789,
        view_poses: [
            [0.1, 0.2, 0.3, 0.9, 1.0, 2.0, 3.0],
            [0.4, 0.5, 0.6, 0.7, -1.0, -2.0, -3.0],
        ],
        width: 320,
        height: 640,
        near_z: 0.1,
        far_z: f32::INFINITY,
        format: FORMAT_RAW_D16,
        fov_angles: [[-a, a, a, -a], [-a, 0.5, a, -a]],
        server_timestamp_unix_ns: 1_700_000_000_000_000_000,
        clock_offset_ns: -42,
        server_receive_unix_ns: 1_700_000_000_000_000_123,
    }
}

#[test]
fn header_offsets_match_relay_layout() {
    let frame = sample_frame();
    let pixels = [7u8, 8, 9, 10];
    let buf = encode_depth_frame_v2(&frame, &pixels);

    assert_eq!(MSG_DEPTH_FRAME_V2, 3);
    assert_eq!(DEPTH_FRAME_V2_HEADER_SIZE, 176);
    assert_eq!(buf.len(), 176 + pixels.len());
    assert_eq!(u32_at(&buf, 0), 176);
    assert_eq!(u64_at(&buf, 4), 123_456_789);
    for (view, off) in [(0usize, 12usize), (1, 40)] {
        for i in 0..7 {
            assert_eq!(f32_at(&buf, off + 4 * i), frame.view_poses[view][i]);
        }
    }
    assert_eq!(u32_at(&buf, 68), 320);
    assert_eq!(u32_at(&buf, 72), 640);
    assert_eq!(f32_at(&buf, 76), 0.1);
    assert!(f32_at(&buf, 80).is_infinite());
    assert_eq!(u32_at(&buf, 84), FORMAT_RAW_D16);
    for (view, off) in [(0usize, 88usize), (1, 104)] {
        for i in 0..4 {
            assert_eq!(f32_at(&buf, off + 4 * i), frame.fov_angles[view][i]);
        }
    }
    for (view, off) in [(0usize, 120usize), (1, 136)] {
        let expected = fov_to_pinhole_intrinsics(frame.fov_angles[view], 320, 320);
        for i in 0..4 {
            assert_eq!(f32_at(&buf, off + 4 * i), expected[i]);
        }
    }
    assert_eq!(u64_at(&buf, 152), 1_700_000_000_000_000_000);
    assert_eq!(i64::from_le_bytes(buf[160..168].try_into().unwrap()), -42);
    assert_eq!(u64_at(&buf, 168), 1_700_000_000_000_000_123);
    assert_eq!(&buf[176..], &pixels);
}

#[test]
fn intrinsics_use_per_view_height() {
    // Same expectations as alvr_packets::tests::view_intrinsics_use_per_view_height.
    let a = std::f32::consts::FRAC_PI_4;
    let [fx, fy, cx, cy] = fov_to_pinhole_intrinsics([-a, a, a, -a], 320, 320);
    let approx = |x: f32, y: f32| (x - y).abs() < 1e-3;
    assert!(approx(fx, 160.0) && approx(fy, 160.0));
    assert!(approx(cx, 160.0) && approx(cy, 160.0));
    let [fx1, _, cx1, _] = fov_to_pinhole_intrinsics([-a, 0.5, a, -a], 320, 320);
    assert!(approx(fx1, 320.0 / (0.5_f32.tan() + 1.0)));
    assert!(approx(cx1, fx1));
}

#[test]
fn tap_file_layout_matches_rktap_py() {
    let dir = std::env::temp_dir().join(format!("link_spike_tap_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("t.rktap");
    let meta = serde_json::json!({"source": "test"});
    {
        let mut w = TapWriter::create(&path, &meta).unwrap();
        w.write(11, MSG_DEPTH_FRAME_V2, b"abc").unwrap();
        w.write(22, 0xFFFF_0001, b"").unwrap();
    }
    let b = std::fs::read(&path).unwrap();
    std::fs::remove_dir_all(&dir).ok();

    assert_eq!(&b[..5], b"RKTAP");
    assert_eq!(u32_at(&b, 5), 1);
    let meta_len = u32_at(&b, 9) as usize;
    let meta_back: serde_json::Value = serde_json::from_slice(&b[13..13 + meta_len]).unwrap();
    assert_eq!(meta_back, meta);
    let mut o = 13 + meta_len;
    assert_eq!(u64_at(&b, o), 11);
    assert_eq!(u32_at(&b, o + 8), MSG_DEPTH_FRAME_V2);
    assert_eq!(u32_at(&b, o + 12), 3);
    assert_eq!(&b[o + 16..o + 19], b"abc");
    o += 19;
    assert_eq!(u64_at(&b, o), 22);
    assert_eq!(u32_at(&b, o + 8), 0xFFFF_0001);
    assert_eq!(u32_at(&b, o + 12), 0);
    assert_eq!(b.len(), o + 16);
}
