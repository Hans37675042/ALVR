// Hardware video encoder wrapping Android AMediaCodec NDK API.
// Used for H.264 encoding of camera and depth feeds on Quest 3.
// Only compiled on Android targets.

#![cfg(target_os = "android")]

use std::ffi::{c_char, c_void};
use std::ptr;

// ── AMediaCodec NDK FFI bindings ──────────────────────────────────────────

#[repr(C)]
struct AMediaCodec {
    _private: [u8; 0],
}

#[repr(C)]
struct AMediaFormat {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Default)]
struct AMediaCodecBufferInfo {
    offset: i32,
    size: i32,
    presentation_time_us: i64,
    flags: u32,
}

const AMEDIACODEC_CONFIGURE_FLAG_ENCODE: u32 = 1;
const AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM: u32 = 4;
const AMEDIACODEC_INFO_TRY_AGAIN_LATER: isize = -1;
const AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED: isize = -2;
const AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED: isize = -3;

const AMEDIA_OK: i32 = 0;

// Timeout in microseconds for dequeue operations
const DEQUEUE_TIMEOUT_US: i64 = 10_000; // 10ms

unsafe extern "C" {
    fn AMediaCodec_createEncoderByType(mime_type: *const c_char) -> *mut AMediaCodec;
    fn AMediaCodec_configure(
        codec: *mut AMediaCodec,
        format: *const AMediaFormat,
        surface: *mut c_void, // ANativeWindow*, null for buffer mode
        crypto: *mut c_void,  // null
        flags: u32,
    ) -> i32;
    fn AMediaCodec_start(codec: *mut AMediaCodec) -> i32;
    fn AMediaCodec_stop(codec: *mut AMediaCodec) -> i32;
    fn AMediaCodec_delete(codec: *mut AMediaCodec) -> i32;
    fn AMediaCodec_dequeueInputBuffer(codec: *mut AMediaCodec, timeout_us: i64) -> isize;
    fn AMediaCodec_getInputBuffer(
        codec: *mut AMediaCodec,
        idx: usize,
        out_size: *mut usize,
    ) -> *mut u8;
    fn AMediaCodec_queueInputBuffer(
        codec: *mut AMediaCodec,
        idx: usize,
        offset: i32,
        size: usize,
        time_us: u64,
        flags: u32,
    ) -> i32;
    fn AMediaCodec_dequeueOutputBuffer(
        codec: *mut AMediaCodec,
        info: *mut AMediaCodecBufferInfo,
        timeout_us: i64,
    ) -> isize;
    fn AMediaCodec_getOutputBuffer(
        codec: *mut AMediaCodec,
        idx: usize,
        out_size: *mut usize,
    ) -> *const u8;
    fn AMediaCodec_releaseOutputBuffer(
        codec: *mut AMediaCodec,
        idx: usize,
        render: bool,
    ) -> i32;

    fn AMediaFormat_new() -> *mut AMediaFormat;
    fn AMediaFormat_delete(format: *mut AMediaFormat) -> i32;
    fn AMediaFormat_setString(format: *mut AMediaFormat, name: *const c_char, value: *const c_char);
    fn AMediaFormat_setInt32(format: *mut AMediaFormat, name: *const c_char, value: i32);
}

/// Hardware H.264 encoder using AMediaCodec.
pub struct HwEncoder {
    codec: *mut AMediaCodec,
    width: u32,
    height: u32,
    frame_idx: u64,
    output_buf: Vec<u8>,
    started: bool,
}

// AMediaCodec is thread-safe per Android docs
unsafe impl Send for HwEncoder {}

impl HwEncoder {
    /// Create and start a new H.264 hardware encoder.
    ///
    /// - `width`, `height`: frame dimensions (must be even)
    /// - `bitrate`: target bitrate in bits/sec (e.g. 2_000_000 for 2 Mbps)
    /// - `fps`: target frame rate
    /// - `idr_interval_sec`: seconds between IDR frames (e.g. 1)
    pub fn new(
        width: u32,
        height: u32,
        bitrate: i32,
        fps: i32,
        idr_interval_sec: i32,
    ) -> Option<Self> {
        unsafe {
            let mime = b"video/avc\0".as_ptr() as *const c_char;
            let codec = AMediaCodec_createEncoderByType(mime);
            if codec.is_null() {
                alvr_common::info!("[HwEncoder] Failed to create H.264 encoder");
                return None;
            }

            let format = AMediaFormat_new();
            if format.is_null() {
                AMediaCodec_delete(codec);
                alvr_common::info!("[HwEncoder] Failed to create AMediaFormat");
                return None;
            }

            // Set format parameters
            let key_mime = b"mime\0".as_ptr() as *const c_char;
            let key_width = b"width\0".as_ptr() as *const c_char;
            let key_height = b"height\0".as_ptr() as *const c_char;
            let key_bitrate = b"bitrate\0".as_ptr() as *const c_char;
            let key_frame_rate = b"frame-rate\0".as_ptr() as *const c_char;
            let key_color_format = b"color-format\0".as_ptr() as *const c_char;
            let key_i_frame_interval = b"i-frame-interval\0".as_ptr() as *const c_char;
            let key_bitrate_mode = b"bitrate-mode\0".as_ptr() as *const c_char;

            AMediaFormat_setString(format, key_mime, mime);
            AMediaFormat_setInt32(format, key_width, width as i32);
            AMediaFormat_setInt32(format, key_height, height as i32);
            AMediaFormat_setInt32(format, key_bitrate, bitrate);
            AMediaFormat_setInt32(format, key_frame_rate, fps);
            // COLOR_FormatYUV420SemiPlanar = 21 (NV12)
            AMediaFormat_setInt32(format, key_color_format, 21);
            AMediaFormat_setInt32(format, key_i_frame_interval, idr_interval_sec);
            // CBR = 2 (constant bitrate for predictable bandwidth)
            AMediaFormat_setInt32(format, key_bitrate_mode, 2);

            let status = AMediaCodec_configure(
                codec,
                format,
                ptr::null_mut(),
                ptr::null_mut(),
                AMEDIACODEC_CONFIGURE_FLAG_ENCODE,
            );

            AMediaFormat_delete(format);

            if status != AMEDIA_OK {
                alvr_common::info!("[HwEncoder] Configure failed: {status}");
                AMediaCodec_delete(codec);
                return None;
            }

            let status = AMediaCodec_start(codec);
            if status != AMEDIA_OK {
                alvr_common::info!("[HwEncoder] Start failed: {status}");
                AMediaCodec_delete(codec);
                return None;
            }

            alvr_common::info!(
                "[HwEncoder] Created {}x{} @ {} bps, {} fps, IDR every {}s",
                width, height, bitrate, fps, idr_interval_sec
            );

            Some(Self {
                codec,
                width,
                height,
                frame_idx: 0,
                output_buf: Vec::new(),
                started: true,
            })
        }
    }

    /// Feed one NV12 frame and retrieve encoded H.264 NAL units.
    ///
    /// `nv12_data` must be exactly `width * height * 3 / 2` bytes
    /// (Y plane: w*h, UV interleaved plane: w*h/2).
    ///
    /// Returns `Some(nal_data)` if encoded output is available, `None` if not yet ready.
    pub fn encode(&mut self, nv12_data: &[u8]) -> Option<&[u8]> {
        let expected_size = (self.width * self.height * 3 / 2) as usize;
        if nv12_data.len() != expected_size {
            alvr_common::error!(
                "[HwEncoder] NV12 data size mismatch: got {}, expected {}",
                nv12_data.len(),
                expected_size
            );
            return None;
        }

        unsafe {
            // Submit input frame
            let input_idx = AMediaCodec_dequeueInputBuffer(self.codec, DEQUEUE_TIMEOUT_US);
            if input_idx < 0 {
                // No input buffer available right now
                return None;
            }

            let mut buf_size: usize = 0;
            let buf_ptr = AMediaCodec_getInputBuffer(self.codec, input_idx as usize, &mut buf_size);
            if buf_ptr.is_null() || buf_size < nv12_data.len() {
                alvr_common::error!(
                    "[HwEncoder] Input buffer too small: {} < {}",
                    buf_size,
                    nv12_data.len()
                );
                return None;
            }

            ptr::copy_nonoverlapping(nv12_data.as_ptr(), buf_ptr, nv12_data.len());

            let pts_us = self.frame_idx * 1_000_000 / 30; // approximate PTS
            AMediaCodec_queueInputBuffer(
                self.codec,
                input_idx as usize,
                0,
                nv12_data.len(),
                pts_us,
                0,
            );
            self.frame_idx += 1;

            // Collect output
            self.output_buf.clear();
            loop {
                let mut info = AMediaCodecBufferInfo::default();
                let output_idx =
                    AMediaCodec_dequeueOutputBuffer(self.codec, &mut info, DEQUEUE_TIMEOUT_US);

                if output_idx == AMEDIACODEC_INFO_TRY_AGAIN_LATER {
                    break;
                } else if output_idx == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED
                    || output_idx == AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED
                {
                    continue;
                } else if output_idx >= 0 {
                    let mut out_size: usize = 0;
                    let out_ptr =
                        AMediaCodec_getOutputBuffer(self.codec, output_idx as usize, &mut out_size);

                    if !out_ptr.is_null() && info.size > 0 {
                        let data_start = out_ptr.add(info.offset as usize);
                        let data_len = info.size as usize;
                        self.output_buf
                            .extend_from_slice(std::slice::from_raw_parts(data_start, data_len));
                    }

                    AMediaCodec_releaseOutputBuffer(self.codec, output_idx as usize, false);

                    if info.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM != 0 {
                        break;
                    }
                } else {
                    break;
                }
            }

            if self.output_buf.is_empty() {
                None
            } else {
                Some(&self.output_buf)
            }
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}

impl Drop for HwEncoder {
    fn drop(&mut self) {
        if self.started {
            unsafe {
                AMediaCodec_stop(self.codec);
                AMediaCodec_delete(self.codec);
            }
            alvr_common::info!("[HwEncoder] Encoder destroyed");
        }
    }
}
