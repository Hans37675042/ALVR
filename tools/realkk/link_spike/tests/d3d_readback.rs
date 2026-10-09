//! D3D11 readback of a two-slice 16-bit depth texture array, as the runtime's environment depth
//! swapchain images are expected to be. Needs a D3D11 capable adapter (no headset, no runtime).
#![cfg(windows)]

use link_spike::d3d::D3d;
use windows::Win32::Graphics::{
    Direct3D11::{
        D3D11_BIND_DEPTH_STENCIL, D3D11_BIND_SHADER_RESOURCE, D3D11_SUBRESOURCE_DATA,
        D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT,
    },
    Dxgi::{
        Common::{DXGI_FORMAT, DXGI_FORMAT_D16_UNORM, DXGI_FORMAT_R16_UNORM, DXGI_SAMPLE_DESC},
        CreateDXGIFactory1, IDXGIFactory1,
    },
};
use windows::core::Interface;

const W: u32 = 8;
const H: u32 = 4;

fn first_adapter_luid() -> (u32, i32) {
    unsafe {
        let f: IDXGIFactory1 = CreateDXGIFactory1().unwrap();
        let luid = f.EnumAdapters1(0).unwrap().GetDesc1().unwrap().AdapterLuid;
        (luid.LowPart, luid.HighPart)
    }
}

fn slice_values(slice: u32) -> Vec<u16> {
    (0..W * H).map(|i| (slice * 1000 + i * 7) as u16).collect()
}

fn make_texture(d3d: &D3d, format: DXGI_FORMAT, bind: i32) -> Option<windows::Win32::Graphics::Direct3D11::ID3D11Texture2D> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: W,
        Height: H,
        MipLevels: 1,
        ArraySize: 2,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: bind as u32,
        CPUAccessFlags: 0,
        MiscFlags: 0,
    };
    let slices: Vec<Vec<u16>> = (0..2).map(slice_values).collect();
    let init: Vec<D3D11_SUBRESOURCE_DATA> = slices
        .iter()
        .map(|s| D3D11_SUBRESOURCE_DATA {
            pSysMem: s.as_ptr() as *const _,
            SysMemPitch: W * 2,
            SysMemSlicePitch: W * H * 2,
        })
        .collect();
    let mut tex = None;
    unsafe { d3d.device.CreateTexture2D(&desc, Some(init.as_ptr()), Some(&mut tex)) }.ok()?;
    tex
}

#[test]
fn reads_both_slices_stacked_top_down() {
    let (low, high) = first_adapter_luid();
    let mut d3d = D3d::create(low, high, 0xb000).expect("D3D11 device");
    // Depth-stencil textures may refuse initial data on some drivers; R16_UNORM still exercises
    // the staging copy and the per-slice Map.
    let tex = match make_texture(&d3d, DXGI_FORMAT_D16_UNORM, D3D11_BIND_DEPTH_STENCIL.0) {
        Some(t) => {
            eprintln!("source texture: D16_UNORM depth-stencil");
            t
        }
        None => {
            eprintln!("source texture: R16_UNORM fallback (D16 refused initial data)");
            make_texture(&d3d, DXGI_FORMAT_R16_UNORM, D3D11_BIND_SHADER_RESOURCE.0)
                .expect("16-bit texture array")
        }
    };

    let bytes = d3d.read_depth_slices(tex.as_raw()).expect("readback");
    let got: Vec<u16> = bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    let mut expected = slice_values(0);
    expected.extend(slice_values(1));
    assert_eq!(got, expected);

    // Second read reuses the cached staging texture.
    assert_eq!(d3d.read_depth_slices(tex.as_raw()).unwrap(), bytes);
}
