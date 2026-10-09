//! Minimal D3D11 device for the OpenXR graphics binding, and CPU readback of the runtime's
//! environment depth texture array.

use serde_json::{Value, json};
use std::ffi::c_void;
use windows::{
    Win32::{
        Foundation::{HMODULE, LUID},
        Graphics::{
            Direct3D::{
                D3D_DRIVER_TYPE_UNKNOWN, D3D_FEATURE_LEVEL, D3D_FEATURE_LEVEL_11_0,
                D3D_FEATURE_LEVEL_11_1,
            },
            Direct3D11::{
                D3D11_CPU_ACCESS_READ, D3D11_CREATE_DEVICE_FLAG, D3D11_MAP_READ,
                D3D11_MAPPED_SUBRESOURCE, D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC,
                D3D11_USAGE_STAGING, D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext,
                ID3D11Texture2D,
            },
            Dxgi::{
                Common::{
                    DXGI_FORMAT, DXGI_FORMAT_D16_UNORM, DXGI_FORMAT_R16_TYPELESS,
                    DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R16_UNORM,
                },
                CreateDXGIFactory1, IDXGIAdapter1, IDXGIFactory1,
            },
        },
    },
    core::Interface,
};

pub struct D3d {
    pub device: ID3D11Device,
    context: ID3D11DeviceContext,
    pub adapter_name: String,
    staging: Option<(ID3D11Texture2D, D3D11_TEXTURE2D_DESC)>,
}

impl D3d {
    /// Create a device on the adapter the runtime asked for (xrGetD3D11GraphicsRequirementsKHR).
    pub fn create(luid_low: u32, luid_high: i32, min_feature_level: i32) -> Result<Self, String> {
        unsafe {
            let factory: IDXGIFactory1 =
                CreateDXGIFactory1().map_err(|e| format!("CreateDXGIFactory1: {e}"))?;
            let mut chosen: Option<(IDXGIAdapter1, String)> = None;
            for i in 0.. {
                let Ok(adapter) = factory.EnumAdapters1(i) else { break };
                let desc = adapter.GetDesc1().map_err(|e| format!("GetDesc1: {e}"))?;
                let LUID { LowPart, HighPart } = desc.AdapterLuid;
                if LowPart == luid_low && HighPart == luid_high {
                    let len = desc.Description.iter().position(|&c| c == 0).unwrap_or(128);
                    chosen = Some((adapter, String::from_utf16_lossy(&desc.Description[..len])));
                    break;
                }
            }
            let (adapter, adapter_name) =
                chosen.ok_or("no DXGI adapter matches the runtime's adapter LUID")?;

            let levels: Vec<D3D_FEATURE_LEVEL> = [D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0]
                .into_iter()
                .filter(|l| l.0 >= min_feature_level)
                .collect();
            let mut device = None;
            let mut context = None;
            D3D11CreateDevice(
                &adapter,
                D3D_DRIVER_TYPE_UNKNOWN,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_FLAG(0),
                Some(&levels),
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
            .map_err(|e| format!("D3D11CreateDevice: {e}"))?;
            Ok(Self {
                device: device.ok_or("D3D11CreateDevice returned no device")?,
                context: context.ok_or("D3D11CreateDevice returned no context")?,
                adapter_name,
                staging: None,
            })
        }
    }

    pub fn device_ptr(&self) -> *mut c_void {
        self.device.as_raw()
    }

    /// Copy both array slices of a runtime depth texture to the CPU, stacked top (slice 0) to
    /// bottom (slice 1), as little-endian u16 rows (D3D11 row 0 is the top row, no flip needed).
    /// `width` x `height` is the per-view size the runtime reported for the swapchain.
    pub fn read_depth_slices(&mut self, texture: *mut c_void, width: u32, height: u32) -> Result<Vec<u8>, String> {
        unsafe {
            let src = ID3D11Texture2D::from_raw_borrowed(&texture).ok_or("null depth texture")?;
            // CopyResource across devices would remove the device instead of failing cleanly.
            let owner = src.GetDevice().map_err(|e| format!("texture GetDevice: {e}"))?;
            if owner.as_raw() != self.device.as_raw() {
                return Err("depth texture belongs to a different D3D11 device".into());
            }
            let mut desc = D3D11_TEXTURE2D_DESC::default();
            src.GetDesc(&mut desc);
            if (desc.Width, desc.Height) != (width, height) {
                return Err(format!(
                    "texture is {}x{} but the swapchain state says {width}x{height}",
                    desc.Width, desc.Height
                ));
            }
            if !is_16bit(desc.Format) {
                return Err(format!("unsupported depth texture format {}", desc.Format.0));
            }
            if desc.ArraySize < 2 {
                return Err(format!("depth texture has {} array slice(s), expected 2", desc.ArraySize));
            }
            if desc.SampleDesc.Count > 1 {
                return Err(format!("depth texture is multisampled ({}x)", desc.SampleDesc.Count));
            }

            let staging = self.staging_for(&desc)?;
            self.context.CopyResource(&staging, src);

            let row_bytes = desc.Width as usize * 2;
            let mut out = vec![0u8; row_bytes * desc.Height as usize * 2];
            for slice in 0..2u32 {
                let subresource = slice * desc.MipLevels;
                let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
                self.context
                    .Map(&staging, subresource, D3D11_MAP_READ, 0, Some(&mut mapped))
                    .map_err(|e| format!("Map slice {slice}: {e}"))?;
                let base = mapped.pData as *const u8;
                for row in 0..desc.Height as usize {
                    let src_row =
                        std::slice::from_raw_parts(base.add(row * mapped.RowPitch as usize), row_bytes);
                    let dst = (slice as usize * desc.Height as usize + row) * row_bytes;
                    out[dst..dst + row_bytes].copy_from_slice(src_row);
                }
                self.context.Unmap(&staging, subresource);
            }
            Ok(out)
        }
    }

    fn staging_for(&mut self, src: &D3D11_TEXTURE2D_DESC) -> Result<ID3D11Texture2D, String> {
        if let Some((tex, desc)) = &self.staging
            && desc.Width == src.Width
            && desc.Height == src.Height
            && desc.ArraySize == src.ArraySize
            && desc.MipLevels == src.MipLevels
        {
            return Ok(tex.clone());
        }
        let mut desc = *src;
        desc.Usage = D3D11_USAGE_STAGING;
        desc.BindFlags = 0;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ.0 as u32;
        desc.MiscFlags = 0;
        let mut first_err = String::new();
        // Depth formats are not always accepted for staging; R16_TYPELESS is copy-compatible.
        for format in [desc.Format, DXGI_FORMAT_R16_TYPELESS] {
            desc.Format = format;
            let mut tex = None;
            match unsafe { self.device.CreateTexture2D(&desc, None, Some(&mut tex)) } {
                Ok(()) => {
                    let tex = tex.ok_or("CreateTexture2D returned no texture")?;
                    self.staging = Some((tex.clone(), desc));
                    return Ok(tex);
                }
                Err(e) if first_err.is_empty() => first_err = format!("format {}: {e}", format.0),
                Err(_) => {}
            }
        }
        Err(format!("cannot create staging texture ({first_err})"))
    }
}

fn is_16bit(f: DXGI_FORMAT) -> bool {
    [DXGI_FORMAT_D16_UNORM, DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R16_UINT]
        .contains(&f)
}

/// Describe a runtime texture for the report.
pub fn texture_desc_json(texture: *mut c_void) -> Value {
    unsafe {
        let Some(tex) = ID3D11Texture2D::from_raw_borrowed(&texture) else {
            return json!(null);
        };
        let mut d = D3D11_TEXTURE2D_DESC::default();
        tex.GetDesc(&mut d);
        json!({
            "width": d.Width, "height": d.Height, "mip_levels": d.MipLevels,
            "array_size": d.ArraySize, "dxgi_format": d.Format.0,
            "sample_count": d.SampleDesc.Count, "usage": d.Usage.0, "bind_flags": d.BindFlags,
        })
    }
}
