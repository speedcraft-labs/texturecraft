//! WGPU fast path for simple paint/fill stacks. Complex projections, masks, and effects stay on
//! the CPU compositor, which remains the fallback and pixel oracle.

use std::sync::Arc;

use texturecraft_engine::compose::{Composite, Inputs, PaintTarget};
use texturecraft_engine::core::{BlendMode, Channel, ChannelKind, LayerKind, Projection, Source};
use texturecraft_engine::geom::NO_TRI;
use texturecraft_engine::texel::{TILE, TILE_AREA, from_u16, from_u16_signed, to_u16, to_u16_signed};
use texturecraft_render::wgpu::{self, util::DeviceExt};

const MAX_BATCH_TILES: usize = 32;
const WORKGROUP: u32 = 64;

const SHADER: &str = r#"
struct ChannelInfo {
    comps: u32,
    signed: u32,
    layers: u32,
    _pad: u32,
    defaults: vec4<f32>,
};
struct LayerInfo {
    kind: u32,
    replace_mode: u32,
    opacity_bits: u32,
    data_offset: u32,
    values: vec4<u32>,
};
struct Params {
    tiles: u32,
    channels: u32,
    layers: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> paint_data: array<u32>;
@group(0) @binding(1) var<storage, read> layer_info: array<LayerInfo>;
@group(0) @binding(2) var<storage, read> channel_info: array<ChannelInfo>;
@group(0) @binding(3) var<storage, read> coverage: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

fn decode(v: u32, signed: u32) -> f32 {
    let x = f32(v & 65535u) / 65535.0;
    if signed != 0u { return x * 2.0 - 1.0; }
    return x;
}

@compute @workgroup_size(64)
fn composite(@builtin(global_invocation_id) id: vec3<u32>) {
    let count = params.tiles * params.channels * 4096u;
    if id.x >= count { return; }
    let texel = id.x % 4096u;
    let channel_index = (id.x / 4096u) % params.channels;
    let tile_index = id.x / (4096u * params.channels);
    let ci = channel_info[channel_index];
    var value = ci.defaults.xyz;
    if coverage[tile_index * 4096u + texel] != 0u {
        for (var layer = 0u; layer < params.layers; layer += 1u) {
            let li = layer_info[(tile_index * params.channels + channel_index) * params.layers + layer];
            if li.kind == 0u { continue; }
            let opacity = bitcast<f32>(li.opacity_bits);
            var source = bitcast<vec3<f32>>(li.values.xyz);
            var weight = opacity;
            if li.kind == 2u {
                let base = li.data_offset + texel * (ci.comps + 1u);
                let cov = decode(paint_data[base + ci.comps], 0u);
                if cov <= 0.0 { continue; }
                source = vec3<f32>(decode(paint_data[base], ci.signed), 0.0, 0.0);
                if ci.comps == 3u {
                    source.y = decode(paint_data[base + 1u], ci.signed);
                    source.z = decode(paint_data[base + 2u], ci.signed);
                }
                source /= cov;
                if li.replace_mode == 0u { weight *= cov; }
            }
            value += (source - value) * weight;
        }
    }
    let out_base = id.x * 3u;
    output[out_base] = bitcast<u32>(value.x);
    output[out_base + 1u] = bitcast<u32>(value.y);
    output[out_base + 2u] = bitcast<u32>(value.z);
}
"#;

/// A reusable compute pipeline tied to the app's existing wgpu device.
pub struct GpuComposite {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
}

impl GpuComposite {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> GpuComposite {
        let module = device
            .create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("texturecraft-composite"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("texturecraft-composite"),
            layout: None,
            module: &module,
            entry_point: Some("composite"),
            compilation_options: Default::default(),
            cache: None,
        });
        GpuComposite { device, queue, pipeline }
    }

    /// Composite a supported stack and return `true`; unsupported stacks or GPU failures are
    /// left to the CPU oracle. Every requested output tile is replaced on success.
    pub fn composite(&self, inp: &Inputs<'_>, out: &mut Composite, tiles: &[(u32, u32)]) -> bool {
        if tiles.is_empty() || !supported(inp) {
            return false;
        }
        let Ok(set) = inp.project.texture_set(inp.ts) else { return false };
        let channels = set.channel_list();
        if channels.is_empty() || out.channels.len() != channels.len() {
            return false;
        }

        for batch in tiles.chunks(MAX_BATCH_TILES) {
            if !self.composite_batch(inp, out, &channels, batch) {
                return false;
            }
        }
        true
    }

    fn composite_batch(&self, inp: &Inputs<'_>, out: &mut Composite, channels: &[Channel], tiles: &[(u32, u32)]) -> bool {
        let Some(layers) = inp.project.texture_set(inp.ts).ok().map(|s| s.stack.as_slice()) else { return false };
        let (channel_count, layer_count, tile_count) = (channels.len(), layers.len(), tiles.len());
        let mut paint_data = Vec::<u32>::new();
        let mut infos = vec![0u32; tile_count * channel_count * layer_count * 8];
        let mut channel_info = Vec::<u32>::with_capacity(channel_count * 8);
        for ch in channels {
            let default = ch.default_value();
            channel_info.extend_from_slice(&[
                ch.components() as u32,
                u32::from(ch.kind() == ChannelKind::Signed),
                layer_count as u32,
                0,
                default[0].to_bits(),
                default[1].to_bits(),
                default[2].to_bits(),
                0,
            ]);
        }

        let mut coverage = Vec::<u32>::with_capacity(tile_count * TILE_AREA);
        for (tile_index, &(tx, ty)) in tiles.iter().enumerate() {
            for local in 0..TILE_AREA {
                let x = tx * TILE + (local as u32 % TILE);
                let y = ty * TILE + (local as u32 / TILE);
                let covered =
                    x < inp.gbuf.width && y < inp.gbuf.height && inp.gbuf.tri.get((y * inp.gbuf.width + x) as usize).is_some_and(|tri| *tri != NO_TRI);
                coverage.push(u32::from(covered));
            }
            for (channel_index, &channel) in channels.iter().enumerate() {
                for (layer_index, id) in layers.iter().enumerate() {
                    let base = ((tile_index * channel_count + channel_index) * layer_count + layer_index) * 8;
                    let Some(layer) = inp.project.layers.get(id) else { return false };
                    if !layer.visible {
                        continue;
                    }
                    let Some(blend) = layer.blend(channel) else { continue };
                    if blend.mode == BlendMode::Disable || blend.opacity <= 0.0 {
                        continue;
                    }
                    let replace_mode = match blend.mode {
                        BlendMode::Normal | BlendMode::Passthrough => 0,
                        BlendMode::Replace => 1,
                        _ => return false,
                    };
                    let opacity = blend.opacity.clamp(0.0, 1.0);
                    match &layer.kind {
                        LayerKind::Paint => {
                            let raw = inp.pixels.layer(layer.id, channel).and_then(|image| image.tile(tx, ty));
                            let preview = inp.preview.filter(|preview| matches!(preview.target, PaintTarget::Layer(target) if target == layer.id));
                            let alpha = preview.and_then(|preview| preview.alpha.tile(tx, ty));
                            let preview_value = preview.and_then(|preview| preview.values.get(&channel));
                            if raw.is_none() && (preview.is_none() || (alpha.is_none() || (!preview.is_some_and(|p| p.erase) && preview_value.is_none()))) {
                                continue;
                            }
                            let expected = TILE_AREA * (channel.components() + 1);
                            if raw.is_some_and(|raw| raw.len() != expected) {
                                return false;
                            }
                            let offset = paint_data.len() as u32;
                            if let Some(alpha) = alpha.filter(|_| preview.is_some_and(|p| p.erase || preview_value.is_some())) {
                                let preview = match preview {
                                    Some(preview) => preview,
                                    None => return false,
                                };
                                let components = channel.components();
                                for texel in 0..TILE_AREA {
                                    let base = texel * (components + 1);
                                    let mut value = [0.0f32; 3];
                                    let mut coverage = raw.and_then(|raw| raw.get(base + components)).copied().map(from_u16).unwrap_or(0.0);
                                    for component in 0..components {
                                        let encoded = raw.and_then(|raw| raw.get(base + component)).copied().unwrap_or(0);
                                        value[component] = if channel.kind() == ChannelKind::Signed { from_u16_signed(encoded) } else { from_u16(encoded) };
                                    }
                                    let alpha = from_u16(alpha.get(texel).copied().unwrap_or(0));
                                    let amount = (alpha * preview.opacity).clamp(0.0, 1.0);
                                    if preview.erase {
                                        value = value.map(|v| v * (1.0 - amount));
                                        coverage *= 1.0 - amount;
                                    } else if let Some(painted) = preview_value {
                                        for component in 0..components {
                                            value[component] = painted[component] * amount + value[component] * (1.0 - amount);
                                        }
                                        coverage = amount + coverage * (1.0 - amount);
                                    }
                                    for component in 0..components {
                                        paint_data.push(u32::from(if channel.kind() == ChannelKind::Signed {
                                            to_u16_signed(value[component])
                                        } else {
                                            to_u16(value[component])
                                        }));
                                    }
                                    paint_data.push(u32::from(to_u16(coverage)));
                                }
                            } else if let Some(raw) = raw {
                                paint_data.extend(raw.iter().map(|value| u32::from(*value)));
                            } else {
                                continue;
                            }
                            infos[base..base + 8].copy_from_slice(&[2, replace_mode, opacity.to_bits(), offset, 0, 0, 0, 0]);
                        }
                        LayerKind::Fill(fill) => {
                            if fill.link.is_some()
                                || fill.projection != Projection::default()
                                || fill.sources.values().any(|src| !matches!(src, Source::Uniform(_)))
                            {
                                return false;
                            }
                            let value = match fill.sources.get(&channel) {
                                Some(Source::Uniform(value)) => *value,
                                _ => channel.default_value(),
                            };
                            if value.iter().any(|component| !component.is_finite()) {
                                return false;
                            }
                            infos[base..base + 8].copy_from_slice(&[
                                1,
                                replace_mode,
                                opacity.to_bits(),
                                0,
                                value[0].to_bits(),
                                value[1].to_bits(),
                                value[2].to_bits(),
                                0,
                            ]);
                        }
                        LayerKind::Group { .. } | LayerKind::Instance { .. } => return false,
                    }
                    if layer.mask.is_some() || !layer.content_effects.is_empty() || layer.height_mask.enabled || layer.geometry_mask.is_some() {
                        return false;
                    }
                }
            }
        }
        if paint_data.is_empty() {
            paint_data.push(0);
        }

        let outputs = tile_count * channel_count * TILE_AREA * 3;
        let output_bytes = (outputs * std::mem::size_of::<u32>()) as u64;
        let input_bytes = (paint_data.len() * std::mem::size_of::<u32>()) as u64;
        let storage_limit = u64::from(self.device.limits().max_storage_buffer_binding_size);
        if output_bytes > storage_limit || input_bytes > storage_limit {
            return false;
        }

        let input_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("texturecraft-composite-input"),
            contents: bytemuck::cast_slice(&paint_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let layer_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("texturecraft-composite-layers"),
            contents: bytemuck::cast_slice(&infos),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let channel_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("texturecraft-composite-channels"),
            contents: bytemuck::cast_slice(&channel_info),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let coverage_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("texturecraft-composite-coverage"),
            contents: bytemuck::cast_slice(&coverage),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("texturecraft-composite-output"),
            size: output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("texturecraft-composite-readback"),
            size: output_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let params = [tile_count as u32, channel_count as u32, layer_count as u32, 0];
        let param_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("texturecraft-composite-params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("texturecraft-composite-bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: layer_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: channel_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: coverage_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: output_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: param_buffer.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("texturecraft-composite") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("texturecraft-composite"), timestamp_writes: None });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let work = (tile_count * channel_count * TILE_AREA) as u32;
            pass.dispatch_workgroups(work.div_ceil(WORKGROUP), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, output_bytes);
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        if self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).is_err() || rx.recv().ok().is_none_or(|result| result.is_err()) {
            return false;
        }
        let mapped = match staging.slice(..).get_mapped_range() {
            Ok(mapped) => mapped,
            Err(_) => return false,
        };
        let Ok(values) = bytemuck::try_cast_slice::<u8, u32>(&mapped) else { return false };
        for (tile_index, &(tx, ty)) in tiles.iter().enumerate() {
            for (channel_index, &channel) in channels.iter().enumerate() {
                let comps = channel.components();
                let signed = channel.kind() == ChannelKind::Signed;
                let image = match out.channels.get_mut(&channel) {
                    Some(image) => image,
                    None => return false,
                };
                let base = (tile_index * channel_count + channel_index) * TILE_AREA * 3;
                let mut tile = Vec::<u16>::with_capacity(TILE_AREA * comps);
                for pixel in 0..TILE_AREA {
                    for component in 0..comps {
                        let raw = values.get(base + pixel * 3 + component).copied().unwrap_or(0);
                        let value = f32::from_bits(raw);
                        tile.push(if signed { to_u16_signed(value) } else { to_u16(value) });
                    }
                }
                image.set_tile(tx, ty, Some(Arc::new(tile)));
            }
        }
        drop(mapped);
        staging.unmap();
        true
    }
}

fn supported(inp: &Inputs<'_>) -> bool {
    if inp.preview.is_some_and(|preview| preview.stencil.is_some() || !preview.carried.is_empty() || matches!(preview.target, PaintTarget::Mask(_))) {
        return false;
    }
    let Ok(set) = inp.project.texture_set(inp.ts) else { return false };
    set.stack.iter().all(|id| {
        let Some(layer) = inp.project.layers.get(id) else { return false };
        !layer.visible
            || (matches!(layer.kind, LayerKind::Paint | LayerKind::Fill(_))
                && layer.mask.is_none()
                && layer.content_effects.is_empty()
                && !layer.height_mask.enabled
                && layer.geometry_mask.is_none())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use texturecraft_engine::geom::glam::Vec2;
    use texturecraft_engine::geom::primitives;

    #[test]
    fn gpu_tiles_match_cpu_oracle_for_simple_paint_and_fill_stack() {
        let Ok(headless) = texturecraft_render::headless::Headless::new() else {
            eprintln!("no wgpu adapter; GPU-compositor oracle check skipped");
            return;
        };
        eprintln!("GPU-compositor oracle adapter: {}", headless.adapter_name);
        let gpu = GpuComposite::new(Arc::new(headless.device), Arc::new(headless.queue));
        let Ok(mut session) = texturecraft_engine::Session::new_project("gpu-oracle", primitives::sphere(1.0, 32, 16), "builtin:sphere", None, 128) else {
            panic!("test project creation failed");
        };
        for (command, args) in [
            ("layer.add.fill", json!({"color": [0.7, 0.18, 0.32], "roughness": 0.43, "metallic": 0.27})),
            ("layer.add.fill", json!({"color": [0.1, 0.6, 0.24], "roughness": 0.71})),
            ("layer.add.paint", json!({})),
        ] {
            if let Err(error) = texturecraft_engine::execute(&mut session, command, &args) {
                panic!("{command} failed in test fixture: {error}");
            }
        }
        session.viewport = Vec2::new(800.0, 800.0);
        if let Err(error) = session.begin_stroke(Vec2::new(180.0, 240.0), session.viewport, 1.0) {
            panic!("test stroke start failed: {error}");
        }
        session.continue_stroke(Vec2::new(620.0, 410.0), 1.0);
        session.update();
        let cpu = session.caches[0].composite.clone();
        session.caches[0].invalidate_all();
        let mut accelerated_tiles = 0usize;
        session.update_for_with_accelerator(
            f64::INFINITY,
            || 0.0,
            |inputs, out, tiles| {
                let accelerated = gpu.composite(inputs, out, tiles);
                if accelerated {
                    accelerated_tiles += tiles.len();
                }
                accelerated
            },
        );
        assert!(accelerated_tiles > 0, "the fixture did not use the GPU fast path");
        let mut mae = 0.0f64;
        let mut samples = 0u64;
        for (&channel, expected) in &cpu.channels {
            let Some(actual) = session.caches[0].composite.channels.get(&channel) else {
                panic!("GPU output omitted channel {channel:?}");
            };
            let components = channel.components();
            for y in 0..128 {
                for x in 0..128 {
                    let a = expected.get(x, y);
                    let b = actual.get(x, y);
                    for component in 0..components {
                        let scale = if channel.kind() == ChannelKind::Signed { 32767.5 } else { 65535.0 };
                        let error = (f64::from(a[component]) - f64::from(b[component])).abs() / scale;
                        mae += error;
                        samples += 1;
                    }
                }
            }
        }
        let mean = mae / samples.max(1) as f64;
        assert!(mean <= 0.00002, "GPU/CPU normalized RGBA MAE {mean} exceeded 0.00002");
    }
}
