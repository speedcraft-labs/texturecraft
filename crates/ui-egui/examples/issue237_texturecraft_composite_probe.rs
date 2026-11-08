#[path = "../src/gpu_composite.rs"]
mod gpu_composite;

use std::collections::BTreeSet;
use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

use gpu_composite::GpuComposite;
use serde_json::json;
use sha2::{Digest, Sha256};
use texturecraft_engine::core::{Channel, ChannelKind};
use texturecraft_engine::geom::primitives;
use texturecraft_engine::texel::{TILE_AREA, to_u16};
use texturecraft_engine::{Session, execute};
use texturecraft_render::headless::Headless;

const RESOLUTION: u32 = 4096;
const TILE_COUNT: usize = 8;
const ERROR_BOUND: f64 = 0.00002;

struct Fixture {
    session: Session,
    tiles: Vec<(u32, u32)>,
}

fn fixture(input: &str) -> Result<Fixture, Box<dyn Error>> {
    let rgba = image::open(input)?.into_rgba8();
    let (source_width, source_height) = rgba.dimensions();
    let mut session = Session::new_project("issue-237-texturecraft", primitives::sphere(1.0, 32, 16), "builtin:sphere", None, RESOLUTION)?;

    for color in [[0.82, 0.2, 0.11], [0.14, 0.67, 0.28], [0.12, 0.25, 0.91]] {
        execute(&mut session, "layer.add.fill", &json!({"color": color, "roughness": 0.48, "metallic": 0.21}))?;
        if let Some(id) = session.selected_layer()
            && let Some(layer) = session.doc.project.layers.get_mut(&id)
        {
            for blend in layer.channels.values_mut() {
                blend.opacity = 0.38;
            }
        }
    }

    let tiles = session.caches[0].covered_tiles.iter().copied().take(TILE_COUNT).collect::<Vec<_>>();
    if tiles.len() != TILE_COUNT {
        return Err("synthetic sphere did not cover enough tiles".into());
    }
    let paint_id = session.doc.project.texture_sets[0].stack[0];
    let base = session.doc.pixels.layer_mut(paint_id, Channel::BaseColor, (RESOLUTION, RESOLUTION));
    for &(tx, ty) in &tiles {
        let mut data = vec![0u16; TILE_AREA * (Channel::BaseColor.components() + 1)];
        for y in 0..64 {
            for x in 0..64 {
                let source = rgba.get_pixel((tx * 64 + x) % source_width, (ty * 64 + y) % source_height).0;
                let offset = ((y * 64 + x) as usize) * 4;
                let alpha = f32::from(source[3]) / 255.0;
                data[offset] = to_u16(f32::from(source[0]) / 255.0 * alpha);
                data[offset + 1] = to_u16(f32::from(source[1]) / 255.0 * alpha);
                data[offset + 2] = to_u16(f32::from(source[2]) / 255.0 * alpha);
                data[offset + 3] = to_u16(alpha);
            }
        }
        base.set_tile(tx, ty, Some(Arc::new(data)));
    }

    let cache = &mut session.caches[0];
    cache.dirty = tiles.iter().copied().collect::<BTreeSet<_>>();
    cache.settle.clear();
    cache.chunk = 0;
    Ok(Fixture { session, tiles })
}

fn digest(session: &Session, tiles: &[(u32, u32)]) -> Result<String, Box<dyn Error>> {
    let mut hash = Sha256::new();
    let composite = &session.caches[0].composite;
    for (channel, image) in &composite.channels {
        hash.update(format!("{channel:?}").as_bytes());
        for &(tx, ty) in tiles {
            let data = image.tile(tx, ty).ok_or("composite tile missing")?;
            for value in data {
                hash.update(value.to_le_bytes());
            }
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn compare(actual: &Session, oracle: &Session, tiles: &[(u32, u32)]) -> Result<(f64, f64), Box<dyn Error>> {
    let mut sum = 0.0f64;
    let mut max = 0.0f64;
    let mut count = 0usize;
    for (channel, expected) in &oracle.caches[0].composite.channels {
        let output = actual.caches[0].composite.channels.get(channel).ok_or("GPU omitted a channel")?;
        let scale = if channel.kind() == ChannelKind::Signed { 32767.5 } else { 65535.0 };
        for &(tx, ty) in tiles {
            let expected = expected.tile(tx, ty).ok_or("oracle tile missing")?;
            let output = output.tile(tx, ty).ok_or("output tile missing")?;
            for (&a, &b) in expected.iter().zip(output) {
                let error = (f64::from(a) - f64::from(b)).abs() / scale;
                sum += error;
                max = max.max(error);
                count += 1;
            }
        }
    }
    let mean = sum / count.max(1) as f64;
    if mean > ERROR_BOUND || max > ERROR_BOUND {
        return Err(format!("GPU/oracle error exceeded {ERROR_BOUND}: mean={mean:.8}, max={max:.8}").into());
    }
    Ok((mean, max))
}

fn main() -> Result<(), Box<dyn Error>> {
    let input = std::env::args().nth(1).ok_or("usage: issue237_texturecraft_composite_probe <corpus-png>")?;
    let use_gpu = std::env::var("TEXTURECRAFT_GPU").is_ok_and(|value| value == "1");

    let mut oracle = fixture(&input)?;
    oracle.session.update_for(f64::INFINITY, || 0.0);
    let oracle_hash = digest(&oracle.session, &oracle.tiles)?;

    let mut run = fixture(&input)?;
    let gpu = if use_gpu {
        let headless = Headless::new().map_err(|error| format!("wgpu headless initialization failed: {error}"))?;
        println!("CRAFT_GPU_ADAPTER={}", headless.adapter_name);
        Some(GpuComposite::new(Arc::new(headless.device), Arc::new(headless.queue)))
    } else {
        None
    };

    let start = Instant::now();
    if let Some(gpu) = &gpu {
        let mut accelerated = 0usize;
        run.session.update_for_with_accelerator(
            f64::INFINITY,
            || start.elapsed().as_secs_f64() * 1000.0,
            |inputs, output, tiles| {
                let used = gpu.composite(inputs, output, tiles);
                if used {
                    accelerated += tiles.len();
                }
                used
            },
        );
        if accelerated != run.tiles.len() {
            return Err(format!("GPU fast path covered {accelerated} of {} fixture tiles", run.tiles.len()).into());
        }
        let (mean, max) = compare(&run.session, &oracle.session, &run.tiles)?;
        println!("TEXTURECRAFT_GPU_ORACLE_BOUND={ERROR_BOUND:.8} mean={mean:.8} max={max:.8}");
    } else {
        run.session.update_for(f64::INFINITY, || start.elapsed().as_secs_f64() * 1000.0);
    }
    let elapsed_ns = start.elapsed().as_nanos();
    println!("CRAFT_JOURNEY_NS={elapsed_ns}");
    println!("CRAFT_OUTPUT_SHA256={oracle_hash}");
    Ok(())
}
