//! The JSON control channel: agents drive the running app with the same commands as the menus,
//! inspect session and UI state, invoke menu items, paint strokes and take screenshots.
//!
//! Methods (params as JSON objects):
//! - `command.run {id, params?}`: run an engine command (programmatic: never opens a dialog).
//! - `command.list {filter?}`: every command with params docs and enabled state.
//! - `session.inspect`: project, texture sets, layer stack, tool, camera, display, history.
//! - `ui.inspect`: the UI state (panels, windows, mode, layout, viewport rect, who draws the title bar).
//! - `ui.menu.list`: the menu catalog with ids, shortcuts and whether each is live.
//! - `ui.menu.invoke {id}`: behave like a menu click (may open a window).
//! - `ui.set {...}`: merge fields into the UI state (e.g. `{"layout":"Only3D"}`).
//! - `viewport.screenshot {view?: "3d"|"2d"}`: PNG of a viewport, base64.
//! - `ui.screenshot`: PNG of the whole window (replies after the next frame).
//! - `ui.focus`: explicitly activate the window and take keyboard focus.

use std::sync::mpsc::{Receiver, Sender, channel};

use base64::Engine as _;
use serde_json::{Value, json};

use crate::TextureCraftApp;

pub struct ControlRequest {
    pub method: String,
    pub params: Value,
    pub reply: Sender<Value>,
}

impl ControlRequest {
    pub fn new(method: String, params: Value) -> (ControlRequest, Receiver<Value>) {
        let (tx, rx) = channel();
        (ControlRequest { method, params, reply: tx }, rx)
    }
}

fn ok(v: Value) -> Value {
    json!({ "ok": true, "result": v })
}

fn err(e: impl std::fmt::Display) -> Value {
    json!({ "ok": false, "error": e.to_string() })
}

pub fn poll(app: &mut TextureCraftApp, ctx: &egui::Context) {
    let Some(rx) = app.control.as_ref() else { return };
    let reqs: Vec<ControlRequest> = rx.try_iter().collect();
    for r in reqs {
        raise_for_control_request(app.raise_without_focus, ctx, &r.method);
        if r.method == "ui.screenshot" {
            // Replies when the next frame has been captured.
            let reply = r.reply;
            let repaint = ctx.clone();
            let callback = egui::ScreenshotCallback::new(move |img| {
                let answer = encode_png(img.size[0] as u32, img.size[1] as u32, img.pixels.iter().flat_map(|c| c.to_array()).collect())
                    .map(|(w, h, png)| ok(json!({ "width": w, "height": h, "png": base64::engine::general_purpose::STANDARD.encode(png) })))
                    .unwrap_or_else(|| err("could not encode the screenshot"));
                let _ = reply.send(answer);
                repaint.request_repaint();
            });
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(callback));
            ctx.request_repaint();
            continue;
        }
        let reply = handle(app, ctx, &r.method, &r.params);
        let _ = r.reply.send(reply);
        ctx.request_repaint();
    }
}

/// Make the UI pass run for a control request without taking the user's keyboard focus.
fn raise_for_control_request(raise_without_focus: Option<fn() -> bool>, ctx: &egui::Context, method: &str) {
    if !((method.starts_with("ui.") && method != "ui.focus") || method == "viewport.screenshot") {
        return;
    }
    // An agent window starts below normal windows; restore its level before raising it in front.
    #[cfg(target_os = "macos")]
    ctx.send_viewport_cmd(egui::ViewportCommand::WindowLevel(egui::viewport::WindowLevel::Normal));
    if let Some(raise) = raise_without_focus {
        raise();
    }
    // A hidden native window still needs a real UI pass for widget state/geometry.
    ctx.send_viewport_cmd(egui::ViewportCommand::RequestPaintWhileHidden);
    ctx.request_repaint();
}

pub(crate) fn encode_png(w: u32, h: u32, rgba: Vec<u8>) -> Option<(u32, u32, Vec<u8>)> {
    let img = image::RgbaImage::from_raw(w, h, rgba)?;
    let mut png = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).ok()?;
    Some((w, h, png))
}

pub fn handle(app: &mut TextureCraftApp, ctx: &egui::Context, method: &str, p: &Value) -> Value {
    app.ui.prefs.apply_resource_limits();
    match method {
        "ping" => ok(json!("pong")),
        "ui.resize" => {
            let size = |k: &str| p.get(k).and_then(Value::as_f64).filter(|v| v.is_finite()).map(|v| v.clamp(640.0, 7680.0) as f32);
            match (size("width"), size("height")) {
                (Some(w), Some(h)) => {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(false));
                    ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(w, h)));
                    ok(Value::Null)
                }
                _ => err("`width` and `height` (points, 640..7680) are required"),
            }
        }
        "ui.focus" => {
            // Explicit request: activate the app and take keyboard focus.
            #[cfg(target_os = "macos")]
            ctx.send_viewport_cmd(egui::ViewportCommand::WindowLevel(egui::viewport::WindowLevel::Normal));
            ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
            ok(Value::Null)
        }
        "debug.frames" => ok(app.profiler.report()),
        "debug.paintTest" => match crate::profile::start_paint_test(app, p) {
            Ok(()) => ok(Value::Null),
            Err(e) => err(e),
        },
        "command.run" => {
            let Some(id) = p.get("id").and_then(Value::as_str) else { return err("`id` is required") };
            let params = p.get("params").cloned().unwrap_or(Value::Null);
            match texturecraft_engine::execute(&mut app.session, id, &params) {
                Ok(v) => ok(v),
                Err(e) => err(e),
            }
        }
        "command.list" => ok(texturecraft_engine::commands::list(&app.session, p.get("filter").and_then(Value::as_str))),
        "session.inspect" => ok(texturecraft_engine::inspect::session(&app.session)),
        "ui.inspect" => {
            let r = app.viewport_rect;
            ok(json!({
                "state": app.ui,
                "viewport": [r.min.x, r.min.y, r.width(), r.height()],
                "fps": app.fps,
                "uiMs": app.ui_ms,
                "titleBar": app.title_bar.name(),
                "baking": app.bake_progress(),
                "bakeStatus": app.bake_status(),
                "bakePreview": texturecraft_engine::bake_preview::preview_json(&app.session),
                "toasts": app.toasts.iter().map(|t| json!({"text": t.text, "error": t.error})).collect::<Vec<_>>(),
            }))
        }
        "ui.bake.start" => {
            // Bake in the background like the Bake button: the 3D view shows the maps as they finish
            let count = app.session.doc.project.texture_sets.len();
            let sets: Vec<usize> = match p.get("ts") {
                Some(value) => match serde_json::from_value::<Vec<usize>>(if value.is_array() { value.clone() } else { json!([value]) }) {
                    Ok(list) if list.iter().all(|&index| index < count) => list,
                    _ => return err(format!("`ts` must be texture set indices below {count}")),
                },
                None => app
                    .session
                    .doc
                    .project
                    .texture_sets
                    .iter()
                    .enumerate()
                    .filter(|(_, texture_set)| texture_set.bake.common.is_enabled)
                    .map(|(index, _)| index)
                    .collect(),
            };
            if app.bake.is_some() {
                return err("a bake is already running");
            }

            if sets.is_empty() {
                return err("no texture set to bake: tick one in Bake mode or pass `ts`");
            }

            app.ui.mode = crate::state::Mode::Baking;
            app.start_bake(sets);
            ok(texturecraft_engine::bake_preview::preview_json(&app.session))
        }
        "ui.bake.cancel" => {
            app.cancel_bake();
            ok(Value::Null)
        }
        "ui.menu.list" => ok(json!(
            crate::menus::CATALOG
                .iter()
                .filter(|i| i.label != "-")
                .map(|i| json!({
                    "menu": i.menu, "submenu": i.sub, "label": i.label, "id": i.id, "shortcut": crate::keymap::shortcut_of(&app.ui.keymap, i.id),
                    "implemented": crate::menus::implemented(i.id), "enabled": crate::menus::enabled(app, i.id),
                }))
                .collect::<Vec<_>>()
        )),
        "ui.menu.invoke" => {
            let Some(id) = p.get("id").and_then(Value::as_str) else { return err("`id` is required") };
            if !crate::menus::implemented(id) {
                return err(format!("menu item `{id}` is not implemented"));
            }
            crate::menus::invoke(app, ctx, id);
            ok(Value::Null)
        }
        "ui.set" => {
            let mut cur = serde_json::to_value(&app.ui).unwrap_or(Value::Null);
            if let (Some(obj), Some(patch)) = (cur.as_object_mut(), p.as_object()) {
                for (k, v) in patch {
                    obj.insert(k.clone(), v.clone());
                }
            }
            match serde_json::from_value(cur) {
                Ok(ui) => {
                    app.ui = ui;
                    app.ui.prefs.apply_resource_limits();
                    ok(Value::Null)
                }
                Err(e) => err(format!("invalid UI state: {e}")),
            }
        }
        "viewport.screenshot" => {
            let view = if p.get("view").and_then(Value::as_str) == Some("2d") { crate::viewport::VIEW_2D } else { crate::viewport::VIEW_3D };
            match screenshot(app, view) {
                Some((w, h, png)) => ok(json!({ "width": w, "height": h, "png": base64::engine::general_purpose::STANDARD.encode(png) })),
                None => err("no rendered view to capture yet"),
            }
        }
        other => err(format!("unknown method `{other}`")),
    }
}

/// Encode a rendered view as PNG.
pub fn screenshot(app: &TextureCraftApp, view: u64) -> Option<(u32, u32, Vec<u8>)> {
    let rs = app.render_state.as_ref()?;
    let renderer = rs.renderer.read();
    let gpu = renderer.callback_resources.get::<crate::gpu::GpuState>()?;
    let (w, h, rgba) = gpu.renderer.read_view(&rs.device, &rs.queue, view)?;
    encode_png(w, h, rgba)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static RAISES: AtomicUsize = AtomicUsize::new(0);

    fn raise_without_focus() -> bool {
        RAISES.fetch_add(1, Ordering::SeqCst);
        true
    }

    fn commands_for(method: &str) -> Vec<egui::ViewportCommand> {
        let ctx = egui::Context::default();
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| raise_for_control_request(Some(raise_without_focus), ui.ctx(), method));
        let commands = output.viewport_output.remove(&egui::ViewportId::ROOT).map(|root| root.commands).unwrap_or_default();
        output.textures_delta.clear();
        commands
    }

    #[test]
    fn ui_requests_and_screenshots_raise_without_focus_but_ui_focus_stays_explicit() {
        RAISES.store(0, Ordering::SeqCst);
        let commands = commands_for("ui.inspect");
        assert_eq!(RAISES.load(Ordering::SeqCst), 1);
        assert!(commands.contains(&egui::ViewportCommand::RequestPaintWhileHidden));
        #[cfg(target_os = "macos")]
        assert!(commands.contains(&egui::ViewportCommand::WindowLevel(egui::viewport::WindowLevel::Normal)));

        let commands = commands_for("ui.screenshot");
        assert_eq!(RAISES.load(Ordering::SeqCst), 2);
        assert!(commands.contains(&egui::ViewportCommand::RequestPaintWhileHidden));

        let commands = commands_for("viewport.screenshot");
        assert_eq!(RAISES.load(Ordering::SeqCst), 3);
        assert!(commands.contains(&egui::ViewportCommand::RequestPaintWhileHidden));

        let commands = commands_for("ui.focus");
        assert_eq!(RAISES.load(Ordering::SeqCst), 3);
        assert!(!commands.contains(&egui::ViewportCommand::RequestPaintWhileHidden));
        #[cfg(target_os = "macos")]
        assert!(!commands.contains(&egui::ViewportCommand::WindowLevel(egui::viewport::WindowLevel::Normal)));
    }
}
