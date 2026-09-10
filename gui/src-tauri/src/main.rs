// CoolUndistort Tauri backend (AGPL-3.0-or-later).
//
// ALL inference runs here, in Rust, via coolundistort-infer.
// Frontend calls `undistort_image` via invoke and gets a base64 PNG back.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use base64::Engine;
use coolundistort_infer::{Calibration, InferMode, InferParams, Task, undistort_full};
use image::ImageFormat;
use std::io::Cursor;

#[tauri::command]
fn undistort_image(
    image_bytes: Vec<u8>,
    task: String,
    mode: String,
    lambda: Option<f32>,
    calib_json: Option<String>,
    angle_deg: Option<f32>,
    onnx_path: Option<String>,
    tps_deltas_json: Option<String>,
    tps_grid: Option<String>,
) -> Result<String, String> {
    let task = Task::parse(&task).map_err(|e| e.to_string())?;
    let mode = InferMode::parse(&mode).map_err(|e| e.to_string())?;
    let mut params = InferParams::default();
    if let Some(json) = calib_json.filter(|s| !s.trim().is_empty()) {
        params.calib = Calibration::load_json(&json).map_err(|e| e.to_string())?;
    } else {
        params.calib = Calibration::division(lambda.unwrap_or(0.35));
    }
    params.angle_deg = angle_deg.unwrap_or(0.0);
    params.onnx_path = onnx_path.filter(|s| !s.trim().is_empty());
    if let Some(raw) = tps_deltas_json.filter(|s| !s.trim().is_empty()) {
        let deltas: Vec<[f32; 2]> =
            serde_json::from_str(&raw).map_err(|e| format!("bad tps deltas json: {e}"))?;
        params.tps_deltas = Some(deltas.into_iter().map(|a| (a[0], a[1])).collect());
    }
    if let Some(g) = tps_grid.filter(|s| !s.trim().is_empty()) {
        let (h, w) = g.split_once(['x', 'X']).unwrap_or(("10", "12"));
        params.tps_grid = (h.parse().unwrap_or(10), w.parse().unwrap_or(12));
    }
    let img = image::load_from_memory(&image_bytes)
        .map_err(|e| format!("bad image: {e}"))?
        .to_rgb8();
    let out = undistort_full(&img, task, mode, &params).map_err(|e| e.to_string())?;
    let mut buf = Cursor::new(Vec::new());
    out.write_to(&mut buf, ImageFormat::Png).map_err(|e| format!("encode: {e}"))?;
    Ok(base64::engine::general_purpose::STANDARD.encode(buf.into_inner()))
}

/// Auto mode with the estimated lambda reported: {image, lambda}.
#[tauri::command]
fn undistort_auto(image_bytes: Vec<u8>, onnx_path: Option<String>) -> Result<serde_json::Value, String> {
    let img = image::load_from_memory(&image_bytes)
        .map_err(|e| format!("bad image: {e}"))?
        .to_rgb8();
    let (out, lambda) =
        coolundistort_infer::onnx::run_auto(&img, onnx_path.as_deref()).map_err(|e| e.to_string())?;
    let mut buf = Cursor::new(Vec::new());
    out.write_to(&mut buf, ImageFormat::Png).map_err(|e| format!("encode: {e}"))?;
    Ok(serde_json::json!({
        "image": base64::engine::general_purpose::STANDARD.encode(buf.into_inner()),
        "lambda": lambda,
    }))
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![undistort_image, undistort_auto])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
