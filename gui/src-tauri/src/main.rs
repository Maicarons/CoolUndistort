// CoolUndistort Tauri backend (AGPL-3.0-or-later).
//
// ALL inference runs here, in Rust, via coolundistort-infer.
// No Python service: the frontend calls `undistort_image` via invoke
// and gets a base64 PNG back for the before/after view.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use base64::Engine;
use coolundistort_infer::{CameraParams, InferMode, Task, undistort};
use image::ImageFormat;
use std::io::Cursor;

#[tauri::command]
fn undistort_image(image_bytes: Vec<u8>, task: String, mode: String, lambda: Option<f32>) -> Result<String, String> {
    let task = Task::parse(&task).map_err(|e| e.to_string())?;
    let mode = InferMode::parse(&mode).map_err(|e| e.to_string())?;
    let img = image::load_from_memory(&image_bytes)
        .map_err(|e| format!("bad image: {e}"))?
        .to_rgb8();
    let out = undistort(&img, task, mode, CameraParams { lambda: lambda.unwrap_or(0.35) })
        .map_err(|e| e.to_string())?;
    let mut buf = Cursor::new(Vec::new());
    out.write_to(&mut buf, ImageFormat::Png).map_err(|e| format!("encode: {e}"))?;
    Ok(base64::engine::general_purpose::STANDARD.encode(buf.into_inner()))
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![undistort_image])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
