// AutoLambda ONNX inference via tract (AGPL-3.0-or-later).
//
// Pure-Rust ONNX runtime: loads weights/autolambda.onnx (trained by
// `python scripts/train_autolambda.py`), regresses the division-model
// lambda from a 128x128 grayscale thumbnail, then undistorts the FULL
// resolution image with the estimated lambda. No calibration needed.

use image::RgbImage;
use tract_onnx::prelude::*;

use crate::{InferError, Task};

/// Bundled model path (repo-relative). CLI/Tauri may override with --onnx.
pub const BUNDLED_MODEL: &str = "weights/autolambda.onnx";

/// Preprocess: grayscale, resize to 128x128, normalize to [0, 1], NCHW.
fn preprocess(img: &RgbImage) -> Result<Tensor, String> {
    let gray = image::imageops::grayscale(img);
    let small = image::imageops::resize(&gray, 128, 128, image::imageops::FilterType::Triangle);
    let data: Vec<f32> = small.pixels().map(|p| p.0[0] as f32 / 255.0).collect();
    Tensor::from_shape(&[1, 1, 128, 128], &data).map_err(|e| format!("tensor shape: {e}"))
}

/// Estimate lambda for an image with the given model file.
pub fn estimate_lambda(img: &RgbImage, path: &str) -> Result<f32, InferError> {
    let model = tract_onnx::onnx()
        .model_for_path(path)
        .map_err(|e| InferError::Onnx(format!("cannot load {path}: {e}")))?
        .into_optimized()
        .map_err(|e| InferError::Onnx(format!("cannot optimize {path}: {e}")))?
        .into_runnable()
        .map_err(|e| InferError::Onnx(format!("cannot make runnable {path}: {e}")))?;
    let input = preprocess(img).map_err(InferError::Onnx)?;
    let result = model.run(tvec!(input.into())).map_err(|e| InferError::Onnx(format!("run: {e}")))?;
    let lambda: f32 = result[0]
        .to_plain_array_view::<f32>()
        .map_err(|e| InferError::Onnx(format!("output: {e}")))?
        .iter()
        .next()
        .copied()
        .ok_or_else(|| InferError::Onnx("empty output".into()))?;
    if !lambda.is_finite() {
        return Err(InferError::Onnx("non-finite lambda".into()));
    }
    Ok(lambda.clamp(0.0, 0.6))
}

fn resolve_model(explicit: Option<&str>) -> String {
    if let Some(p) = explicit.filter(|s| !s.is_empty()) {
        return p.to_owned();
    }
    if std::path::Path::new(BUNDLED_MODEL).exists() {
        return BUNDLED_MODEL.to_owned();
    }
    if let Ok(exe) = std::env::current_exe() {
        for dir in exe.ancestors().skip(1) {
            let cand = dir.join(BUNDLED_MODEL);
            if cand.exists() {
                return cand.to_string_lossy().into_owned();
            }
        }
    }
    BUNDLED_MODEL.to_owned()
}

/// Validate a checkpoint path and describe what would run.
/// Used by CLI (`--mode checkpoint --onnx model.onnx`) and Tauri.
pub fn run_checkpoint(img: &RgbImage, task: Task, path: &str) -> Result<RgbImage, InferError> {
    let _ = task;
    let model_path = resolve_model(Some(path));
    let lambda = estimate_lambda(img, &model_path)?;
    Ok(crate::fisheye::undistort_division(img, lambda))
}

/// Auto mode: estimate lambda with the bundled model, then undistort.
pub fn run_auto(img: &RgbImage, explicit: Option<&str>) -> Result<(RgbImage, f32), InferError> {
    let model_path = resolve_model(explicit);
    let lambda = estimate_lambda(img, &model_path)?;
    Ok((crate::fisheye::undistort_division(img, lambda), lambda))
}
