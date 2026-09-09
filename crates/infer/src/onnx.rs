// Checkpoint / ONNX hook (AGPL-3.0-or-later).
//
// P1 integration point: Python training exports UniRect-lite to ONNX
// (`python scripts/export_onnx.py`), and this module loads it with the
// `ort` crate. Until a real `*.onnx` file is supplied, checkpoint mode
// returns a clear `NeedsWeights` error instead of silently degrading.

use image::RgbImage;

use crate::{InferError, Task};

/// Validate a checkpoint path and describe what would run.
/// Used by CLI (`--mode checkpoint --onnx model.onnx`) and Tauri.
pub fn run_checkpoint(img: &RgbImage, task: Task, path: &str) -> Result<RgbImage, InferError> {
    let _ = (img, task);
    if !std::path::Path::new(path).exists() {
        return Err(InferError::Onnx(format!("checkpoint not found: {path}")));
    }
    Err(InferError::Onnx(format!(
        "loaded {path}, but ONNX runtime is not wired yet: add `ort` dependency, \
         feed (image, prompt) per docs/TRAINING.md §6, then replace this stub"
    )))
}
