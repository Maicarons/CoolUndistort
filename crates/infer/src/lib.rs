// CoolUndistort Rust inference core (AGPL-3.0-or-later).
//
// P0: division-model fisheye undistortion (single parameter, no weights).
// P1: RP-TPS grid warp + ONNX weights hook up here.
// Python is training-only; all inference paths go through this crate.

pub mod fisheye;
pub mod prompt;
pub mod tps;

use image::RgbImage;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    T1,
    T2,
    T3,
    T4,
}

impl Task {
    pub fn parse(s: &str) -> Result<Self, InferError> {
        match s.to_lowercase().as_str() {
            "t1" => Ok(Self::T1),
            "t2" => Ok(Self::T2),
            "t3" => Ok(Self::T3),
            "t4" => Ok(Self::T4),
            other => Err(InferError::UnknownTask(other.to_owned())),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferMode {
    Fisheye,
    Tps,
    Checkpoint,
}

impl InferMode {
    pub fn parse(s: &str) -> Result<Self, InferError> {
        match s.to_lowercase().as_str() {
            "fisheye" => Ok(Self::Fisheye),
            "tps" => Ok(Self::Tps),
            "checkpoint" => Ok(Self::Checkpoint),
            other => Err(InferError::UnknownMode(other.to_owned())),
        }
    }
}

/// Placeholder camera params; replaced by real calibration (P0) or a
/// calibration file loader.
#[derive(Debug, Clone, Copy)]
pub struct CameraParams {
    pub lambda: f32,
}

impl Default for CameraParams {
    fn default() -> Self {
        Self { lambda: 0.35 }
    }
}

#[derive(Debug, Error)]
pub enum InferError {
    #[error("unknown task: {0}")]
    UnknownTask(String),
    #[error("unknown mode: {0}")]
    UnknownMode(String),
    #[error("mode {0} needs P1 weights (see docs/TRAINING.md)")]
    NeedsWeights(&'static str),
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
}

/// Single entry point for ALL inference: CLI, Tauri commands, batch jobs.
pub fn undistort(
    img: &RgbImage,
    task: Task,
    mode: InferMode,
    params: CameraParams,
) -> Result<RgbImage, InferError> {
    let _ = task;
    match mode {
        InferMode::Fisheye => Ok(fisheye::undistort_division(img, params.lambda)),
        InferMode::Tps => Err(InferError::NeedsWeights("tps")),
        InferMode::Checkpoint => Err(InferError::NeedsWeights("checkpoint")),
    }
}
