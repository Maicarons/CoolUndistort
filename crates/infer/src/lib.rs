// CoolUndistort Rust inference core (AGPL-3.0-or-later).
//
// Python is training-only; ALL inference (CLI, Tauri, batch) goes through
// `undistort()` in this crate.

pub mod calibration;
pub mod fisheye;
pub mod onnx;
pub mod prompt;
pub mod rotation;
pub mod sampler;
pub mod tps;

use image::RgbImage;
use thiserror::Error;

pub use calibration::{Calibration, CameraModel};

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
    /// Blind: estimate lambda with the bundled ONNX model, then undistort.
    Auto,
    Fisheye,
    Tps,
    Checkpoint,
}

impl InferMode {
    pub fn parse(s: &str) -> Result<Self, InferError> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "fisheye" => Ok(Self::Fisheye),
            "tps" => Ok(Self::Tps),
            "checkpoint" => Ok(Self::Checkpoint),
            other => Err(InferError::UnknownMode(other.to_owned())),
        }
    }
}

/// Backwards-compatible single-parameter alias: Division{lambda}.
#[derive(Debug, Clone, Copy)]
pub struct CameraParams {
    pub lambda: f32,
}

impl Default for CameraParams {
    fn default() -> Self {
        Self { lambda: 0.35 }
    }
}

impl From<CameraParams> for calibration::Calibration {
    fn from(p: CameraParams) -> Self {
        Self::division(p.lambda)
    }
}

/// Extra knobs shared by CLI / Tauri / batch.
#[derive(Debug, Clone, Default)]
pub struct InferParams {
    pub calib: calibration::Calibration,
    /// T4: clockwise degrees present in the input; correction rotates back.
    pub angle_deg: f32,
    /// TPS mode: per-control-point (dx, dy) in normalized units.
    pub tps_deltas: Option<Vec<(f32, f32)>>,
    pub tps_grid: (usize, usize),
    /// Checkpoint mode: exported ONNX path (P1).
    pub onnx_path: Option<String>,
}

#[derive(Debug, Error)]
pub enum InferError {
    #[error("unknown task: {0}")]
    UnknownTask(String),
    #[error("unknown mode: {0}")]
    UnknownMode(String),
    #[error("bad parameter: {0}")]
    BadParam(String),
    #[error("mode {0} needs P1 weights (see docs/TRAINING.md)")]
    NeedsWeights(&'static str),
    #[error("onnx: {0}")]
    Onnx(String),
    #[error("io: {0}")]
    Io(String),
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
    let full = InferParams { calib: params.into(), ..Default::default() };
    undistort_full(img, task, mode, &full)
}

pub fn undistort_full(
    img: &RgbImage,
    task: Task,
    mode: InferMode,
    params: &InferParams,
) -> Result<RgbImage, InferError> {
    match mode {
        InferMode::Auto => {
            let (out, _) = onnx::run_auto(img, params.onnx_path.as_deref())?;
            Ok(out)
        }
        InferMode::Fisheye => {
            // Task-aware defaults: T1 faces need a gentler warp, T4 is
            // rotation-only (identity geometry + rotation fix).
            let mut calib = params.calib;
            if task == Task::T1 {
                if let calibration::CameraModel::Division { lambda } = calib.model {
                    calib.model = calibration::CameraModel::Division { lambda: lambda * 0.7 };
                }
            }
            if task == Task::T4 {
                calib = calibration::Calibration::identity();
            }
            let out = calib.undistort(img);
            if task == Task::T4 && params.angle_deg.abs() > 1e-6 {
                Ok(rotation::rotate_image(&out, -params.angle_deg))
            } else {
                Ok(out)
            }
        }
        InferMode::Tps => {
            let deltas = params.tps_deltas.as_ref().ok_or(InferError::NeedsWeights("tps"))?;
            let (gh, gw) = params.tps_grid;
            let (gh, gw) = if gh == 0 || gw == 0 { (10, 12) } else { (gh, gw) };
            tps::tps_warp_with_deltas(img, gh, gw, deltas)
        }
        InferMode::Checkpoint => {
            let path = params.onnx_path.as_deref().ok_or(InferError::NeedsWeights("checkpoint"))?;
            onnx::run_checkpoint(img, task, path)
        }
    }
}
