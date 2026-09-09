// Camera calibration models + JSON load/save (AGPL-3.0-or-later).
//
// All params are resolution-independent (normalized); pixel values are
// derived from the input image size at runtime.
// Real calibration values come from `scripts/calibrate_opencv.py`
// (checkerboard + OpenCV) which exports this exact JSON schema.

use serde::{Deserialize, Serialize};

/// Full camera model. Centers are fixed at the image center.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "model", rename_all = "snake_case")]
pub enum CameraModel {
    /// Identity (useful for T4 rotation-only inputs).
    Identity,
    /// Division model: r_d = r / (1 + lambda * r^2). lambda ~ 0.2-0.5.
    Division { lambda: f32 },
    /// Brown-Conrady radial + tangential (OpenCV `calibrateCamera` order).
    BrownConrady { k1: f32, k2: f32, k3: f32, p1: f32, p2: f32 },
}

impl Default for CameraModel {
    fn default() -> Self {
        Self::Division { lambda: 0.35 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    #[serde(default)]
    pub model: CameraModel,
}

impl Default for Calibration {
    fn default() -> Self {
        Self { model: CameraModel::default() }
    }
}

impl Calibration {
    pub fn identity() -> Self {
        Self { model: CameraModel::Identity }
    }

    pub fn division(lambda: f32) -> Self {
        Self { model: CameraModel::Division { lambda } }
    }

    pub fn load_json(text: &str) -> Result<Self, crate::InferError> {
        let bad = |e: String| crate::InferError::BadParam(format!("bad calibration json: {e}"));
        let v: serde_json::Value = serde_json::from_str(text).map_err(|e| bad(e.to_string()))?;
        let get = |k: &str| -> Result<f32, crate::InferError> {
            v.get(k).and_then(|x| x.as_f64()).map(|x| x as f32).ok_or_else(|| bad(format!("missing '{k}'")))
        };
        let model = v.get("model").and_then(|m| m.as_str()).ok_or_else(|| bad("missing 'model'".into()))?;
        let model = match model {
            "identity" => CameraModel::Identity,
            "division" => CameraModel::Division { lambda: get("lambda")? },
            "brown_conrady" | "brown-conrady" => CameraModel::BrownConrady {
                k1: get("k1")?,
                k2: v.get("k2").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
                k3: v.get("k3").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
                p1: v.get("p1").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
                p2: v.get("p2").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
            },
            other => return Err(bad(format!("unknown model '{other}'"))),
        };
        Ok(Self { model })
    }

    pub fn to_json_pretty(&self) -> String {
        let mut m = serde_json::Map::new();
        match self.model {
            CameraModel::Identity => {
                m.insert("model".into(), serde_json::Value::String("identity".into()));
            }
            CameraModel::Division { lambda } => {
                m.insert("model".into(), serde_json::Value::String("division".into()));
                m.insert("lambda".into(), serde_json::json!(lambda));
            }
            CameraModel::BrownConrady { k1, k2, k3, p1, p2 } => {
                m.insert("model".into(), serde_json::Value::String("brown_conrady".into()));
                m.insert("k1".into(), serde_json::json!(k1));
                m.insert("k2".into(), serde_json::json!(k2));
                m.insert("k3".into(), serde_json::json!(k3));
                m.insert("p1".into(), serde_json::json!(p1));
                m.insert("p2".into(), serde_json::json!(p2));
            }
        }
        serde_json::to_string_pretty(&m).unwrap_or_else(|_| "{}".into())
    }

    /// Backward undistort: for each OUTPUT (rectified) pixel, find the
    /// SOURCE (distorted) coordinate and resample.
    pub fn undistort(&self, img: &image::RgbImage) -> image::RgbImage {
        use crate::sampler::remap;
        let (w, h) = (img.width() as f32, img.height() as f32);
        let cx = w / 2.0;
        let cy = h / 2.0;
        let f = w.max(h);
        let norm = (cx * cx + cy * cy).sqrt();
        match self.model {
            CameraModel::Identity => img.clone(),
            CameraModel::Division { lambda } => remap(img, |x, y| {
                let dx = (x as f32 - cx) / norm;
                let dy = (y as f32 - cy) / norm;
                let rd = (dx * dx + dy * dy).sqrt();
                let r = crate::fisheye::invert_radius(rd, lambda);
                let s = if rd > 1e-6 { r / rd } else { 1.0 };
                (cx + dx * s * norm, cy + dy * s * norm)
            }),
            CameraModel::BrownConrady { k1, k2, k3, p1, p2 } => remap(img, |x, y| {
                let xn = (x as f32 - cx) / f;
                let yn = (y as f32 - cy) / f;
                let r2 = xn * xn + yn * yn;
                let r4 = r2 * r2;
                let r6 = r4 * r2;
                let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
                let xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn);
                let yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn;
                (cx + xd * f, cy + yd * f)
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_roundtrip() {
        let mut img = image::RgbImage::new(16, 12);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = image::Rgb([(x * 13) as u8, (y * 17) as u8, 7]);
        }
        let out = Calibration::identity().undistort(&img);
        assert!(out.pixels().zip(img.pixels()).all(|(a, b)| a == b));
    }

    #[test]
    fn json_roundtrip() {
        let c = Calibration::division(0.3);
        let back = Calibration::load_json(&c.to_json_pretty()).unwrap();
        assert_eq!(c, back);
        assert!(Calibration::load_json("{bad").is_err());
    }

    #[test]
    fn brown_conrady_zero_is_identity() {
        let img = image::RgbImage::new(32, 24);
        let c = Calibration { model: CameraModel::BrownConrady { k1: 0.0, k2: 0.0, k3: 0.0, p1: 0.0, p2: 0.0 } };
        let out = c.undistort(&img);
        assert_eq!(out.dimensions(), img.dimensions());
    }
}
