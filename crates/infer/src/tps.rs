// RP-TPS warp skeleton (P1 target).
//
// Mirrors the Python training model (coolundistort/model/deformation.py):
// two-step residual control points, both samples taken from the ORIGINAL
// input to avoid intermediate interpolation buildup. The learned predictors
// C0/C1 come from training (ONNX weights, loaded in P1); this module owns
// the closed-form TPS solve + grid_sample, in Rust, with no Python.

/// Uniform base control points in normalized [-1, 1] coords.
pub fn base_points(grid_h: usize, grid_w: usize) -> Vec<(f32, f32)> {
    let mut pts = Vec::with_capacity(grid_h * grid_w);
    for iy in 0..grid_h {
        for ix in 0..grid_w {
            let x = if grid_w > 1 { ix as f32 / (grid_w - 1) as f32 * 2.0 - 1.0 } else { 0.0 };
            let y = if grid_h > 1 { iy as f32 / (grid_h - 1) as f32 * 2.0 - 1.0 } else { 0.0 };
            pts.push((x, y));
        }
    }
    pts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_grid_count_and_corners() {
        let pts = base_points(10, 12);
        assert_eq!(pts.len(), 120);
        assert_eq!(pts[0], (-1.0, -1.0));
        assert_eq!(pts[119], (1.0, 1.0));
    }
}
