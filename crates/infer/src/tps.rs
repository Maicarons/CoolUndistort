// Thin-plate spline warp, closed-form (AGPL-3.0-or-later).
//
// Mirrors the Python training warp (coolundistort/model/deformation.py):
// uniform base points on the rectified grid, TPS maps each output pixel to
// a source coordinate in the distorted input. Solved with Gaussian
// elimination (n = 120 -> 123x123 system, milliseconds).

use image::RgbImage;

use crate::InferError;

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

fn u_kernel(r2: f64) -> f64 {
    if r2 < 1e-12 {
        0.0
    } else {
        r2 * r2.ln()
    }
}

fn solve_linear(a: &mut [Vec<f64>], b: &mut [f64]) -> Result<Vec<f64>, InferError> {
    let n = a.len();
    if b.len() != n || a.iter().any(|r| r.len() != n) {
        return Err(InferError::BadParam("tps: non-square system".into()));
    }
    for col in 0..n {
        let mut piv = col;
        for row in (col + 1)..n {
            if a[row][col].abs() > a[piv][col].abs() {
                piv = row;
            }
        }
        if a[piv][col].abs() < 1e-10 {
            return Err(InferError::BadParam("tps: singular system".into()));
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for row in (col + 1)..n {
            let f = a[row][col] / a[col][col];
            for k in col..n {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut s = b[row];
        for k in (row + 1)..n {
            s -= a[row][k] * x[k];
        }
        x[row] = s / a[row][row];
    }
    Ok(x)
}

/// Solve TPS mapping src -> dst (normalized coords).
/// Returns (weights[n x 2], affine[2 x 3]).
pub fn tps_solve(src: &[(f32, f32)], dst: &[(f32, f32)]) -> Result<(Vec<(f64, f64)>, [[f64; 3]; 2]), InferError> {
    let n = src.len();
    if n == 0 || n != dst.len() {
        return Err(InferError::BadParam("tps: src/dst length mismatch".into()));
    }
    let m = n + 3;
    let mut mat = vec![vec![0.0; m]; m];
    for i in 0..n {
        for j in 0..n {
            let dx = src[i].0 as f64 - src[j].0 as f64;
            let dy = src[i].1 as f64 - src[j].1 as f64;
            mat[i][j] = u_kernel(dx * dx + dy * dy);
        }
        mat[i][n] = 1.0;
        mat[i][n + 1] = src[i].0 as f64;
        mat[i][n + 2] = src[i].1 as f64;
        mat[n][i] = 1.0;
        mat[n + 1][i] = src[i].0 as f64;
        mat[n + 2][i] = src[i].1 as f64;
    }
    let mut bx = vec![0.0; m];
    let mut by = vec![0.0; m];
    for i in 0..n {
        bx[i] = dst[i].0 as f64;
        by[i] = dst[i].1 as f64;
    }
    let mut ax = mat.clone();
    let wx = solve_linear(&mut ax, &mut bx)?;
    let mut ay = mat;
    let wy = solve_linear(&mut ay, &mut by)?;
    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        w.push((wx[i], wy[i]));
    }
    Ok((w, [[wx[n], wx[n + 1], wx[n + 2]], [wy[n], wy[n + 1], wy[n + 2]]]))
}

/// Warp `img` with an explicit src -> dst control-point mapping.
pub fn tps_warp(img: &RgbImage, src: &[(f32, f32)], dst: &[(f32, f32)]) -> Result<RgbImage, InferError> {
    let (w, aff) = tps_solve(src, dst)?;
    let (iw, ih) = (img.width() as f32, img.height() as f32);
    let map = |x: u32, y: u32| {
        let nx = x as f64 / (iw - 1.0) as f64 * 2.0 - 1.0;
        let ny = y as f64 / (ih - 1.0) as f64 * 2.0 - 1.0;
        let mut sx = aff[0][0] + aff[0][1] * nx + aff[0][2] * ny;
        let mut sy = aff[1][0] + aff[1][1] * nx + aff[1][2] * ny;
        for (k, s) in src.iter().enumerate() {
            let dx = nx - s.0 as f64;
            let dy = ny - s.1 as f64;
            let u = u_kernel(dx * dx + dy * dy);
            sx += w[k].0 * u;
            sy += w[k].1 * u;
        }
        (((sx + 1.0) / 2.0 * (iw - 1.0) as f64) as f32, ((sy + 1.0) / 2.0 * (ih - 1.0) as f64) as f32)
    };
    Ok(crate::sampler::remap(img, map))
}

/// RP-TPS style warp: base grid + per-point deltas (both normalized).
pub fn tps_warp_with_deltas(
    img: &RgbImage,
    grid_h: usize,
    grid_w: usize,
    deltas: &[(f32, f32)],
) -> Result<RgbImage, InferError> {
    let base = base_points(grid_h, grid_w);
    if deltas.len() != base.len() {
        return Err(InferError::BadParam(format!(
            "tps: need {} deltas for {grid_h}x{grid_w} grid, got {}",
            base.len(),
            deltas.len()
        )));
    }
    let dst: Vec<(f32, f32)> = base.iter().zip(deltas).map(|(b, d)| (b.0 + d.0, b.1 + d.1)).collect();
    tps_warp(img, &base, &dst)
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

    #[test]
    fn identity_warp_is_near_lossless() {
        let mut img = RgbImage::new(40, 32);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = image::Rgb([(x * 6) as u8, (y * 8) as u8, 33]);
        }
        let base = base_points(4, 5);
        let out = tps_warp(&img, &base, &base).unwrap();
        assert_eq!(out.dimensions(), img.dimensions());
        let mut diff = 0u64;
        for (a, b) in out.pixels().zip(img.pixels()) {
            for c in 0..3 {
                diff += (a.0[c] as i32 - b.0[c] as i32).abs() as u64;
            }
        }
        let mean = diff as f64 / (40.0 * 32.0 * 3.0);
        assert!(mean < 3.0, "mean abs diff {mean}");
    }

    #[test]
    fn small_shift_moves_content() {
        let mut img = RgbImage::new(40, 32);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = image::Rgb([(x * 6) as u8, (y * 8) as u8, 33]);
        }
        let base = base_points(4, 5);
        let deltas = vec![(0.05, 0.0); base.len()];
        let out = tps_warp_with_deltas(&img, 4, 5, &deltas).unwrap();
        assert_ne!(out.get_pixel(20, 16), img.get_pixel(20, 16));
    }

    #[test]
    fn bad_delta_count_errors() {
        let img = RgbImage::new(16, 16);
        assert!(tps_warp_with_deltas(&img, 4, 5, &[(0.0, 0.0); 3]).is_err());
    }
}
