// Division-model fisheye undistortion (single-parameter, no calibration files).
//
// Forward (distorted from undistorted, normalized coords, r in [0, ~1]):
//   r_d = r / (1 + lambda * r^2)
// Backward (what we need for warping): invert per-pixel with a few
// Newton iterations on f(r) = r / (1 + l r^2) - r_d = 0.
// Backward mapping keeps the output frame identical to the input frame.

use image::RgbImage;

fn invert_radius(rd: f32, lambda: f32) -> f32 {
    if rd <= 1e-6 {
        return 0.0;
    }
    let mut r = rd;
    for _ in 0..8 {
        let denom = 1.0 + lambda * r * r;
        let f = r / denom - rd;
        let df = (1.0 - lambda * r * r) / (denom * denom);
        if df.abs() < 1e-6 {
            break;
        }
        r -= f / df;
        if r < 0.0 {
            r = 0.0;
            break;
        }
    }
    r
}

fn sample_bilinear(img: &RgbImage, x: f32, y: f32) -> [u8; 3] {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let mut out = [0u32; 3];
    for dy in 0..2 {
        for dx in 0..2 {
            let sx = (x0 + dx).clamp(0, w - 1) as u32;
            let sy = (y0 + dy).clamp(0, h - 1) as u32;
            let wgt = if dx == 0 { 1.0 - fx } else { fx } * if dy == 0 { 1.0 - fy } else { fy };
            let p = img.get_pixel(sx, sy).0;
            for c in 0..3 {
                out[c] += (p[c] as f32 * wgt) as u32;
            }
        }
    }
    [out[0].min(255) as u8, out[1].min(255) as u8, out[2].min(255) as u8]
}

/// Undistort with the division model. lambda ~ 0.2-0.5 for typical wide lenses;
/// lambda = 0 is a no-op (useful for T4 rotation-only inputs).
pub fn undistort_division(img: &RgbImage, lambda: f32) -> RgbImage {
    let (w, h) = (img.width(), img.height());
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let norm = (cx * cx + cy * cy).sqrt();
    let mut out = RgbImage::new(w, h);
    for (x, y, px) in out.enumerate_pixels_mut() {
        let dx = (x as f32 - cx) / norm;
        let dy = (y as f32 - cy) / norm;
        let rd = (dx * dx + dy * dy).sqrt();
        let r = invert_radius(rd, lambda);
        let s = if rd > 1e-6 { r / rd } else { 1.0 };
        *px = image::Rgb(sample_bilinear(img, cx + dx * s * norm, cy + dy * s * norm));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn lambda_zero_is_identity() {
        let mut img = RgbImage::new(16, 12);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = Rgb([(x * 13) as u8, (y * 17) as u8, 7]);
        }
        let out = undistort_division(&img, 0.0);
        assert_eq!(out.dimensions(), img.dimensions());
        for (a, b) in out.pixels().zip(img.pixels()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn output_shape_preserved() {
        let img = RgbImage::new(64, 48);
        let out = undistort_division(&img, 0.35);
        assert_eq!(out.dimensions(), (64, 48));
    }
}
