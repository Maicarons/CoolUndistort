// Division-model fisheye (AGPL-3.0-or-later).
//
// Forward (distorted from undistorted, normalized coords):
//   r_d = r / (1 + lambda * r^2)
// Backward (used for warping): Newton-invert per pixel.

use image::RgbImage;

pub fn invert_radius(rd: f32, lambda: f32) -> f32 {
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

/// Undistort with the division model. lambda ~ 0.2-0.5 for typical wide
/// lenses; lambda = 0 is a no-op (useful for T4 rotation-only inputs).
pub fn undistort_division(img: &RgbImage, lambda: f32) -> RgbImage {
    crate::calibration::Calibration::division(lambda).undistort(img)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lambda_zero_is_identity() {
        let mut img = RgbImage::new(16, 12);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = image::Rgb([(x * 13) as u8, (y * 17) as u8, 7]);
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

    #[test]
    fn positive_lambda_moves_corners() {
        let mut img = RgbImage::new(32, 32);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = image::Rgb([x as u8 * 8, y as u8 * 8, 0]);
        }
        let out = undistort_division(&img, 0.5);
        assert_ne!(out.get_pixel(0, 0), img.get_pixel(0, 0));
        assert_eq!(out.get_pixel(16, 16), img.get_pixel(16, 16));
    }
}
