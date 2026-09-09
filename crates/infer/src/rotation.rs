// Rotation correction for T4 (AGPL-3.0-or-later).
//
// Backward-mapped rotation about the image center. Positive angles are
// clockwise (screen coords, y down). The `angle_deg` passed to
// `undistort_full` is the tilt PRESENT in the input; correction applies
// the inverse.

use image::RgbImage;

/// Rotate `img` by `angle_deg` clockwise, same output frame.
pub fn rotate_image(img: &RgbImage, angle_deg: f32) -> RgbImage {
    let (w, h) = (img.width() as f32, img.height() as f32);
    let cx = w / 2.0;
    let cy = h / 2.0;
    let t = angle_deg.to_radians();
    let (cos, sin) = (t.cos(), t.sin());
    crate::sampler::remap(img, |x, y| {
        let dx = x as f32 - cx;
        let dy = y as f32 - cy;
        (cx + cos * dx - sin * dy, cy + sin * dx + cos * dy)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_rotation_is_identity() {
        let mut img = RgbImage::new(16, 12);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = image::Rgb([(x * 13) as u8, (y * 17) as u8, 7]);
        }
        let out = rotate_image(&img, 0.0);
        assert!(out.pixels().zip(img.pixels()).all(|(a, b)| a == b));
    }

    #[test]
    fn full_turn_returns() {
        let img = RgbImage::new(24, 24);
        let out = rotate_image(&img, 360.0);
        assert_eq!(out.dimensions(), img.dimensions());
    }
}
