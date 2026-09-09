// Shared bilinear sampler + backward remap (AGPL-3.0-or-later).
use image::RgbImage;

/// Bilinear sample with edge clamping.
pub fn sample_bilinear(img: &RgbImage, x: f32, y: f32) -> [u8; 3] {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let fx = (x - x0 as f32).clamp(0.0, 1.0);
    let fy = (y - y0 as f32).clamp(0.0, 1.0);
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

/// Backward remap: out(x,y) = sample(img, map(x,y)).
pub fn remap(img: &RgbImage, map: impl Fn(u32, u32) -> (f32, f32)) -> RgbImage {
    let (w, h) = (img.width(), img.height());
    let mut out = RgbImage::new(w, h);
    for (x, y, px) in out.enumerate_pixels_mut() {
        let (sx, sy) = map(x, y);
        *px = image::Rgb(sample_bilinear(img, sx, sy));
    }
    out
}
