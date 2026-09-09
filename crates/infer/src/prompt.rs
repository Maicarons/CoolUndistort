// Visual prompts for T1..T4 (AGPL-3.0-or-later).
//
// Mirrors coolundistort/model/prompts.py, which bakes prompt PNGs at
// TRAINING time. At inference time the Rust side derives the runtime
// prompt from the task id (+ optional border detection for T2/T3).

/// Runtime prompt kind derived from the task id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptKind {
    /// T1: face mask (P1: from detector; P0: full-frame ones).
    Face,
    /// T2/T3: irregular-boundary mask (auto-detected dark borders).
    Boundary,
    /// T4: all-ones (whole frame matters for rotation).
    FullFrame,
}

pub fn prompt_for_task(task: crate::Task) -> PromptKind {
    match task {
        crate::Task::T1 => PromptKind::Face,
        crate::Task::T2 | crate::Task::T3 => PromptKind::Boundary,
        crate::Task::T4 => PromptKind::FullFrame,
    }
}

/// P0 prompt image: full-frame ones (f32, row-major).
pub fn make_prompt_image(kind: PromptKind, width: u32, height: u32) -> Vec<f32> {
    let _ = kind;
    vec![1.0f32; (width * height) as usize]
}

/// Boundary mask for stitched / rectified wide images: pixels brighter
/// than `threshold` count as valid content (dark padding -> 0).
/// Matches the Python `border_mask` used to bake training prompts.
pub fn border_mask(img: &image::RgbImage, threshold: u8) -> Vec<f32> {
    img.pixels()
        .map(|p| {
            let luma = (p.0[0] as u32 * 299 + p.0[1] as u32 * 587 + p.0[2] as u32 * 114) / 1000;
            if luma > threshold as u32 { 1.0 } else { 0.0 }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_prompt_mapping() {
        assert_eq!(prompt_for_task(crate::Task::T1), PromptKind::Face);
        assert_eq!(prompt_for_task(crate::Task::T2), PromptKind::Boundary);
        assert_eq!(prompt_for_task(crate::Task::T3), PromptKind::Boundary);
        assert_eq!(prompt_for_task(crate::Task::T4), PromptKind::FullFrame);
    }

    #[test]
    fn border_mask_separates_padding() {
        let mut img = image::RgbImage::new(8, 8);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = if x < 2 { image::Rgb([0, 0, 0]) } else { image::Rgb([200, 200, 200]) };
            let _ = y;
        }
        let m = border_mask(&img, 16);
        assert_eq!(m[0], 0.0);
        assert_eq!(m[7], 1.0);
    }
}
