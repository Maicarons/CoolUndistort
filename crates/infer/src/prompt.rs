// Visual prompts for T1..T4 (mirrors coolundistort/model/prompts.py,
// which is used at TRAINING time to bake prompt PNGs; at inference time
// the Rust side regenerates the runtime prompt from the task id).

/// Runtime prompt kind derived from the task id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptKind {
    /// T1: face mask (P1: from detector; P0: full-frame ones).
    Face,
    /// T2/T3: irregular-boundary mask (P1: from border detection).
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
/// P1 replaces Face/Boundary with detector outputs.
pub fn make_prompt_image(kind: PromptKind, width: u32, height: u32) -> Vec<f32> {
    let _ = kind;
    vec![1.0f32; (width * height) as usize]
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
}
