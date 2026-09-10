// CoolUndistort CLI (AGPL-3.0-or-later): file in, file out, pure Rust.
//
//   Single : coolundistort --input in.jpg --output out.png [--mode auto]   (default: blind, bundled weights)
//   Calib  : ... --mode fisheye [--calib calib.json | --lambda F]
//   T4     : ... --task t4 --angle 2.5         (tilt present in the input)
//   TPS    : ... --mode tps --deltas d.json [--grid 10x12]
//   Weights: ... --mode checkpoint --onnx model.onnx
//   Batch  : coolundistort --batch in_dir --out-dir out_dir [same flags]
//   Metrics: add --eval -> writes JSON {psnr_self, ssim_self, lambda_est} instead of image

use std::path::{Path, PathBuf};

use coolundistort_infer::{Calibration, CameraParams, InferError, InferMode, InferParams, Task, undistort_full};
use image::RgbImage;

fn usage() -> ! {
    eprintln!(
        "usage:\n  coolundistort --input <img> --output <img|json> [--task t1|t2|t3|t4] [--mode auto|fisheye|tps|checkpoint]\n                    [--calib calib.json | --lambda F] [--angle DEG] [--deltas d.json --grid HxW] [--onnx m.onnx] [--eval]\n  coolundistort --batch <in_dir> --out-dir <out_dir> [same flags]"
    );
    std::process::exit(2);
}

fn arg_value(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone())
}

fn fail(msg: String) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1);
}

fn parse_grid(s: &str) -> (usize, usize) {
    let (h, w) = s.split_once(['x', 'X']).unwrap_or(("10", "12"));
    (h.parse().unwrap_or(10), w.parse().unwrap_or(12))
}

fn build_params(args: &[String]) -> InferParams {
    let mut p = InferParams::default();
    if let Some(c) = arg_value(args, "--calib") {
        let text = std::fs::read_to_string(&c).unwrap_or_else(|e| fail(format!("cannot read {c}: {e}")));
        p.calib = Calibration::load_json(&text).unwrap_or_else(|e| fail(e.to_string()));
    } else if let Some(l) = arg_value(args, "--lambda") {
        let lambda: f32 = l.parse().unwrap_or_else(|_| usage());
        p.calib = Calibration::division(lambda);
    } else {
        p.calib = CameraParams::default().into();
    }
    p.angle_deg = arg_value(args, "--angle").map(|s| s.parse().unwrap_or_else(|_| usage())).unwrap_or(0.0);
    p.tps_grid = parse_grid(&arg_value(args, "--grid").unwrap_or_else(|| "10x12".into()));
    if let Some(d) = arg_value(args, "--deltas") {
        let text = std::fs::read_to_string(&d).unwrap_or_else(|e| fail(format!("cannot read {d}: {e}")));
        let raw: Vec<[f32; 2]> =
            serde_json::from_str(&text).unwrap_or_else(|e| fail(format!("bad deltas json: {e}")));
        p.tps_deltas = Some(raw.into_iter().map(|a| (a[0], a[1])).collect());
    }
    p.onnx_path = arg_value(args, "--onnx");
    p
}

/// Structural similarity (luma, global) as a cheap self-reference metric.
fn ssim(a: &RgbImage, b: &RgbImage) -> f64 {
    let luma = |p: &image::Rgb<u8>| 0.299 * p.0[0] as f64 + 0.587 * p.0[1] as f64 + 0.114 * p.0[2] as f64;
    let n = (a.width() * a.height()) as f64;
    let (mut mx, mut my) = (0.0, 0.0);
    for (pa, pb) in a.pixels().zip(b.pixels()) {
        mx += luma(pa);
        my += luma(pb);
    }
    mx /= n;
    my /= n;
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for (pa, pb) in a.pixels().zip(b.pixels()) {
        let (dx, dy) = (luma(pa) - mx, luma(pb) - my);
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    sxx /= n;
    syy /= n;
    sxy /= n;
    let (c1, c2) = (6.5025, 58.5225);
    ((2.0 * mx * my + c1) * (2.0 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
}

fn psnr(a: &RgbImage, b: &RgbImage) -> f64 {
    let mut se = 0u64;
    let n = (a.width() * a.height() * 3) as u64;
    for (pa, pb) in a.pixels().zip(b.pixels()) {
        for c in 0..3 {
            let d = pa.0[c] as i32 - pb.0[c] as i32;
            se += (d * d) as u64;
        }
    }
    if se == 0 {
        return 100.0;
    }
    let mse = se as f64 / n as f64;
    20.0 * (255.0 / mse.sqrt()).log10()
}

fn run_one(input: &Path, output: &Path, task: Task, mode: InferMode, params: &InferParams, eval: bool) {
    let img = image::open(input).unwrap_or_else(|e| fail(format!("cannot read {}: {e}", input.display()))).to_rgb8();
    // Auto mode also reports the estimated lambda.
    let (out, lambda_est) = if mode == InferMode::Auto {
        coolundistort_infer::onnx::run_auto(&img, params.onnx_path.as_deref())
            .unwrap_or_else(|e| fail(e.to_string()))
    } else {
        let out = undistort_full(&img, task, mode, params).unwrap_or_else(|e| {
            if let InferError::NeedsWeights(_) = e {
                fail(format!("{e}"));
            }
            fail(e.to_string())
        });
        (out, f32::NAN)
    };
    if eval {
        let m = serde_json::json!({
            "mode": format!("{mode:?}"), "task": format!("{task:?}"),
            "psnr_self": psnr(&img, &out), "ssim_self": ssim(&img, &out),
            "lambda_est": if lambda_est.is_finite() { serde_json::json!(lambda_est) } else { serde_json::Value::Null },
        });
        std::fs::write(output, serde_json::to_string_pretty(&m).unwrap()).unwrap();
    } else {
        out.save(output).unwrap_or_else(|e| fail(format!("cannot write {}: {e}", output.display())));
    }
    if lambda_est.is_finite() {
        println!("wrote {} (lambda_est={:.4})", output.display(), lambda_est);
    } else {
        println!("wrote {}", output.display());
    }
}

fn is_image(p: &Path) -> bool {
    matches!(p.extension().and_then(|e| e.to_str()).map(|e| e.to_lowercase()).as_deref(),
        Some("jpg") | Some("jpeg") | Some("png") | Some("bmp") | Some("tif") | Some("tiff") | Some("webp"))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let task = Task::parse(&arg_value(&args, "--task").unwrap_or_else(|| "t2".into())).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2);
    });
    let mode = InferMode::parse(&arg_value(&args, "--mode").unwrap_or_else(|| "auto".into())).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2);
    });
    let params = build_params(&args);
    let eval = args.iter().any(|a| a == "--eval");

    if let Some(batch) = arg_value(&args, "--batch") {
        let out_dir = PathBuf::from(arg_value(&args, "--out-dir").unwrap_or_else(|| usage()));
        std::fs::create_dir_all(&out_dir).unwrap_or_else(|e| fail(e.to_string()));
        let mut n = 0;
        let mut entries: Vec<PathBuf> = std::fs::read_dir(&batch)
            .unwrap_or_else(|e| fail(e.to_string()))
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| is_image(p))
            .collect();
        entries.sort();
        for input in entries {
            let stem = input.file_stem().unwrap().to_string_lossy();
            let ext = if eval { "json".to_string() } else { input.extension().unwrap().to_string_lossy().into_owned() };
            run_one(&input, &out_dir.join(format!("{stem}.{ext}")), task, mode, &params, eval);
            n += 1;
        }
        println!("done: {n} files");
        return;
    }

    let input = arg_value(&args, "--input").unwrap_or_else(|| usage());
    let output = arg_value(&args, "--output").unwrap_or_else(|| usage());
    run_one(Path::new(&input), Path::new(&output), task, mode, &params, eval);
}
