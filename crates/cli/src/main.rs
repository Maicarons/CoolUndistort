// CoolUndistort CLI (AGPL-3.0-or-later): file in, file out, pure Rust.
//
//   coolundistort --input in.jpg --output out.png --task t2 --mode fisheye [--lambda 0.35]
//   coolundistort --input in.jpg --output metrics.json --eval   (self-PSNR probe)

use std::path::PathBuf;

use coolundistort_infer::{CameraParams, InferMode, Task, undistort};

fn usage() -> ! {
    eprintln!("usage: coolundistort --input <img> --output <img|json> [--task t1|t2|t3|t4] [--mode fisheye|tps|checkpoint] [--lambda F] [--eval]");
    std::process::exit(2);
}

fn arg_value(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone())
}

fn psnr(a: &image::RgbImage, b: &image::RgbImage) -> f64 {
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = arg_value(&args, "--input").unwrap_or_else(|| usage());
    let output = arg_value(&args, "--output").unwrap_or_else(|| usage());
    let task = Task::parse(&arg_value(&args, "--task").unwrap_or_else(|| "t2".into())).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2);
    });
    let mode = InferMode::parse(&arg_value(&args, "--mode").unwrap_or_else(|| "fisheye".into())).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2);
    });
    let lambda: f32 = arg_value(&args, "--lambda").map(|s| s.parse().unwrap_or_else(|_| usage())).unwrap_or(0.35);

    let img = image::open(&PathBuf::from(&input)).unwrap_or_else(|e| {
        eprintln!("error: cannot read {input}: {e}");
        std::process::exit(1);
    }).to_rgb8();

    let out = undistort(&img, task, mode, CameraParams { lambda }).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });

    if args.iter().any(|a| a == "--eval") {
        let m = serde_json::json!({"mode": format!("{mode:?}"), "psnr_self": psnr(&img, &out)});
        std::fs::write(&output, serde_json::to_string_pretty(&m).unwrap()).unwrap();
        println!("wrote {output}");
        return;
    }
    out.save(&output).unwrap_or_else(|e| {
        eprintln!("error: cannot write {output}: {e}");
        std::process::exit(1);
    });
    println!("wrote {output}");
}
