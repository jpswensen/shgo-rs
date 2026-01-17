use std::hint;
use shgo_rs::shgo;

fn main() {
    let rastrigin_fake_long = |x: &[f64], a: f64| {
        let n = x.len() as f64;
        let mut dummy = 0.0;
        let iters = rand::random::<f64>() * 19_000_000.0 + 11_000_000.0;
        for _ in 0..iters as i64 {
            dummy += x.iter().fold(0.0, |acc, &xi| acc + (xi * xi).sin() + (xi).cos());
        }
        let dummy = hint::black_box(dummy);
        a * n + x.iter().map(|&xi| xi*xi - a*(2.0*std::f64::consts::PI*xi).cos()).sum::<f64>() + dummy * 1E-20
    };

    let bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)];
    let result = shgo(move |x| rastrigin_fake_long(x, 10.0), &bounds, None, None, None::<fn(&[f64])>, None, None, None, None, Some(15));
    println!("Rastrigin (fake long) SHGO result: {}", result);
}
