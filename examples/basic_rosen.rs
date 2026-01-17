use shgo_rs::shgo;

fn rosen(x: &[f64]) -> f64 {
    x.windows(2)
        .map(|w| {
            let xi = w[0];
            let xi1 = w[1];
            100.0 * (xi1 - xi * xi).powi(2) + (1.0 - xi).powi(2)
        })
        .sum()
}

fn main() {
    let bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)];
    let result = shgo(rosen, &bounds, None, None, None::<fn(&[f64])>, None, None, None, None, Some(1));
    println!("Rosenbrock SHGO result: {}", result);
}
