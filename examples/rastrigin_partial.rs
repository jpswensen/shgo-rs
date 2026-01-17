use shgo_rs::shgo;

fn main() {
    // Rastrigin via closure with a fixed parameter `a`
    let rastrigin = |x: &[f64], a: f64| {
        let n = x.len() as f64;
        a * n + x.iter().map(|&xi| xi*xi - a*(2.0*std::f64::consts::PI*xi).cos()).sum::<f64>()
    };
    let rastrigin_partial = move |x: &[f64]| rastrigin(x, 10.0);

    let bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)];
    let result = shgo(rastrigin_partial, &bounds, None, None, None::<fn(&[f64])>, None, None, None, None, Some(1));
    println!("Rastrigin (partial) SHGO result: {}", result);
}
