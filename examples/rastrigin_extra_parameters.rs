use std::collections::HashMap;
use shgo_rs::{shgo, MinimizerKwargs, ShgoOptions};

fn main() {
    let rastrigin = |x: &[f64], a: f64| {
        let n = x.len() as f64;
        a * n + x.iter().map(|&xi| xi*xi - a*(2.0*std::f64::consts::PI*xi).cos()).sum::<f64>()
    };
    let rastrigin_partial = move |x: &[f64]| rastrigin(x, 10.0);

    let options = ShgoOptions {
        maxfev: None,
        f_min: None,
        f_tol: Some(1e-6),
        maxiter: None,
        maxev: None,
        maxtime: None,
        minhgrd: None,
        minimize_every_iter: Some(false),
        local_iter: None,
        infty_constraints: None,
        disp: None,
        symmetry: None,
    };

    let mut local_opts = HashMap::new();
    local_opts.insert("tol".to_string(), 1e-6);
    local_opts.insert("maxiter".to_string(), 200.0);
    let minimizer_kwargs = MinimizerKwargs {
        method: Some("COBYQA".to_string()),
        options: Some(local_opts),
    };

    let d = 3;
    let bounds: Vec<(f64,f64)> = std::iter::repeat((-5.0, 5.0)).take(d).collect();

    let n = 150;
    let iters = 10;
    let result = shgo(
        rastrigin_partial,
        &bounds,
        Some(n),
        Some(iters),
        None::<fn(&[f64])>,
        Some(minimizer_kwargs),
        Some(options),
        None,
        Some("sobol"),
        None,
    );
    println!("Rastrigin (extra params) SHGO result: {}", result);
}
