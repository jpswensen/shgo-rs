use std::sync::Arc;
use shgo_rs::{shgo, ConstraintSpec, ConstraintType};

fn main() {
    let f = |x: &[f64]| 24.55*x[0] + 26.75*x[1] + 39.0*x[2] + 40.50*x[3];
    let a = vec![ vec![2.3, 5.6, 11.1, 1.3], ];
    let lb = vec![5.0];
    let ub = vec![f64::INFINITY];
    let g2 = |x: &[f64]| {
        let q = 0.28*x[0]*x[0] + 0.19*x[1]*x[1] + 20.5*x[2]*x[2] + 0.62*x[3]*x[3];
        (12.0*x[0] + 11.9*x[1] + 41.8*x[2] + 52.1*x[3] - 21.0) - 1.645 * q.sqrt()
    };
    let g2_vec = move |x: &[f64]| vec![g2(x)];
    let h1 = |x: &[f64]| x[0] + x[1] + x[2] + x[3] - 1.0; // == 0

    let constraints = vec![
        ConstraintSpec::Linear { a, lb, ub, keep_feasible: None },
        ConstraintSpec::NonlinearBounds { fun: Arc::new(g2_vec), lb: vec![0.0], ub: vec![f64::INFINITY] },
        ConstraintSpec::Dict { ctype: ConstraintType::Eq, fun: Arc::new(h1) },
    ];

    let bounds: Vec<(f64,f64)> = std::iter::repeat((0.0, 1.0)).take(4).collect();

    let result = shgo(
        f,
        &bounds,
        Some(150),
        None,
        None::<fn(&[f64])>,
        None,
        None,
        Some(constraints),
        None,
        Some(1),
    );

    println!("Cattle-feed (structured) SHGO result:\n{}", result);
}
