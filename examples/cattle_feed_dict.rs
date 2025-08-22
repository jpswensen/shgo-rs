use shgo_rs::{shgo, ConstraintSpec, ConstraintType};

fn main() {
    let f = |x: &[f64]| 24.55*x[0] + 26.75*x[1] + 39.0*x[2] + 40.50*x[3];
    let g1 = |x: &[f64]| 2.3*x[0] + 5.6*x[1] + 11.1*x[2] + 1.3*x[3] - 5.0; // >= 0
    let g2 = |x: &[f64]| {
        let q = 0.28*x[0]*x[0] + 0.19*x[1]*x[1] + 20.5*x[2]*x[2] + 0.62*x[3]*x[3];
        (12.0*x[0] + 11.9*x[1] + 41.8*x[2] + 52.1*x[3] - 21.0) - 1.645 * q.sqrt()
    }; // >= 0
    let h1 = |x: &[f64]| x[0] + x[1] + x[2] + x[3] - 1.0; // == 0

    let constraints = vec![
        ConstraintSpec::Dict { ctype: ConstraintType::Ineq, fun: std::sync::Arc::new(g1) },
        ConstraintSpec::Dict { ctype: ConstraintType::Ineq, fun: std::sync::Arc::new(g2) },
        ConstraintSpec::Dict { ctype: ConstraintType::Eq,   fun: std::sync::Arc::new(h1) },
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

    println!("Cattle-feed (dict) SHGO result:\n{}", result);
}
