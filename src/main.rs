#![allow(dead_code)] // Until I have this in a lib, let the dead code be OK to prevent warnings

use std::{env, hint};
use std::fmt;
use std::sync::Arc;
use std::collections::HashMap;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, IntoPyDict};


#[derive(Debug, Clone)]
struct OptimizeResult {
    // Primary solution
    x: Vec<f64>,
    fun: Option<f64>,
    success: Option<bool>,
    status: Option<i64>,
    message: Option<String>,
    nfev: Option<i64>,
    nit: Option<i64>,

    // SHGO-specific additional outputs
    // Values of the objective at local minima
    funl: Option<Vec<f64>>,        // shape (k,)
    // Coordinates of local minima
    xl: Option<Vec<Vec<f64>>>,     // shape (k, d)
    // Local optimizer evaluation counts (if provided by SciPy)
    nlfev: Option<i64>,
    nljev: Option<i64>,
    nlhev: Option<i64>,
}

impl OptimizeResult {
    fn write_pretty(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Helpers
        fn fmt_f(v: f64) -> String { format!("{:.6e}", v) }
        fn fmt_vec(v: &Vec<f64>) -> String {
            if v.is_empty() { return "[]".to_string(); }
            let body = v.iter().map(|&x| fmt_f(x)).collect::<Vec<_>>().join(" ");
            format!("[{}]", body)
        }
        fn fmt_mat(m: &Vec<Vec<f64>>, indent: usize) -> String {
            if m.is_empty() { return "[]".to_string(); }
            if m.len() == 1 {
                return format!("[{}]", fmt_vec(&m[0]).trim_start_matches('[').trim_end_matches(']').to_string());
            }
            let pad = " ".repeat(indent);
            let rows = m.iter().map(|row| fmt_vec(row)).collect::<Vec<_>>();
            format!("[\n{}{}\n{}]", pad, rows.join(&format!("\n{}", pad)), " ")
        }

        // Print in SciPy-like order and alignment
        if let Some(s) = self.success { writeln!(f, "success: {}", if s { "True" } else { "False" })?; }
        if let Some(fun) = self.fun { writeln!(f, "     fun: {}", fmt_f(fun))?; }
        if let Some(ref funl) = self.funl { writeln!(f, "    funl: {}", fmt_vec(funl))?; }
        writeln!(f, "       x: {}", fmt_vec(&self.x))?;
        if let Some(ref xl) = self.xl { writeln!(f, "      xl: {}", fmt_mat(xl, 9))?; }
        if let Some(nit) = self.nit { writeln!(f, "     nit: {}", nit)?; }
        if let Some(nfev) = self.nfev { writeln!(f, "    nfev: {}", nfev)?; }
        if let Some(nlfev) = self.nlfev { writeln!(f, "   nlfev: {}", nlfev)?; }
        if let Some(nljev) = self.nljev { writeln!(f, "   nljev: {}", nljev)?; }
        if let Some(nlhev) = self.nlhev { writeln!(f, "   nlhev: {}", nlhev)?; }
        if let Some(status) = self.status { writeln!(f, "   status: {}", status)?; }
        if let Some(ref msg) = self.message { writeln!(f, "  message: {}", msg)?; }
        Ok(())
    }

    pub fn to_pretty_string(&self) -> String {
        struct Buf(String);
        impl fmt::Write for Buf {
            fn write_str(&mut self, s: &str) -> fmt::Result { self.0.push_str(s); Ok(()) }
        }
        let mut s = String::new();
        // Use a temporary Formatter-like wrapper via write! macro into a string
        // Simpler: just build via the same helpers
        // Reuse Display implementation
        use fmt::Write as _;
        if let Some(succ) = self.success { let _ = write!(s, "success: {}\n", if succ { "True" } else { "False" }); }
        if let Some(fun) = self.fun { let _ = write!(s, "     fun: {:.6e}\n", fun); }
        if let Some(ref funl) = self.funl {
            let body = funl.iter().map(|&x| format!("{:.6e}", x)).collect::<Vec<_>>().join(" ");
            let _ = write!(s, "    funl: [{}]\n", body);
        }
        let xv = self.x.iter().map(|&x| format!("{:.6e}", x)).collect::<Vec<_>>().join(" ");
        let _ = write!(s, "       x: [{}]\n", xv);
        if let Some(ref xl) = self.xl {
            if xl.is_empty() {
                let _ = write!(s, "      xl: []\n");
            } else if xl.len() == 1 {
                let row = xl[0].iter().map(|&x| format!("{:.6e}", x)).collect::<Vec<_>>().join(" ");
                let _ = write!(s, "      xl: [[{}]]\n", row);
            } else {
                let rows = xl.iter().map(|r| r.iter().map(|&x| format!("{:.6e}", x)).collect::<Vec<_>>().join(" ")).collect::<Vec<_>>();
                let _ = write!(s, "      xl: [\n            {}\n          ]\n", rows.join("\n            "));
            }
        }
        if let Some(nit) = self.nit { let _ = write!(s, "     nit: {}\n", nit); }
        if let Some(nfev) = self.nfev { let _ = write!(s, "    nfev: {}\n", nfev); }
        if let Some(nlfev) = self.nlfev { let _ = write!(s, "   nlfev: {}\n", nlfev); }
        if let Some(nljev) = self.nljev { let _ = write!(s, "   nljev: {}\n", nljev); }
        if let Some(nlhev) = self.nlhev { let _ = write!(s, "   nlhev: {}\n", nlhev); }
        if let Some(status) = self.status { let _ = write!(s, "   status: {}\n", status); }
        if let Some(ref msg) = self.message { let _ = write!(s, "  message: {}\n", msg); }
        s
    }
}

impl fmt::Display for OptimizeResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.write_pretty(f)
    }
}

fn try_get_attr<'py, T>(obj: &Bound<'py, PyAny>, name: &str) -> Option<T>
where
    T: pyo3::FromPyObject<'py>,
{
    match obj.getattr(name) {
        Ok(val) => val.extract::<T>().ok(),
        Err(_) => None,
    }
}

fn extract_optimize_result<'py>(
    _py: Python<'py>,
    result: Bound<'py, PyAny>,
    ) -> PyResult<OptimizeResult> {
    // Small helpers to extract numpy arrays if present
    fn try_get_array1<'py>(obj: &Bound<'py, PyAny>, name: &str) -> Option<Vec<f64>> {
        match obj.getattr(name) {
            Ok(val) => {
                let arr: PyReadonlyArray1<f64> = val.extract().ok()?;
                let slice = arr.as_slice().ok()?;
                Some(slice.to_vec())
            }
            Err(_) => None,
        }
    }

    fn try_get_array2<'py>(obj: &Bound<'py, PyAny>, name: &str) -> Option<Vec<Vec<f64>>> {
        match obj.getattr(name) {
            Ok(val) => {
                let arr: PyReadonlyArray2<f64> = val.extract().ok()?;
                let view = arr.as_array();
                let rows: Vec<Vec<f64>> = view
                    .outer_iter()
                    .map(|row| row.to_vec())
                    .collect();
                Some(rows)
            }
            Err(_) => None,
        }
    }

    // x (required)
    let x_obj = result.getattr("x")?;
    let x_arr: PyReadonlyArray1<f64> = x_obj.extract()?;
    let x_vec = x_arr.as_slice()?.to_vec();

    // Optional fields
    let fun: Option<f64> = try_get_attr(&result, "fun");
    let success: Option<bool> = try_get_attr(&result, "success");
    let status: Option<i64> = try_get_attr(&result, "status");
    let message: Option<String> = try_get_attr(&result, "message");
    let nfev: Option<i64> = try_get_attr(&result, "nfev");
    // SciPy names iterations as 'nit' for many optimizers; shgo also exposes 'nit'
    let nit: Option<i64> = try_get_attr(&result, "nit");

    // SHGO-specific fields (present in SciPy's shgo OptimizeResult)
    let funl: Option<Vec<f64>> = try_get_array1(&result, "funl");
    let xl: Option<Vec<Vec<f64>>> = try_get_array2(&result, "xl");
    let nlfev: Option<i64> = try_get_attr(&result, "nlfev");
    let nljev: Option<i64> = try_get_attr(&result, "nljev");
    let nlhev: Option<i64> = try_get_attr(&result, "nlhev");

    Ok(OptimizeResult {
        x: x_vec,
        fun,
        success,
        status,
        message,
        nfev,
        nit,
        funl,
        xl,
        nlfev,
        nljev,
        nlhev,
    })
}


// Using scipy as a template, implement the rosen function found in scipy.optimize
fn rosen(x: &[f64]) -> f64 {
    // sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
    x.windows(2).map(|w| {
        let xi = w[0];
        let xi1 = w[1];
        100.0 * (xi1 - xi * xi).powi(2) + (1.0 - xi).powi(2)
    }).sum()
}

fn rastrigin(x: &[f64], a:f64) -> f64 {
    let n = x.len() as f64;
    let rv = a * n + x.iter().map(|&xi| xi*xi - a*(2.0*std::f64::consts::PI*xi).cos()).sum::<f64>();

    rv
}

fn rastrigin_fake_long(x: &[f64], a:f64) -> f64 {
    let n = x.len() as f64;

    // Dummy computation: Loop many times over a trig-heavy operation depending on x
    // to simulate expense (adjust loop iterations for desired "length").
    // This depends on runtime input, so it can't be precomputed or folded away.
    let mut dummy = 0.0;

    // get a random number between 1M and 10M
    let iters = rand::random::<f64>() * 19_000_000.0 + 11_000_000.0;
    for _ in 0..iters as i64 { // Example: 100k iterations; tune as needed
        dummy += x.iter().fold(0.0, |acc, &xi| acc + (xi * xi).sin() + (xi).cos());
    }

    // Optional: Pass through black_box for maximal pessimism (prevents any assumptions about dummy)
    dummy = hint::black_box(dummy);

    let rv = a * n + x.iter().map(|&xi| xi*xi - a*(2.0*std::f64::consts::PI*xi).cos()).sum::<f64>();

    rv + dummy * 1E-20
}

#[pyclass] // must be Sync in free-threaded builds
pub struct Objective {
    func: Arc<dyn Fn(&[f64]) -> f64 + Send + Sync + 'static>,
}

#[pymethods]
impl Objective {
    fn __call__(&self, py: Python<'_>, x: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        // Borrow while attached to the interpreter, then COPY so the compute
        // closure holds no Python refs.
        let v: Vec<f64> = x.as_slice()?.to_vec();

        // Detach from the interpreter while running heavy Rust code.
        let f = self.func.clone();
        let val = py.allow_threads(|| (f)(&v));
        Ok(val)
    }
}

#[pyclass]
struct ConstraintFunction {
    func: Arc<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync + 'static>,
}

#[pymethods]
impl ConstraintFunction {
    fn __call__(&self, py: Python<'_>, x: PyReadonlyArray1<'_, f64>) -> PyResult<Py<PyAny>> {
        let v: Vec<f64> = x.as_slice()?.to_vec();
        let f = self.func.clone();
        let out: Vec<f64> = py.allow_threads(|| (f)(&v));
        // Return as a NumPy array via the numpy crate (avoids PyList Result issues)
    let arr = PyArray1::<f64>::from_vec(py, out);
    // Convert Bound<PyArray> -> Bound<PyAny> -> Py<PyAny>
    Ok(arr.into_any().unbind())
    }
}

#[derive(Debug, Clone)]
pub enum ConstraintType { Eq, Ineq }

impl ConstraintType {
    fn as_str(&self) -> &'static str { match self { ConstraintType::Eq => "eq", ConstraintType::Ineq => "ineq" } }
}

#[derive(Clone)]
pub enum ConstraintSpec {
    // LinearConstraint: A x between lb and ub
    Linear {
        a: Vec<Vec<f64>>, // shape (m, n)
        lb: Vec<f64>,     // shape (m,)
        ub: Vec<f64>,     // shape (m,)
        keep_feasible: Option<bool>,
    },
    // NonlinearConstraint with bounds: fun(x) vector -> between lb and ub
    NonlinearBounds {
        fun: Arc<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync + 'static>,
        lb: Vec<f64>,
        ub: Vec<f64>,
    },
    // Dict-style scalar constraint: {'type': 'eq'|'ineq', 'fun': callable}
    Dict {
        ctype: ConstraintType,
        fun: Arc<dyn Fn(&[f64]) -> f64 + Send + Sync + 'static>,
    },
}

// Concrete structs for each constraint kind, plus a combined container
#[derive(Clone)]
struct LinearConstr {
    a: Vec<Vec<f64>>, // (m, n)
    lb: Vec<f64>,     // (m,)
    ub: Vec<f64>,     // (m,)
    keep_feasible: Option<bool>,
}

#[derive(Clone)]
struct NonlinearConstrBounds {
    fun: Arc<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync + 'static>,
    lb: Vec<f64>,
    ub: Vec<f64>,
}

#[derive(Clone)]
struct DictConstr {
    ctype: ConstraintType,
    fun: Arc<dyn Fn(&[f64]) -> f64 + Send + Sync + 'static>,
}

#[derive(Clone, Default)]
struct CombinedConstraints {
    linear: Vec<LinearConstr>,
    nonlinear: Vec<NonlinearConstrBounds>,
    dict: Vec<DictConstr>,
}

impl From<Vec<ConstraintSpec>> for CombinedConstraints {
    fn from(specs: Vec<ConstraintSpec>) -> Self {
        let mut cc = CombinedConstraints::default();
        for s in specs {
            match s {
                ConstraintSpec::Linear { a, lb, ub, keep_feasible } => cc.linear.push(LinearConstr { a, lb, ub, keep_feasible }),
                ConstraintSpec::NonlinearBounds { fun, lb, ub } => cc.nonlinear.push(NonlinearConstrBounds { fun, lb, ub }),
                ConstraintSpec::Dict { ctype, fun } => cc.dict.push(DictConstr { ctype, fun }),
            }
        }
        cc
    }
}

impl CombinedConstraints {
    fn to_py_list<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let mut items: Vec<Bound<'py, PyAny>> = Vec::with_capacity(
            self.linear.len() + self.nonlinear.len() + self.dict.len(),
        );

        // Import optimize once
        let optimize = PyModule::import(py, "scipy.optimize").unwrap();

        // Linear constraints (auto-split rows that are equalities vs inequalities)
        if !self.linear.is_empty() {
            let constr_cls = optimize.getattr("LinearConstraint").unwrap();
            for lc in &self.linear {
                let tol = 0.0_f64; // exact compare is fine for our constructed values; adjust if needed
                let mut a_eq: Vec<Vec<f64>> = Vec::new();
                let mut lb_eq: Vec<f64> = Vec::new();
                let mut ub_eq: Vec<f64> = Vec::new();
                let mut a_in: Vec<Vec<f64>> = Vec::new();
                let mut lb_in: Vec<f64> = Vec::new();
                let mut ub_in: Vec<f64> = Vec::new();

                for (i, row) in lc.a.iter().enumerate() {
                    if i < lc.lb.len() && i < lc.ub.len() && (lc.lb[i] - lc.ub[i]).abs() <= tol {
                        a_eq.push(row.clone());
                        lb_eq.push(lc.lb[i]);
                        ub_eq.push(lc.ub[i]);
                    } else {
                        a_in.push(row.clone());
                        lb_in.push(lc.lb.get(i).copied().unwrap_or(f64::NEG_INFINITY));
                        ub_in.push(lc.ub.get(i).copied().unwrap_or(f64::INFINITY));
                    }
                }

                let kwargs_c = PyDict::new(py);
                if let Some(kf) = lc.keep_feasible { kwargs_c.set_item("keep_feasible", kf).unwrap(); }

                if !a_in.is_empty() {
                    let a_arr = PyArray2::<f64>::from_vec2(py, &a_in).unwrap();
                    let lb_arr = PyArray1::<f64>::from_vec(py, lb_in);
                    let ub_arr = PyArray1::<f64>::from_vec(py, ub_in);
                    let obj_any = constr_cls.call((a_arr, lb_arr, ub_arr), Some(&kwargs_c)).unwrap();
                    items.push(obj_any);
                }
                if !a_eq.is_empty() {
                    let a_arr = PyArray2::<f64>::from_vec2(py, &a_eq).unwrap();
                    let lb_arr = PyArray1::<f64>::from_vec(py, lb_eq.clone());
                    let ub_arr = PyArray1::<f64>::from_vec(py, ub_eq.clone());
                    let obj_any = constr_cls.call((a_arr, lb_arr, ub_arr), Some(&kwargs_c)).unwrap();
                    items.push(obj_any);
                }
            }
        }

        // Nonlinear bounds constraints
        if !self.nonlinear.is_empty() {
            let constr_cls = optimize.getattr("NonlinearConstraint").unwrap();
            for nc in &self.nonlinear {
                let fun_obj = Py::new(py, ConstraintFunction { func: nc.fun.clone() }).unwrap();
                let lb_arr = PyArray1::<f64>::from_vec(py, nc.lb.clone());
                let ub_arr = PyArray1::<f64>::from_vec(py, nc.ub.clone());
                let obj_any = constr_cls.call((fun_obj, lb_arr, ub_arr), None).unwrap();
                items.push(obj_any);
            }
        }

        // Dict constraints
        for dc in &self.dict {
            let dict = PyDict::new(py);
            dict.set_item("type", dc.ctype.as_str()).unwrap();
            let fun_obj = Py::new(py, Objective { func: dc.fun.clone() }).unwrap();
            dict.set_item("fun", fun_obj).unwrap();
            items.push(dict.into_any());
        }

        PyList::new(py, items).unwrap()
    }
}

// Convert a Rust HashMap to a Python dict
fn rust_dict_to_pydict<'py>(
    py: Python<'py>,
    map: &HashMap<String, f64>,
) -> Bound<'py, PyDict> {
    let dict: Bound<'py, PyDict> = PyDict::new(py);
    for (k, v) in map {
        dict.set_item(k, v).unwrap();
    }
    dict
}

// Rust implementation of `MinimizerKwargs` to match the Python API
struct MinimizerKwargs {
    method: Option<String>,
    options: Option<HashMap<String, f64>>,
    // TODO: Add more fields as needed
}

impl MinimizerKwargs {
    
    fn to_pydict<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let dict = PyDict::new(py);
        if let Some(ref m) = self.method {
            dict.set_item("method", m).unwrap();
        }
        if let Some(ref opts) = self.options {
            dict.set_item("options", rust_dict_to_pydict(py, opts)).unwrap();
        }
        dict
    }
}

struct ShgoOptions {
    maxfev: Option<i64>, // Optional: maximum function evaluations
    f_min: Option<f64>, // Optional: minimum function value
    f_tol: Option<f64>, // Optional: function tolerance
    maxiter: Option<i64>, // Optional: maximum iterations
    maxev: Option<i64>, // Optional: maximum evaluations
    maxtime: Option<f64>, // Optional: maximum time allowed
    minhgrd: Option<i64>, // Optional: minimum gradient norm
    symmetry: Option<Vec<usize>>, // Only support explicit list of symmetric variable indices
    minimize_every_iter: Option<bool>, // Optional: whether to minimize at every iteration
    local_iter: Option<i64>, // Optional: number of local iterations per global iteration
    infty_constraints: Option<bool>, // Optional: whether to allow infinite constraints (e.g., unbounded)
    disp: Option<bool>, // Optional: whether to display convergence messages
    
    // Add more as needed
}

impl ShgoOptions {
    fn to_pydict<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let dict = PyDict::new(py);
        if let Some(val) = self.maxfev { dict.set_item("maxfev", val).unwrap(); }
        if let Some(val) = self.f_min { dict.set_item("f_min", val).unwrap(); }
        if let Some(val) = self.f_tol { dict.set_item("f_tol", val).unwrap(); }
        if let Some(val) = self.maxiter { dict.set_item("maxiter", val).unwrap(); }
        if let Some(val) = self.maxev { dict.set_item("maxev", val).unwrap(); }
        if let Some(val) = self.maxtime { dict.set_item("maxtime", val).unwrap(); }
        if let Some(val) = self.minhgrd { dict.set_item("minhgrd", val).unwrap(); }
        if let Some(val) = self.minimize_every_iter { dict.set_item("minimize_every_iter", val).unwrap(); }
        if let Some(val) = self.local_iter { dict.set_item("local_iter", val).unwrap(); }
        if let Some(val) = self.infty_constraints { dict.set_item("infty_constraints", val).unwrap(); }
        if let Some(val) = self.disp { dict.set_item("disp", val).unwrap(); }
        if let Some(ref sym) = self.symmetry {
            let sym_list = PyList::new(py, sym.iter().copied()).unwrap();
            dict.set_item("symmetry", sym_list).unwrap();
        }
        dict
    }
}


// Variant that returns the full OptimizeResult for richer information
fn shgo<F, C>(
    objective: F,
    bounds: &[(f64, f64)],
    n: Option<usize>,
    iters: Option<usize>,
    callback: Option<C>,
    minimizer_kwargs: Option<MinimizerKwargs>,
    options: Option<ShgoOptions>,
    constraints: Option<Vec<ConstraintSpec>>,
    sampling_method: Option<&str>,
    workers: Option<usize>,
) -> OptimizeResult
where
    F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    C: Fn(&[f64]) + Send + Sync + 'static,
{
    // Force single-threading in NumPy/BLAS to avoid conflicts with our threading
    std::env::set_var("OMP_NUM_THREADS", "1");
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");
    std::env::set_var("MKL_NUM_THREADS", "1");
    std::env::set_var("NUMEXPR_NUM_THREADS", "1");

    Python::with_gil(|py| {
        // Build same kwargs as in shgo()
        // let optimize = PyModule::import(py, "scipy.optimize").expect("pip install scipy");
    
        
        // Manually add venv site-packages if VIRTUAL_ENV is set
        if let Ok(venv) = env::var("VIRTUAL_ENV") {
            let sys = py.import("sys").unwrap();
            let version_info = sys.getattr("version_info").unwrap();
            let major: i32 = version_info.getattr("major").unwrap().extract().unwrap();
            let minor: i32 = version_info.getattr("minor").unwrap().extract().unwrap();
            let lib_dir = format!("python{}.{}t", major, minor);
            let site_packages = format!("{}/lib/{}/site-packages", venv, lib_dir);
            let sys_path = sys.getattr("path").unwrap();
            sys_path.call_method1("insert", (0, site_packages)).unwrap();
        }

        // Diagnostic code about the interpreter
        // let sys = py.import("sys").unwrap();
        // let executable: String = sys.getattr("executable").unwrap().extract().unwrap();
        // let prefix: String = sys.getattr("prefix").unwrap().extract().unwrap();
        // let sys_path: Vec<String> = sys.getattr("path").unwrap().extract().unwrap();
        // println!("Python executable: {}", executable);
        // println!("Python prefix: {}", prefix);
        // println!("sys.path:\n{:#?}", sys_path);


        // Safety: Configure multiprocessing to prevent runaway process creation
        let mp = PyModule::import(py, "multiprocessing").unwrap();
        let _ = mp.call_method1("set_start_method", ("spawn", true)); // force=True

        // Ensure that scipy.optimize is installed
        let optimize = PyModule::import(py, "scipy.optimize")
            .expect("pip install scipy");

        let obj = Py::new(py, Objective { func: Arc::new(objective) }).expect("create Objective");
        let py_bounds = PyList::new(py, bounds.iter().map(|(a, b)| (a, b))).unwrap();
        let kwargs: Bound<PyDict> = PyDict::new(py);
        kwargs.set_item("bounds", &py_bounds).unwrap();
        if let Some(method) = sampling_method { kwargs.set_item("sampling_method", method).unwrap(); }
        if let Some(opts) = options { kwargs.set_item("options", &opts.to_pydict(py)).unwrap(); }
        if let Some(min_kwargs) = minimizer_kwargs { kwargs.set_item("minimizer_kwargs", &min_kwargs.to_pydict(py)).unwrap(); }
        if let Some(n) = n { kwargs.set_item("n", n).unwrap(); }
        if let Some(iters) = iters { kwargs.set_item("iters", iters).unwrap(); }
        if let Some(cb) = callback { let cb_obj = Py::new(py, Callback { func: Arc::new(cb) }).unwrap(); kwargs.set_item("callback", cb_obj).unwrap(); }

        // Build constraints if provided (using CombinedConstraints helper)
        if let Some(constraints) = constraints {
            let combined: CombinedConstraints = constraints.into();
            let py_constraints = combined.to_py_list(py);
            kwargs.set_item("constraints", py_constraints).unwrap();
        }

        let py_workers = workers.unwrap_or(1);
        let result_any: Bound<PyAny> = if py_workers > 1 {
            println!("Using {} workers for parallel SHGO", py_workers);
            let futures = PyModule::import(py, "concurrent.futures").unwrap();
            let tpe_cls = futures.getattr("ThreadPoolExecutor").unwrap();
            let tpe = tpe_cls.call((), Some(&[("max_workers", py_workers)].into_py_dict(py).unwrap())).unwrap();
            let map_callable = tpe.getattr("map").unwrap();
            kwargs.set_item("workers", &map_callable).unwrap();
            let res = optimize.getattr("shgo").unwrap().call((obj,), Some(&kwargs)).unwrap();
            let _ = tpe.call_method1("shutdown", (true,));
            res
        } else {
            optimize.getattr("shgo").unwrap().call((obj,), Some(&kwargs)).unwrap()
        };

        extract_optimize_result(py, result_any).expect("extract OptimizeResult")
    })
}


#[pyclass]
struct Callback {
    func: Arc<dyn Fn(&[f64]) + Send + Sync>,
}

#[pymethods]
impl Callback {
    fn __call__(&self, py: Python, x: numpy::PyReadonlyArray1<f64>) -> PyResult<()> {
        let v: Vec<f64> = x.as_slice()?.to_vec();
        py.allow_threads(|| (self.func)(&v));
        Ok(())
    }
}


fn main() {
    // test_basic_rosen();
    // test_rastrigin_partial();
    test_rastrigin_extra_parameters();
    // test_rastrigin_fake_long_partial();
    test_cattle_feed_constraints();
    test_cattle_feed_constraints_structured();
}


fn test_basic_rosen() {
    // Example using the rosen function and default parameters (same as SHGO documentation)
    let bounds = [(0.0,2.0), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0),(0.0,2.0), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0)];
    let result = shgo(rosen, &bounds, 
        None, 
        None, 
        None::<fn(&[f64])>, 
        None, 
        None, 
    None,
        None, 
        Some(1));
    println!("SHGO result for rosen: {}", result);
    
    // Assert that the result.x is very close to [1,1,...]
    for (i, &x) in result.x.iter().enumerate() {
        assert!((x - 1.0).abs() < 1e-5, "Result at index {} is not close to 1: {}", i, x);
    }
    println!("Basic rosen test passed");
}

fn test_rastrigin_partial() {
    // An example of a partial evaluation, since Rust can't do a tuple of extra arguments of different types.
    // So instead, capture these extra parameters through a closure
    let rastrigin_partial = |x: &[f64]| rastrigin(x, 10.0);
    
    let bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)];
    let result = shgo(rastrigin_partial, &bounds, 
        None, 
        None, 
        None::<fn(&[f64])>, 
        None, 
        None, 
    None,
        None, 
        Some(1));
    println!("SHGO result for rastrigin: {}", result);

    // Make sure the result.x is is very close to [0.0,0.0, ...]
    for (i, &x) in result.x.iter().enumerate() {
        assert!((x - 0.0).abs() < 1e-5, "Result at index {} is not close to 0: {}", i, x);
    }
    println!("Rastrigin with closure test passed");
}

fn test_rastrigin_extra_parameters() {
    // An example with rastrigin and non-standard parameters
    let rastrigin_partial = |x: &[f64]| rastrigin(x, 10.0);
    
    let options = ShgoOptions {
        maxfev: None,
        f_min: None,
        f_tol: Some(1e-6),
        maxiter: None,
        maxev: None,
        maxtime: None,
        minhgrd: None,
        symmetry: None,
        minimize_every_iter: Some(true), // Disable serial local-min on every iter to expose parallel evals
        local_iter: None,
        infty_constraints: None,
        disp: None,
    };

    let mut local_opts = std::collections::HashMap::new();
    local_opts.insert("tol".to_string(), 1e-6);
    local_opts.insert("maxiter".to_string(), 200.0);
    let minimizer_kwargs = MinimizerKwargs {
        method: Some("COBYQA".to_string()),
        options: Some(local_opts),
    };

    let my_callback = |x: &[f64]| {
        println!("Callback at x = {:?}", x);
    };

    let d = 3;
    let bounds: Vec<(f64,f64)> = std::iter::repeat((-5.0, 5.0)).take(d).collect();

    let n = 150;
    let iters = 10;
    let result = shgo(
        rastrigin_partial,
        &bounds,
        Some(n), // n: moderate sampling for visible parallel work
        Some(iters),   // iters: small to minimize serial work
        Some(my_callback), // callback
        Some(minimizer_kwargs),
        Some(options),
    None,
        Some("sobol"), // sampling_method
        None,       // workers: start with 4 to see threading effect
    );
    println!("SHGO result for rastrigin with non-standard params: {}", result);

    // Make sure the result.x is is very close to [0.0,0.0, ...]
    for (i, &x) in result.x.iter().enumerate() {
        assert!((x - 0.0).abs() < 1e-5, "Result at index {} is not close to 0: {}", i, x);
    }
    println!("Rastrigin with non-standard params test passed");
}

fn test_rastrigin_fake_long_partial() {
    // Give and example of tht fake long rastrigin with many CPUs
    let rastrigin_fake_long_partial = |x: &[f64]| rastrigin_fake_long(x, 10.0);
    
    let bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)];
    let result = shgo(rastrigin_fake_long_partial, &bounds, 
        None, 
        None, 
        None::<fn(&[f64])>, 
        None, 
        None, 
    None,
        None, 
        Some(15));
    println!("SHGO result for fake long rastrigin: {}", result);

    // Make sure the result.x is is very close to [0.0,0.0, ...]
    for (i, &x) in result.x.iter().enumerate() {
        assert!((x - 0.0).abs() < 1e-5, "Result at index {} is not close to 0: {}", i, x);
    }
    println!("Long-running Rastrigin test passed");
}

fn test_cattle_feed_constraints() {
    // SHGO documentation example: cattle-feed linear objective with nonlinear and equality constraints
    let f = |x: &[f64]| 24.55*x[0] + 26.75*x[1] + 39.0*x[2] + 40.50*x[3];

    let g1 = |x: &[f64]| 2.3*x[0] + 5.6*x[1] + 11.1*x[2] + 1.3*x[3] - 5.0; // >= 0
    let g2 = |x: &[f64]| {
        let q = 0.28*x[0]*x[0] + 0.19*x[1]*x[1] + 20.5*x[2]*x[2] + 0.62*x[3]*x[3];
        (12.0*x[0] + 11.9*x[1] + 41.8*x[2] + 52.1*x[3] - 21.0) - 1.645 * q.sqrt()
    }; // >= 0
    let h1 = |x: &[f64]| x[0] + x[1] + x[2] + x[3] - 1.0; // == 0

    let constraints = vec![
        ConstraintSpec::Dict { ctype: ConstraintType::Ineq, fun: Arc::new(g1) },
        ConstraintSpec::Dict { ctype: ConstraintType::Ineq, fun: Arc::new(g2) },
        ConstraintSpec::Dict { ctype: ConstraintType::Eq,   fun: Arc::new(h1) },
    ];

    let bounds: Vec<(f64,f64)> = std::iter::repeat((0.0, 1.0)).take(4).collect();

    let result = shgo(
        f,
        &bounds,
        Some(150), // n=150 as in docs
        None,
        None::<fn(&[f64])>,
        None,
        None,
        Some(constraints),
        None,
        Some(1),
    );

    println!("Cattle-feed SHGO result:\n{}", result);

    // Basic checks: bounds, feasibility, and equality approximately satisfied
    let x = &result.x;
    for (i,&xi) in x.iter().enumerate() {
        assert!(xi >= -1e-10 && xi <= 1.0 + 1e-10, "x[{}] out of bounds: {}", i, xi);
    }
    let g1v = g1(x);
    let g2v = g2(x);
    let h1v = h1(x);
    assert!(g1v >= -1e-6, "g1 violated: {}", g1v);
    assert!(g2v >= -1e-6, "g2 violated: {}", g2v);
    assert!(h1v.abs() <= 1e-6, "h1 not equal to 0: {}", h1v);
    if let Some(s) = result.success { assert!(s, "Optimization did not report success"); }

    println!("Cattle-feed constraints test passed");
}

fn test_cattle_feed_constraints_structured() {
    // Same cattle-feed problem. Keep the same constraint TYPES as the dict test
    // (g1: ineq, g2: ineq, h1: eq) but mix representations:
    // - g1 as LinearConstraint (ineq)
    // - g2 as NonlinearConstraint (ineq)
    // - h1 as dict (eq)
    let f = |x: &[f64]| 24.55*x[0] + 26.75*x[1] + 39.0*x[2] + 40.50*x[3];

    // g1: 2.3x0 + 5.6x1 + 11.1x2 + 1.3x3 >= 5  (linear inequality)
    let a = vec![
        vec![2.3, 5.6, 11.1, 1.3],
    ];
    let lb = vec![5.0];
    let ub = vec![f64::INFINITY];

    // g2 as nonlinear inequality: g2(x) >= 0
    let g2 = |x: &[f64]| {
        let q = 0.28*x[0]*x[0] + 0.19*x[1]*x[1] + 20.5*x[2]*x[2] + 0.62*x[3]*x[3];
        (12.0*x[0] + 11.9*x[1] + 41.8*x[2] + 52.1*x[3] - 21.0) - 1.645 * q.sqrt()
    };
    let g2_vec = move |x: &[f64]| vec![g2(x)];
    // h1 as equality via dict to test mixing methods
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

    // Validate feasibility
    let x = &result.x;
    for (i,&xi) in x.iter().enumerate() {
        assert!(xi >= -1e-10 && xi <= 1.0 + 1e-10, "x[{}] out of bounds: {}", i, xi);
    }
    // Check linear constraints
    let g1v = 2.3*x[0] + 5.6*x[1] + 11.1*x[2] + 1.3*x[3] - 5.0;
    let h1v = x[0] + x[1] + x[2] + x[3] - 1.0;
    let g2v = {
        let q = 0.28*x[0]*x[0] + 0.19*x[1]*x[1] + 20.5*x[2]*x[2] + 0.62*x[3]*x[3];
        (12.0*x[0] + 11.9*x[1] + 41.8*x[2] + 52.1*x[3] - 21.0) - 1.645 * q.sqrt()
    };
    assert!(g1v >= -1e-6, "g1 violated: {}", g1v);
    assert!(g2v >= -1e-6, "g2 violated: {}", g2v);
    assert!(h1v.abs() <= 1e-6, "h1 not equal to 0: {}", h1v);
    if let Some(s) = result.success { assert!(s, "Optimization did not report success"); }

    println!("Cattle-feed structured constraints test passed");
}
