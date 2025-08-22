use std::{env, sync::Arc};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyList};

use crate::options::{MinimizerKwargs, ShgoOptions};
use crate::constraints::{ConstraintSpec, CombinedConstraints};
use crate::pybridge::{Objective, Callback};
use crate::results::extract_optimize_result;

pub fn shgo<F, C>(
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
) -> crate::results::OptimizeResult
where
    F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    C: Fn(&[f64]) + Send + Sync + 'static,
{
    std::env::set_var("OMP_NUM_THREADS", "1");
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");
    std::env::set_var("MKL_NUM_THREADS", "1");
    std::env::set_var("NUMEXPR_NUM_THREADS", "1");

    Python::with_gil(|py| {
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

        let mp = PyModule::import(py, "multiprocessing").unwrap();
        let _ = mp.call_method1("set_start_method", ("spawn", true));

        let optimize = PyModule::import(py, "scipy.optimize").expect("pip install scipy");

        let obj = Py::new(py, Objective { func: Arc::new(objective) }).expect("create Objective");
        let py_bounds = PyList::new(py, bounds.iter().map(|(a, b)| (a, b))).unwrap();
        let kwargs: pyo3::Bound<PyDict> = PyDict::new(py);
        kwargs.set_item("bounds", &py_bounds).unwrap();
        if let Some(method) = sampling_method { kwargs.set_item("sampling_method", method).unwrap(); }
        if let Some(opts) = options { kwargs.set_item("options", &opts.to_pydict(py)).unwrap(); }
        if let Some(min_kwargs) = minimizer_kwargs { kwargs.set_item("minimizer_kwargs", &min_kwargs.to_pydict(py)).unwrap(); }
        if let Some(n) = n { kwargs.set_item("n", n).unwrap(); }
        if let Some(iters) = iters { kwargs.set_item("iters", iters).unwrap(); }
        if let Some(cb) = callback { let cb_obj = Py::new(py, Callback { func: Arc::new(cb) }).unwrap(); kwargs.set_item("callback", cb_obj).unwrap(); }
        if let Some(constraints) = constraints { let py_constraints = CombinedConstraints::from(constraints).to_py_list(py); kwargs.set_item("constraints", py_constraints).unwrap(); }

        let py_workers = workers.unwrap_or(1);
        let result_any: pyo3::Bound<pyo3::types::PyAny> = if py_workers > 1 {
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
