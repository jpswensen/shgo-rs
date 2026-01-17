use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

pub struct MinimizerKwargs {
    pub method: Option<String>,
    pub options: Option<HashMap<String, f64>>, // extend as needed
}

impl MinimizerKwargs {
    fn rust_dict_to_pydict<'py>(py: Python<'py>, map: &HashMap<String, f64>) -> pyo3::Bound<'py, PyDict> {
        let dict: pyo3::Bound<'py, PyDict> = PyDict::new(py);
        for (k, v) in map { dict.set_item(k, v).unwrap(); }
        dict
    }

    pub fn to_pydict<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, PyDict> {
        let dict = PyDict::new(py);
        if let Some(ref m) = self.method { dict.set_item("method", m).unwrap(); }
        if let Some(ref opts) = self.options { dict.set_item("options", Self::rust_dict_to_pydict(py, opts)).unwrap(); }
        dict
    }
}

pub struct ShgoOptions {
    pub maxfev: Option<i64>,
    pub f_min: Option<f64>,
    pub f_tol: Option<f64>,
    pub maxiter: Option<i64>,
    pub maxev: Option<i64>,
    pub maxtime: Option<f64>,
    pub minhgrd: Option<i64>,
    pub minimize_every_iter: Option<bool>,
    pub local_iter: Option<i64>,
    pub infty_constraints: Option<bool>,
    pub disp: Option<bool>,
    pub symmetry: Option<Vec<usize>>, // list only
}

impl ShgoOptions {
    pub fn to_pydict<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, PyDict> {
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
        if let Some(ref sym) = self.symmetry { let lst = pyo3::types::PyList::new(py, sym.iter().copied()).unwrap(); dict.set_item("symmetry", lst).unwrap(); }
        dict
    }
}
