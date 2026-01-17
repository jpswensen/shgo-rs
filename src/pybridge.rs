use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::sync::Arc;

#[pyclass]
pub struct Objective {
    pub(crate) func: Arc<dyn Fn(&[f64]) -> f64 + Send + Sync + 'static>,
}

#[pymethods]
impl Objective {
    fn __call__(&self, py: Python<'_>, x: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let v: Vec<f64> = x.as_slice()?.to_vec();
        let f = self.func.clone();
        let val = py.detach(|| (f)(&v));
        Ok(val)
    }
}

#[pyclass]
pub struct ConstraintFunction {
    pub(crate) func: Arc<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync + 'static>,
}

#[pymethods]
impl ConstraintFunction {
    fn __call__(&self, py: Python<'_>, x: PyReadonlyArray1<'_, f64>) -> PyResult<Py<PyAny>> {
        let v: Vec<f64> = x.as_slice()?.to_vec();
        let f = self.func.clone();
        let out: Vec<f64> = py.detach(|| (f)(&v));
        let arr = PyArray1::<f64>::from_vec(py, out);
        Ok(arr.into_any().unbind())
    }
}

#[pyclass]
pub struct Callback { pub(crate) func: Arc<dyn Fn(&[f64]) + Send + Sync> }

#[pymethods]
impl Callback {
    fn __call__(&self, py: Python, x: numpy::PyReadonlyArray1<f64>) -> PyResult<()> {
        let v: Vec<f64> = x.as_slice()?.to_vec();
        py.detach(|| (self.func)(&v));
        Ok(())
    }
}
