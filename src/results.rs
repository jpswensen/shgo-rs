use pyo3::prelude::*;
use pyo3::types::PyAny;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use std::fmt;

#[derive(Debug, Clone)]
pub struct OptimizeResult {
    pub x: Vec<f64>,
    pub fun: Option<f64>,
    pub success: Option<bool>,
    pub status: Option<i64>,
    pub message: Option<String>,
    pub nfev: Option<i64>,
    pub nit: Option<i64>,
    pub funl: Option<Vec<f64>>,        // shape (k,)
    pub xl: Option<Vec<Vec<f64>>>,     // shape (k, d)
    pub nlfev: Option<i64>,
    pub nljev: Option<i64>,
    pub nlhev: Option<i64>,
}

impl OptimizeResult {
    fn write_pretty(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

pub fn extract_optimize_result<'py>(
    _py: Python<'py>,
    result: Bound<'py, PyAny>,
) -> PyResult<OptimizeResult> {
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
                let rows: Vec<Vec<f64>> = view.outer_iter().map(|row| row.to_vec()).collect();
                Some(rows)
            }
            Err(_) => None,
        }
    }

    let x_obj = result.getattr("x")?;
    let x_arr: PyReadonlyArray1<f64> = x_obj.extract()?;
    let x_vec = x_arr.as_slice()?.to_vec();

    let fun: Option<f64> = try_get_attr(&result, "fun");
    let success: Option<bool> = try_get_attr(&result, "success");
    let status: Option<i64> = try_get_attr(&result, "status");
    let message: Option<String> = try_get_attr(&result, "message");
    let nfev: Option<i64> = try_get_attr(&result, "nfev");
    let nit: Option<i64> = try_get_attr(&result, "nit");

    let funl: Option<Vec<f64>> = try_get_array1(&result, "funl");
    let xl: Option<Vec<Vec<f64>>> = try_get_array2(&result, "xl");
    let nlfev: Option<i64> = try_get_attr(&result, "nlfev");
    let nljev: Option<i64> = try_get_attr(&result, "nljev");
    let nlhev: Option<i64> = try_get_attr(&result, "nlhev");

    Ok(OptimizeResult { x: x_vec, fun, success, status, message, nfev, nit, funl, xl, nlfev, nljev, nlhev })
}
