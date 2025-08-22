use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use std::sync::Arc;

use crate::pybridge::{Objective, ConstraintFunction};

#[derive(Debug, Clone)]
pub enum ConstraintType { Eq, Ineq }

impl ConstraintType {
    pub fn as_str(&self) -> &'static str { match self { ConstraintType::Eq => "eq", ConstraintType::Ineq => "ineq" } }
}

#[derive(Clone)]
pub enum ConstraintSpec {
    Linear { a: Vec<Vec<f64>>, lb: Vec<f64>, ub: Vec<f64>, keep_feasible: Option<bool> },
    NonlinearBounds { fun: Arc<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync + 'static>, lb: Vec<f64>, ub: Vec<f64> },
    Dict { ctype: ConstraintType, fun: Arc<dyn Fn(&[f64]) -> f64 + Send + Sync + 'static> },
}

#[derive(Clone)]
struct LinearConstr { a: Vec<Vec<f64>>, lb: Vec<f64>, ub: Vec<f64>, keep_feasible: Option<bool> }
#[derive(Clone)]
struct NonlinearConstrBounds { fun: Arc<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync + 'static>, lb: Vec<f64>, ub: Vec<f64> }
#[derive(Clone)]
struct DictConstr { ctype: ConstraintType, fun: Arc<dyn Fn(&[f64]) -> f64 + Send + Sync + 'static> }

#[derive(Clone, Default)]
pub struct CombinedConstraints { linear: Vec<LinearConstr>, nonlinear: Vec<NonlinearConstrBounds>, dict: Vec<DictConstr> }

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
    pub fn to_py_list<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, PyList> {
        let mut items: Vec<pyo3::Bound<'py, PyAny>> = Vec::with_capacity(self.linear.len() + self.nonlinear.len() + self.dict.len());
        let optimize = PyModule::import(py, "scipy.optimize").unwrap();

        if !self.linear.is_empty() {
            let constr_cls = optimize.getattr("LinearConstraint").unwrap();
            for lc in &self.linear {
                let tol = 0.0_f64;
                let mut a_eq: Vec<Vec<f64>> = Vec::new();
                let mut lb_eq: Vec<f64> = Vec::new();
                let mut ub_eq: Vec<f64> = Vec::new();
                let mut a_in: Vec<Vec<f64>> = Vec::new();
                let mut lb_in: Vec<f64> = Vec::new();
                let mut ub_in: Vec<f64> = Vec::new();
                for (i, row) in lc.a.iter().enumerate() {
                    if i < lc.lb.len() && i < lc.ub.len() && (lc.lb[i] - lc.ub[i]).abs() <= tol {
                        a_eq.push(row.clone()); lb_eq.push(lc.lb[i]); ub_eq.push(lc.ub[i]);
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
