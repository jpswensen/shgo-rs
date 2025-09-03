//! Minimal C-ABI for calling shgo from C/C++
use std::os::raw::c_void;
use std::ptr::null_mut;

use crate::shgo;
use crate::options::{MinimizerKwargs, ShgoOptions};
use crate::constraints::{ConstraintSpec, ConstraintType};
use std::collections::HashMap;
use std::ffi::CStr;

#[repr(C)]
pub struct ShgoResultC {
    // pointer to heap-allocated array of length n
    pub x: *mut f64,
    pub n: usize,
    pub fun: f64,
    pub success: bool,
}

#[repr(C)]
pub struct Bounds {
    pub lo: f64,
    pub hi: f64,
}

// Callback signatures for objective and optional callback
// typedef double (*shgo_objective_t)(const double* x, size_t n, void* user);
pub type ShgoObjective = extern "C" fn(*const f64, usize, *mut c_void) -> f64;
// typedef void (*shgo_callback_t)(const double* x, size_t n, void* user);
pub type ShgoCallback = extern "C" fn(*const f64, usize, *mut c_void);

#[no_mangle]
pub extern "C" fn shgo_run_c(
    obj: ShgoObjective,
    user_data: *mut c_void,
    bounds_ptr: *const Bounds,
    bounds_len: usize,
    workers: usize,
) -> ShgoResultC {
    // Safe view of bounds
    let bounds: Vec<(f64, f64)> = if bounds_ptr.is_null() || bounds_len == 0 {
        vec![]
    } else {
        let slice = unsafe { std::slice::from_raw_parts(bounds_ptr, bounds_len) };
        slice.iter().map(|b| (b.lo, b.hi)).collect()
    };

    // Wrap C fn into Rust closure that captures user_data by move.
    // Avoid capturing raw pointers directly; store as usize which is Send+Sync.
    let user_raw = user_data as usize;
    let objective = move |x: &[f64]| -> f64 {
        let ud = user_raw as *mut c_void;
        obj(x.as_ptr(), x.len(), ud)
    };

    let result = shgo(
        objective,
        &bounds,
        None,
        None,
        None::<fn(&[f64])>,
        None,
        None,
        None,
        None,
        Some(workers.max(1)),
    );

    // Copy x to heap for C side, provide free function below
    let mut x_vec = result.x.clone();
    let x_ptr = if x_vec.is_empty() {
        null_mut()
    } else {
        let ptr = x_vec.as_mut_ptr();
        std::mem::forget(x_vec);
        ptr
    };
    ShgoResultC { x: x_ptr, n: result.x.len(), fun: result.fun.unwrap_or(f64::NAN), success: result.success.unwrap_or(false) }
}

#[no_mangle]
pub extern "C" fn shgo_result_free(res: *mut ShgoResultC) {
    if res.is_null() { return; }
    unsafe {
        let r = &mut *res;
        if !r.x.is_null() && r.n > 0 {
            let _ = Vec::from_raw_parts(r.x, r.n, r.n);
            r.x = null_mut();
            r.n = 0;
        }
    }
}

// -------- Extended C API: options, kwargs, constraints --------

#[repr(C)]
pub struct MinimizerOptionKV {
    pub key: *const i8, // C string (UTF-8)
    pub value: f64,
}

#[repr(C)]
pub struct MinimizerKwargsC {
    pub method: *const i8, // nullable
    pub options: *const MinimizerOptionKV, // nullable
    pub options_len: usize,
}

#[repr(C)]
pub struct ShgoOptionsC {
    pub has_maxfev: bool, pub maxfev: usize,
    pub has_f_min: bool, pub f_min: f64,
    pub has_f_tol: bool, pub f_tol: f64,
    pub has_maxiter: bool, pub maxiter: usize,
    pub has_maxev: bool, pub maxev: usize,
    pub has_maxtime: bool, pub maxtime: f64,
    pub has_minhgrd: bool, pub minhgrd: usize,
    pub has_minimize_every_iter: bool, pub minimize_every_iter: bool,
    pub has_local_iter: bool, pub local_iter: usize,
    pub has_infty_constraints: bool, pub infty_constraints: f64,
    pub has_disp: bool, pub disp: bool,
    // symmetry as list of indices
    pub symmetry: *const usize, // nullable
    pub symmetry_len: usize,
}

#[repr(C)]
pub struct LinearConstraintC {
    pub a: *const f64, // row-major m x n
    pub rows: usize,
    pub cols: usize,
    pub lb: *const f64, // len m
    pub ub: *const f64, // len m
    pub has_keep_feasible: bool,
    pub keep_feasible: bool,
}

pub type NonlinearVecFunC = extern "C" fn(*const f64, usize, *mut f64, usize, *mut c_void);

#[repr(C)]
pub struct NonlinearBoundsC {
    pub fun: NonlinearVecFunC,
    pub user: *mut c_void,
    pub m: usize, // number of outputs
    pub lb: *const f64, // len m
    pub ub: *const f64, // len m
}

#[repr(C)]
pub struct DictConstraintC {
    pub ctype: i32, // 0=eq, 1=ineq
    pub fun: ShgoObjective,
    pub user: *mut c_void,
}

#[no_mangle]
pub extern "C" fn shgo_run_c_ex(
    obj: ShgoObjective,
    obj_user: *mut c_void,
    bounds_ptr: *const Bounds,
    bounds_len: usize,
    n: usize, // 0 = None
    iters: usize, // 0 = None
    cb: Option<ShgoCallback>,
    cb_user: *mut c_void,
    minimizer: *const MinimizerKwargsC, // nullable
    options: *const ShgoOptionsC, // nullable
    lin_ptr: *const LinearConstraintC, lin_len: usize,
    nlb_ptr: *const NonlinearBoundsC, nlb_len: usize,
    dict_ptr: *const DictConstraintC, dict_len: usize,
    sampling_method: *const i8, // nullable C string
    workers: usize,
) -> ShgoResultC {
    // bounds
    let bounds: Vec<(f64, f64)> = if bounds_ptr.is_null() || bounds_len == 0 {
        vec![]
    } else {
        let slice = unsafe { std::slice::from_raw_parts(bounds_ptr, bounds_len) };
        slice.iter().map(|b| (b.lo, b.hi)).collect()
    };

    // objective closure
    let obj_user_raw = obj_user as usize;
    let objective = move |x: &[f64]| -> f64 { obj(x.as_ptr(), x.len(), obj_user_raw as *mut c_void) };

    // callback
    let callback_opt: Option<Box<dyn Fn(&[f64]) + Send + Sync + 'static>> = cb.map(|f| {
        let cb_user_raw = cb_user as usize;
        Box::new(move |x: &[f64]| f(x.as_ptr(), x.len(), cb_user_raw as *mut c_void)) as _
    });

    // minimizer kwargs
    let minimizer_kwargs = if minimizer.is_null() { None } else {
        let m = unsafe { &*minimizer };
        let method = if m.method.is_null() { None } else { Some(unsafe { CStr::from_ptr(m.method) }.to_string_lossy().into_owned()) };
        let mut map: HashMap<String, f64> = HashMap::new();
        if !m.options.is_null() && m.options_len > 0 {
            let opts = unsafe { std::slice::from_raw_parts(m.options, m.options_len) };
            for kv in opts {
                if !kv.key.is_null() {
                    let k = unsafe { CStr::from_ptr(kv.key) }.to_string_lossy().into_owned();
                    map.insert(k, kv.value);
                }
            }
        }
        Some(MinimizerKwargs { method, options: if map.is_empty() { None } else { Some(map) } })
    };

    // shgo options
    let options_rs = if options.is_null() { None } else {
        let o = unsafe { &*options };
        let symmetry = if !o.symmetry.is_null() && o.symmetry_len > 0 {
            Some(unsafe { std::slice::from_raw_parts(o.symmetry, o.symmetry_len) }.to_vec())
        } else { None };
        Some(ShgoOptions {
            maxfev: if o.has_maxfev { Some(o.maxfev as i64) } else { None },
            f_min: if o.has_f_min { Some(o.f_min) } else { None },
            f_tol: if o.has_f_tol { Some(o.f_tol) } else { None },
            maxiter: if o.has_maxiter { Some(o.maxiter as i64) } else { None },
            maxev: if o.has_maxev { Some(o.maxev as i64) } else { None },
            maxtime: if o.has_maxtime { Some(o.maxtime) } else { None },
            minhgrd: if o.has_minhgrd { Some(o.minhgrd as i64) } else { None },
            minimize_every_iter: if o.has_minimize_every_iter { Some(o.minimize_every_iter) } else { None },
            local_iter: if o.has_local_iter { Some(o.local_iter as i64) } else { None },
            infty_constraints: if o.has_infty_constraints { Some(o.infty_constraints != 0.0) } else { None },
            disp: if o.has_disp { Some(o.disp) } else { None },
            symmetry,
        })
    };

    // constraints
    let mut all_constraints: Vec<ConstraintSpec> = Vec::new();
    if !lin_ptr.is_null() && lin_len > 0 {
        let slice = unsafe { std::slice::from_raw_parts(lin_ptr, lin_len) };
        for lc in slice {
            let m = lc.rows; let n = lc.cols;
            let a = if m > 0 && n > 0 && !lc.a.is_null() {
                let data = unsafe { std::slice::from_raw_parts(lc.a, m * n) };
                let mut rows_v = Vec::with_capacity(m);
                for r in 0..m {
                    let start = r * n; let end = start + n;
                    rows_v.push(data[start..end].to_vec());
                }
                rows_v
            } else { Vec::new() };
            let lb = if !lc.lb.is_null() && m > 0 { unsafe { std::slice::from_raw_parts(lc.lb, m) }.to_vec() } else { Vec::new() };
            let ub = if !lc.ub.is_null() && m > 0 { unsafe { std::slice::from_raw_parts(lc.ub, m) }.to_vec() } else { Vec::new() };
            all_constraints.push(ConstraintSpec::Linear { a, lb, ub, keep_feasible: if lc.has_keep_feasible { Some(lc.keep_feasible) } else { None } });
        }
    }
    if !nlb_ptr.is_null() && nlb_len > 0 {
        let slice = unsafe { std::slice::from_raw_parts(nlb_ptr, nlb_len) };
        for nb in slice {
            let m = nb.m;
            let fun = nb.fun;
            let user_raw = nb.user as usize;
            let fun_closure = move |x: &[f64]| -> Vec<f64> {
                let mut out = vec![0.0_f64; m];
                fun(x.as_ptr(), x.len(), out.as_mut_ptr(), m, user_raw as *mut c_void);
                out
            };
            let have_lb = !nb.lb.is_null() && m > 0;
            let have_ub = !nb.ub.is_null() && m > 0;
            let lb = if have_lb { unsafe { std::slice::from_raw_parts(nb.lb, m) }.to_vec() } else { vec![0.0; m] };
            let ub = if have_ub { unsafe { std::slice::from_raw_parts(nb.ub, m) }.to_vec() } else { vec![f64::INFINITY; m] };
            all_constraints.push(ConstraintSpec::NonlinearBounds { fun: std::sync::Arc::new(fun_closure), lb, ub });
        }
    }
    if !dict_ptr.is_null() && dict_len > 0 {
        let slice = unsafe { std::slice::from_raw_parts(dict_ptr, dict_len) };
        for dc in slice {
            let ctype = match dc.ctype { 0 => ConstraintType::Eq, _ => ConstraintType::Ineq };
            let fun = dc.fun;
            let user_raw = dc.user as usize;
            let fun_closure = move |x: &[f64]| -> f64 { fun(x.as_ptr(), x.len(), user_raw as *mut c_void) };
            all_constraints.push(ConstraintSpec::Dict { ctype, fun: std::sync::Arc::new(fun_closure) });
        }
    }

    // sampling method
    let sampling_string: Option<String> = if sampling_method.is_null() { None } else { Some(unsafe { CStr::from_ptr(sampling_method) }.to_string_lossy().into_owned()) };

    // choose callback to pass into shgo with a concrete closure type
    let result = if let Some(cb_box) = callback_opt {
        shgo(
            objective,
            &bounds,
            if n > 0 { Some(n) } else { None },
            if iters > 0 { Some(iters) } else { None },
            Some(move |x: &[f64]| (cb_box)(x)),
            minimizer_kwargs,
            options_rs,
            if all_constraints.is_empty() { None } else { Some(all_constraints) },
            sampling_string.as_deref(),
            if workers > 0 { Some(workers) } else { None },
        )
    } else {
        shgo(
            objective,
            &bounds,
            if n > 0 { Some(n) } else { None },
            if iters > 0 { Some(iters) } else { None },
            None::<fn(&[f64])>,
            minimizer_kwargs,
            options_rs,
            if all_constraints.is_empty() { None } else { Some(all_constraints) },
            sampling_string.as_deref(),
            if workers > 0 { Some(workers) } else { None },
        )
    };

    // Copy x to heap for C side
    let mut x_vec = result.x.clone();
    let x_ptr = if x_vec.is_empty() { null_mut() } else { let ptr = x_vec.as_mut_ptr(); std::mem::forget(x_vec); ptr };
    ShgoResultC { x: x_ptr, n: result.x.len(), fun: result.fun.unwrap_or(f64::NAN), success: result.success.unwrap_or(false) }
}

#[no_mangle]
pub extern "C" fn shgo_run_c_ex_opt(
    obj: ShgoObjective,
    obj_user: *mut c_void,
    bounds_ptr: *const Bounds,
    bounds_len: usize,
    n: *const usize,
    iters: *const usize,
    cb: Option<ShgoCallback>,
    cb_user: *mut c_void,
    minimizer: *const MinimizerKwargsC,
    options: *const ShgoOptionsC,
    lin_ptr: *const LinearConstraintC, lin_len: usize,
    nlb_ptr: *const NonlinearBoundsC, nlb_len: usize,
    dict_ptr: *const DictConstraintC, dict_len: usize,
    sampling_method: *const i8,
    workers: *const usize,
) -> ShgoResultC {
    // convert pointer numerics to Option
    let n_opt = if n.is_null() { None } else { Some(unsafe { *n }) };
    let iters_opt = if iters.is_null() { None } else { Some(unsafe { *iters }) };
    let workers_opt = if workers.is_null() { None } else { Some(unsafe { *workers }) };
    // forward to existing ex by mapping Options to sentinel 0
    shgo_run_c_ex(
        obj,
        obj_user,
        bounds_ptr,
        bounds_len,
        n_opt.unwrap_or(0),
        iters_opt.unwrap_or(0),
        cb,
        cb_user,
        minimizer,
        options,
        lin_ptr,
        lin_len,
        nlb_ptr,
        nlb_len,
        dict_ptr,
        dict_len,
        sampling_method,
        workers_opt.unwrap_or(0),
    )
}
