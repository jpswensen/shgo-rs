#pragma once
#include "shgo_rs.h"
#include <vector>
#include <string>
#include <optional>
#include <functional>

namespace shgo {

struct ShgoBounds { double lo, hi; };

struct MinimizerKwargs {
    std::optional<std::string> method;
    std::vector<std::pair<std::string,double>> options;
};

struct Options {
    std::optional<size_t> maxfev;
    std::optional<double> f_min;
    std::optional<double> f_tol;
    std::optional<size_t> maxiter;
    std::optional<size_t> maxev;
    std::optional<double> maxtime;
    std::optional<size_t> minhgrd;
    std::optional<bool> minimize_every_iter;
    std::optional<size_t> local_iter;
    std::optional<bool> infty_constraints;
    std::optional<bool> disp;
    std::vector<size_t> symmetry;
};

using Objective = std::function<double(const double*, size_t)>;
using Callback  = std::function<void(const double*, size_t)>;

struct LinearConstraint {
    std::vector<double> a; // row-major m x n
    size_t rows{0}, cols{0};
    std::vector<double> lb, ub; // len m
    std::optional<bool> keep_feasible;
};

using NonlinearVecFun = std::function<void(const double*, size_t, double*, size_t)>;
struct NonlinearBounds {
    NonlinearVecFun fun; size_t m{0};
    std::vector<double> lb, ub; // len m
};

struct DictConstraint {
    enum Type { Eq=0, Ineq=1 } type{Ineq};
    Objective fun;
};

struct Result { std::vector<double> x; double fun{0}; bool success{false}; };

inline MinimizerKwargsC to_c(const MinimizerKwargs& mk, std::vector<MinimizerOptionKV>& storage) {
    storage.clear(); storage.reserve(mk.options.size());
    for (auto& kv : mk.options) storage.push_back(MinimizerOptionKV{ kv.first.c_str(), kv.second });
    MinimizerKwargsC c{};
    c.method = mk.method ? mk.method->c_str() : nullptr;
    c.options = storage.empty() ? nullptr : storage.data();
    c.options_len = storage.size();
    return c;
}

inline ShgoOptionsC to_c(const Options& o, std::vector<size_t>& sym_storage) {
    ShgoOptionsC c{};
    if (o.maxfev) { c.has_maxfev = true; c.maxfev = *o.maxfev; }
    if (o.f_min) { c.has_f_min = true; c.f_min = *o.f_min; }
    if (o.f_tol) { c.has_f_tol = true; c.f_tol = *o.f_tol; }
    if (o.maxiter) { c.has_maxiter = true; c.maxiter = *o.maxiter; }
    if (o.maxev) { c.has_maxev = true; c.maxev = *o.maxev; }
    if (o.maxtime) { c.has_maxtime = true; c.maxtime = *o.maxtime; }
    if (o.minhgrd) { c.has_minhgrd = true; c.minhgrd = *o.minhgrd; }
    if (o.minimize_every_iter) { c.has_minimize_every_iter = true; c.minimize_every_iter = *o.minimize_every_iter; }
    if (o.local_iter) { c.has_local_iter = true; c.local_iter = *o.local_iter; }
    if (o.infty_constraints) { c.has_infty_constraints = true; c.infty_constraints = *o.infty_constraints ? 1.0 : 0.0; }
    if (o.disp) { c.has_disp = true; c.disp = *o.disp; }
    sym_storage = o.symmetry; c.symmetry = sym_storage.empty() ? nullptr : sym_storage.data(); c.symmetry_len = sym_storage.size();
    return c;
}

inline Result shgo(
    Objective obj,
    const std::vector<ShgoBounds>& bounds,
    std::optional<size_t> n = std::nullopt,
    std::optional<size_t> iters = std::nullopt,
    Callback cb = nullptr,
    const MinimizerKwargs* minimizer = nullptr,
    const Options* options = nullptr,
    const std::vector<LinearConstraint>* linear = nullptr,
    const std::vector<NonlinearBounds>* nlb = nullptr,
    const std::vector<DictConstraint>* dict = nullptr,
    const char* sampling_method = nullptr,
    std::optional<size_t> workers = std::nullopt
) {
    // C shims storage
    std::vector<ShgoBounds> b = bounds;
    const size_t* n_ptr = n ? &*n : nullptr;
    const size_t* iters_ptr = iters ? &*iters : nullptr;
    const size_t* workers_ptr = workers ? &*workers : nullptr;

    // Objective trampoline
    struct ObjCtx { Objective* f; } ctx{ &obj };
    auto c_obj = [](const double* x, size_t nx, void* user)->double{
        auto* c = static_cast<ObjCtx*>(user);
        return (*c->f)(x, nx);
    };

    // Callback trampoline
    ShgoCallback c_cb = nullptr;
    struct CbCtx { Callback* f; } cbctx{ &cb };
    auto c_cb_impl = [](const double* x, size_t nx, void* user){
        auto* c = static_cast<CbCtx*>(user);
        if (c && c->f && *c->f) (*c->f)(x, nx);
    };
    if (cb) c_cb = +c_cb_impl;

    // MinimizerKwargsC
    std::vector<MinimizerOptionKV> opt_storage;
    MinimizerKwargsC mkc{}; const MinimizerKwargsC* mkc_ptr = nullptr;
    if (minimizer) { mkc = to_c(*minimizer, opt_storage); mkc_ptr = &mkc; }

    // OptionsC
    std::vector<size_t> sym_storage; ShgoOptionsC oc{}; const ShgoOptionsC* oc_ptr = nullptr;
    if (options) { oc = to_c(*options, sym_storage); oc_ptr = &oc; }

    // Linear constraints
    std::vector<LinearConstraintC> lcs; const LinearConstraintC* lcs_ptr = nullptr;
    if (linear && !linear->empty()) {
        lcs.reserve(linear->size());
        for (auto& L : *linear) {
            LinearConstraintC c{}; c.rows = L.rows; c.cols = L.cols; c.a = L.a.data();
            c.lb = L.lb.empty()? nullptr : L.lb.data(); c.ub = L.ub.empty()? nullptr : L.ub.data();
            if (L.keep_feasible) { c.has_keep_feasible = true; c.keep_feasible = *L.keep_feasible; }
            lcs.push_back(c);
        }
        lcs_ptr = lcs.data();
    }

    // Nonlinear bounds
    std::vector<NonlinearBoundsC> nbcs; const NonlinearBoundsC* nbcs_ptr = nullptr;
    std::vector<NonlinearVecFun> nlfuns_storage;
    if (nlb && !nlb->empty()) {
        nbcs.reserve(nlb->size());
        nlfuns_storage.reserve(nlb->size());
        for (auto& N : *nlb) {
            nlfuns_storage.push_back(N.fun);
            NonlinearBoundsC c{}; c.m = N.m; c.lb = N.lb.empty()? nullptr : N.lb.data(); c.ub = N.ub.empty()? nullptr : N.ub.data();
            c.user = &nlfuns_storage.back();
            c.fun = +[](const double* x, size_t nx, double* out, size_t m, void* user){
                auto* f = static_cast<NonlinearVecFun*>(user);
                (*f)(x, nx, out, m);
            };
            nbcs.push_back(c);
        }
        nbcs_ptr = nbcs.data();
    }

    // Dict constraints
    std::vector<DictConstraintC> dcs; const DictConstraintC* dcs_ptr = nullptr;
    std::vector<Objective> dictfun_storage;
    if (dict && !dict->empty()) {
        dcs.reserve(dict->size());
        dictfun_storage.reserve(dict->size());
        for (auto& D : *dict) {
            dictfun_storage.push_back(D.fun);
            DictConstraintC c{}; c.ctype = (D.type==DictConstraint::Eq?0:1); c.user = &dictfun_storage.back();
            c.fun = +[](const double* x, size_t nx, void* user)->double{
                auto* f = static_cast<Objective*>(user);
                return (*f)(x, nx);
            };
            dcs.push_back(c);
        }
        dcs_ptr = dcs.data();
    }

    // Call C API
    ShgoResultC r = shgo_run_c_ex_opt(
        +c_obj, &ctx,
    reinterpret_cast<const ::Bounds*>(b.data()), b.size(),
        n_ptr, iters_ptr,
        c_cb, &cbctx,
        mkc_ptr, oc_ptr,
        lcs_ptr, lcs.size(),
        nbcs_ptr, nbcs.size(),
        dcs_ptr, dcs.size(),
        sampling_method,
        workers_ptr
    );

    // Copy result
    Result out; out.fun = r.fun; out.success = r.success; out.x.assign(r.x, r.x + r.n);
    shgo_result_free(&r);
    return out;
}

} // namespace shgo
