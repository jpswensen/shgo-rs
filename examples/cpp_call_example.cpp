/*
Build using the following command (assuming you have a Python venv in the folder "./env" with scipy installed)

PY=./env/bin/python3
PYLIB=$($PY -c 'import sysconfig,glob,os; d=sysconfig.get_config_var("LIBDIR") or sysconfig.get_config_var("LIBPL") or ""; m=glob.glob(os.path.join(d,"libpython*.dylib")); print(m[0] if m else "")')
c++ -std=c++17 -I include examples/cpp_call_example.cpp "$(pwd)/target/release/libshgo_rs.dylib" "$PYLIB" \
  -Wl,-rpath,"$(dirname "$PYLIB")" -Wl,-rpath,"$(pwd)/target/release" \
  -o cpp_call_example

*/

#include "shgo_rs.hpp"
#include <cstdio>
#include <cmath>

int main() {
    using namespace shgo;

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // Simple usage: objective + bounds only
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    Objective rastrigin = [](const double* x, size_t n){
        const double a = 10.0; double s = a * (double)n;
        for (size_t i=0;i<n;++i) s += x[i]*x[i] - a * std::cos(2.0*M_PI*x[i]);
        return s;
    };
    std::vector<ShgoBounds> b = { {-5,5}, {-5,5}, {-5,5} };
    auto r1 = shgo::shgo(rastrigin, b, std::nullopt, std::nullopt);
    std::printf("simple: fun=%.6f success=%d\n", r1.fun, (int)r1.success);

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // Intermediate usage: options + minimizer + callback + sampling + workers (no constraints)
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    MinimizerKwargs mk2; mk2.method = std::string("SLSQP"); mk2.options = {{"tol",1e-8},{"maxiter",1500}};
    Options opt2; opt2.f_tol = 1e-8; opt2.minimize_every_iter = false; opt2.disp = false;
    // Callback cb2 = [](const double* x, size_t n){
    //     std::printf("cb2: ["); for (size_t i=0;i<n;++i) std::printf("%s%.4f", (i? ",":""), x[i]); std::printf("]\n");
    // };
    auto r2 = shgo::shgo(rastrigin, b, /*n*/std::nullopt, /*iters*/std::nullopt,
                          /*callback*/nullptr, /*minimizerkwargs*/&mk2, /*options*/&opt2,
                          /*linear*/nullptr, /*nonlinear*/nullptr, /*dict*/nullptr,
                          /*method*/"simplicial", /*workers*/12);
    std::printf("intermediate: fun=%.6f success=%d\n", r2.fun, (int)r2.success);

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // Complex usage: options, minimizer, linear + nonlinear + dict constraints, callback
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    MinimizerKwargs mk; mk.method = std::string("COBYQA"); mk.options = {{"tol",1e-6},{"maxiter",200}};
    Options opt; opt.f_tol = 1e-6; opt.minimize_every_iter = false; // symmetry left empty for demo

    // Linear: 2.3 x0 + 5.6 x1 + 11.1 x2 + 1.3 x3 >= 5
    LinearConstraint L; L.rows=1; L.cols=4; L.a = {2.3,5.6,11.1,1.3}; L.lb = {5.0}; L.ub = {INFINITY};

    // Nonlinear: one output g2(x) >= 0
    NonlinearBounds N; N.m=1; N.fun = [](const double* x, size_t n, double* out, size_t m){
        (void)n; (void)m;
        double q = 0.28*x[0]*x[0] + 0.19*x[1]*x[1] + 20.5*x[2]*x[2] + 0.62*x[3]*x[3];
        out[0] = (12.0*x[0] + 11.9*x[1] + 41.8*x[2] + 52.1*x[3] - 21.0) - 1.645 * std::sqrt(q);
    };

    // Dict equality: h(x) == 0
    DictConstraint D; D.type = DictConstraint::Eq; D.fun = [](const double* x, size_t n){
        (void)n; return x[0] + x[1] + x[2] + x[3] - 1.0; };

    // Callback to print points
    Callback cb = [](const double* x, size_t n){
        std::printf("cb: ["); for (size_t i=0;i<n;++i) std::printf("%s%.4f", (i? ",":""), x[i]); std::printf("]\n");
    };

    std::vector<ShgoBounds> b4 = { {0,1},{0,1},{0,1},{0,1} };
    std::vector<LinearConstraint> lvec{L};
    std::vector<NonlinearBounds> nlvec{N};
    std::vector<DictConstraint> dvec{D};
    auto r3 = shgo::shgo(rastrigin, b4, /*n*/150, /*iters*/0, cb, &mk, &opt,
                          &lvec, &nlvec, &dvec,
                          "sobol", /*workers*/1);
    std::printf("complex: fun=%.6f success=%d\n", r3.fun, (int)r3.success);

    return 0;
}
