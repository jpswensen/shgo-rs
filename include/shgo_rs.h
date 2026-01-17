#ifndef SHGO_RS_H
#define SHGO_RS_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double* x;
    size_t n;
    double fun;
    bool success;
} ShgoResultC;

typedef struct {
    double lo;
    double hi;
} Bounds;

typedef double (*ShgoObjective)(const double* x, size_t n, void* user);

typedef void (*ShgoCallback)(const double* x, size_t n, void* user);

ShgoResultC shgo_run_c(ShgoObjective obj, void* user, const Bounds* bounds, size_t bounds_len, size_t workers);
void shgo_result_free(ShgoResultC* res);

// ---- Extended API: options, kwargs, constraints ----
typedef struct {
    const char* key;
    double value;
} MinimizerOptionKV;

typedef struct {
    const char* method; // nullable
    const MinimizerOptionKV* options; // nullable
    size_t options_len;
} MinimizerKwargsC;

typedef struct {
    bool has_maxfev; size_t maxfev;
    bool has_f_min; double f_min;
    bool has_f_tol; double f_tol;
    bool has_maxiter; size_t maxiter;
    bool has_maxev; size_t maxev;
    bool has_maxtime; double maxtime;
    bool has_minhgrd; size_t minhgrd;
    bool has_minimize_every_iter; bool minimize_every_iter;
    bool has_local_iter; size_t local_iter;
    bool has_infty_constraints; double infty_constraints;
    bool has_disp; bool disp;
    const size_t* symmetry; size_t symmetry_len; // nullable
} ShgoOptionsC;

typedef struct {
    const double* a; // row-major m x n
    size_t rows;
    size_t cols;
    const double* lb; // len m
    const double* ub; // len m
    bool has_keep_feasible; bool keep_feasible;
} LinearConstraintC;

typedef void (*NonlinearVecFunC)(const double* x, size_t n, double* out, size_t m, void* user);

typedef struct {
    NonlinearVecFunC fun; void* user; size_t m;
    const double* lb; const double* ub; // len m
} NonlinearBoundsC;

typedef struct {
    int ctype; // 0=eq, 1=ineq
    ShgoObjective fun; void* user;
} DictConstraintC;

ShgoResultC shgo_run_c_ex(
    ShgoObjective obj, void* obj_user,
    const Bounds* bounds, size_t bounds_len,
    size_t n, size_t iters,
    ShgoCallback cb, void* cb_user,
    const MinimizerKwargsC* minimizer, const ShgoOptionsC* options,
    const LinearConstraintC* lin, size_t lin_len,
    const NonlinearBoundsC* nlb, size_t nlb_len,
    const DictConstraintC* dict, size_t dict_len,
    const char* sampling_method,
    size_t workers
);

// Variant where numeric parameters are optional via pointer-null
ShgoResultC shgo_run_c_ex_opt(
    ShgoObjective obj, void* obj_user,
    const Bounds* bounds, size_t bounds_len,
    const size_t* n, const size_t* iters,
    ShgoCallback cb, void* cb_user,
    const MinimizerKwargsC* minimizer, const ShgoOptionsC* options,
    const LinearConstraintC* lin, size_t lin_len,
    const NonlinearBoundsC* nlb, size_t nlb_len,
    const DictConstraintC* dict, size_t dict_len,
    const char* sampling_method,
    const size_t* workers
);

#ifdef __cplusplus
}
#endif

#endif // SHGO_RS_H
