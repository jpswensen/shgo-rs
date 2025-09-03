/*
Build using the following command (assuming you have a Python venv in the folder "./env" with scipy installed)

PY=./env/bin/python3 
PYLIB=$($PY -c 'import sysconfig,glob,os; d=sysconfig.get_config_var("LIBDIR") or sysconfig.get_config_var("LIBPL") or ""; m=glob.glob(os.path.join(d,"libpython*.dylib")); print(m[0] if m else "")')
cc -I include examples/c_call_example.c "$(pwd)/target/release/libshgo_rs.dylib" "$PYLIB" \
  -Wl,-rpath,"$(dirname "$PYLIB")" -Wl,-rpath,"$(pwd)/target/release" \
  -o c_call_example
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stddef.h>
#include <math.h>

// Headers for the C ABI
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

extern ShgoResultC shgo_run_c(ShgoObjective obj, void* user, const Bounds* bounds, size_t bounds_len, size_t workers);
extern void shgo_result_free(ShgoResultC* res);

static double rastrigin(const double* x, size_t n, void* user) {
    (void)user;
    const double a = 10.0;
    double sum = a * (double)n;
    for (size_t i = 0; i < n; ++i) {
        double xi = x[i];
        sum += xi*xi - a * cos(2.0 * 3.141592653589793 * xi);
    }
    return 2.0+sum;
}

int main(void) {
    Bounds b[3] = { {-5.0, 5.0}, {-5.0, 5.0}, {-5.0, 5.0} };
    ShgoResultC res = shgo_run_c(rastrigin, NULL, b, 3, 1);
    printf("fun=%.6f success=%d n=%zu\n", res.fun, res.success, res.n);
    for (size_t i = 0; i < res.n; ++i) {
        printf("x[%zu]=%.9f\n", i, res.x[i]);
    }
    shgo_result_free(&res);
    return 0;
}
