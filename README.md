# shgo-rs
A Rust wrapper for the Python/SciPy implementation of the shgo global optimization algorithm

## Features

* This should mirror the Python+SciPy implementation pretty closely
* Rust uses Option<> and Some() to be able to specify optional parameters
* We created Rust-native companion structs of ```OptimizeResult```, ```MinimizerKwargs```, and ```ShgoOptions``` to mirror their Python counterparts
* The one deviation from the Python function signature is that Rust doesn't allow tuples of mixed types. So, an example is given where a partial function closure is used to capture non-optimized parameters so we don't have to include and ```args``` parameter of mixed types.
* Four examples are provided:
  * Rosen with default parameters (same as the SHGO Python documentation)
  * Rastrigin with a partial function closure to demonstrate how to perform the alternative to the ```args``` parameter in Python
  * Rastrigin with a lot of extra parameters given
  * Rastrigin with some artificial extra computation to demonstrate the multi-thread functionality

## Installation

1. Create a python venv and install scipy (I am using my existing 3.13t interpreter to create it, but you can use anything recent)
   ```
   python3.13t -m venv env
   source env/bin/activate
   pip install scipy
   ```
3. Build the Rust program that includes the wrapper and some test examples (make sure you are in the environment you want to use while in the Rust program)
  ```
  source env/bin/activate
  RUSTFLAGS="-C target-cpu=native" cargo build --release
  ```
  * You can use ```RUSTFLAGS="-C target-cpu=native"``` and ```--release``` to create an optimized build
  * With just ```cargo build``` it will make a debug version
4. Run the compiled program
   ```
   ./target/release/shgo-rs
   ```
   * Alternatively, you can use ```./target/debug/shgo-rs``` if you built the debug version

## TODOs

* I'm not sure the multi-threading is working as well as I think it should. When I run the ```test_rastrigin_fake_long_partial()``` example, I do see the processor usage spike the percentage relative to the number of workers, but then it falls back to about 1-CPU load for the latter portions of the solution.
* I don't have the specification of constraints working yet. This may be a little more difficult to allow both specification of constraints by multiple methods: (a) ```NonlinearConstraint``` or ```LinearConstraint``` or (b) using a dictionary with named function callbacks
* I don't have all of the possible parameters implemented in ShgoOptions yet
