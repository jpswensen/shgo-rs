# shgo-rs
A Rust wrapper for the Python/SciPy implementation of the shgo global optimization algorithm

## Features

* This should mirror the Python+SciPy implementation pretty closely
* Rust uses Option<> and Some() to be able to specify optional parameters
* We created Rust-native companion structs of ```OptimizeResult```, ```MinimizerKwargs```, and ```ShgoOptions``` to mirror their Python counterparts
* We created Rust-native companion struct to do LinearConstraint, NonlinearConstraint, and dict+function based constraints
* The one deviation from the Python function signature is that Rust doesn't allow tuples of mixed types. So, an example is given where a partial function closure is used to capture non-optimized parameters so we don't have to include and ```args``` parameter of mixed types.
* The new version also has a wrapper for C and C++, along with examples of how to compile and use. These wrappers should feel pretty seamless and follow the same pattern as the Python and Rust interfaces. For C++, the wrapper uses std::optional to allow optional parameters.
* Six examples are provided:
  * Rosen with default parameters (same as the SHGO Python documentation)
  * Rastrigin with a partial function closure to demonstrate how to perform the alternative to the ```args``` parameter in Python
  * Rastrigin with a lot of extra parameters given
  * Rastrigin with some artificial extra computation to demonstrate the multi-thread functionality
  * The cattle feed problem with constraints in dict form
  * The cattle feed problem with constraints in a mix of the three different options: LinearConstraint, NonlinearConstraint, and dict

## Installation

1. Create a python venv and install scipy (I am using my existing 3.13t interpreter to create it, but you can use anything recent)
   ```
   python3.13t -m venv env
   source env/bin/activate
   pip install scipy
   ```
3. Run the examples
  ```
  source env/bin/activate
  RUSTFLAGS="-C target-cpu=native" cargo run --release --example basic_rosen
  RUSTFLAGS="-C target-cpu=native" cargo run --release  --example rastrigin_partial
  RUSTFLAGS="-C target-cpu=native" cargo run --release  --example rastrigin_extra_parameters
  RUSTFLAGS="-C target-cpu=native" cargo run --release  --example rastrigin_fake_long
  RUSTFLAGS="-C target-cpu=native" cargo run --release  --example cattle_feed_dict
  RUSTFLAGS="-C target-cpu=native" cargo run --release  --example cattle_feed_structured
  ```
  * You can use ```RUSTFLAGS="-C target-cpu=native"``` and ```--release``` to create an optimized build
  * With just ```cargo run --example <name_of_example>``` it will make a debug version
4. You can include this crate in your own program by pointing directly to this github repository from the [dependencies] section of your Cargo.toml
   ```
   shgo-rs = { git = "https://github.com/jpswensen/shgo-rs" }
   ```

## TODOs

* I'm not sure the multi-threading is working as well as I think it should. When I run the ```test_rastrigin_fake_long_partial()``` example, I do see the processor usage spike the percentage relative to the number of workers, but then it falls back to about 1-CPU load for the latter portions of the solution. I also see (if I print out thread ID) that it is running on different threads, but still not sure if simultaneous)
* Missing ShgoOptions: jac, hess, hessp (I'm not 100% sure how to implement these because they then require the objective function to have multiple return values)
