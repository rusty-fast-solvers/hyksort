# HykSort

A Rusty implementation of HykSort from Sundar et. al [1] for ```Vec<T>``` arrays distributed using MPI.

The local sorts are not at all optimised, and this implementation uses Rust's ```.sort()``` method implemented via the ```Ord``` trait.


## Compile

Flags for taking advantage of fast math, and AVX2 on processors where available.

```bash
export RUSTFLAGS="-C target-feature=+avx2,+fma"
cargo build --release
```

## Test 

Run parallel tests.

```bash
cargo build && mpirun -n <nprocs> target/<debug/release>/parallel_tests
```

## References

[1] Sundar, H., Malhotra, D., & Biros, G. (2013, June). Hyksort: a new variant of hypercube quicksort on distributed memory architectures. In Proceedings of the 27th international ACM conference on international conference on supercomputing (pp. 293-302).
