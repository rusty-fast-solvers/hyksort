# HykSort

A rusty implementation of HykSort from Sundar et. al for ```Vec<u64>``` arrays
distributed using MPI.

## Caveats

1. Generically typed arrays not yet supported.
2. The splitter selection in this algorithm is not optimal, and relies heavily
on the distribution of your data. Use with caution.

## Compile

Flags for taking advantage of fast math, and avx2 on processors where available.

```bash
export RUSTFLAGS="-C target-feature=+avx2,+fma"
cargo build --release
```

### Archer 2

Scaling experiments have been performed on Archer 2, with 6 billion integers on
64 ranks spread across 32 nodes being sorted in approximately 50 seconds.

## References

[1] Sundar, H., Malhotra, D., & Biros, G. (2013, June). Hyksort: a new variant of hypercube quicksort on distributed memory architectures. In Proceedings of the 27th international ACM conference on international conference on supercomputing (pp. 293-302).
