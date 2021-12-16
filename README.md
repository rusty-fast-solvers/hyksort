# HykSort

## Compile

Flags for taking advantage of fast math, and avx2 on processors where available.

```bash
export RUSTFLAGS="-C target-feature=+avx2,+fma"
cargo build --release
```