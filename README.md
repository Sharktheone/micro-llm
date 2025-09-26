# micro-llm

Minimal, highly modular local LLM experimentation workspace written in Rust. The focus is on:
- Very small, explicit abstraction surface
- Pluggable backends (start with CPU) for tensor ops
- Zero / near‑zero copy model weight loading (safetensors, mmap)
- Simple tokenizer support (currently: subset of Hugging Face tokenizers JSON format)
- Clear separation between: backend primitives, execution (model logic), and frontend (binary)

Project status: early prototype / scaffolding. APIs are unstable.

## Design Highlights

1. Trait‑Driven Backend Abstraction
   The Backend + SupportsDType design lets each backend expose an associated Tensor type per dtype without enormous enums or dynamic dispatch. Rust associated types + specialization keep call sites generic while enabling tight monomorphization.

2. Dim Type Family
   Dim1 / Dim2 / Dim3 implement a Dim trait carrying associated Larger / Smaller types. This enables compile‑time dimension transitions (e.g. map_axis reduces rank, unsqueeze increases rank) while remaining ergonomic. Future: macro or const‑generic based expansion to N dims.

3. Store Parameterization
   Tensor<'a, T, B, S, D>: S encodes data ownership category (OwnedStore / RefStore / LoadStore) allowing:
   - Zero‑copy slices / views (RefStore)
   - Memory‑mapped model weights (LoadStore)
   - Owned writable buffers (OwnedStore)

4. Controlled Surface Area
   Only a minimal arithmetic + map + axis reduction set is defined now; more ops (matmul, layernorm, softmax, rotary embeddings, attention) will layer on top inside micro_executor to keep the backend lean.

## Current Capabilities

- Parse & enumerate safetensors files (test in micro_executor::load_safetensors)
- Represent tensors with generic ops (addition, scalar ops, mapping, slicing, axis reductions – interface defined, CPU impl partially forthcoming)
- Load Hugging Face tokenizer vocab (subset) into an internal structure
- Basic dtype support: f32, f64, f16 (bf16 wrapper present in executor for reading weights)



## Safety & Performance Notes

- Unsafe usage limited (mmap + bytemuck casts for weight views) and should be audited further.
- Rayon used for parallel map operations (simple data parallel loops) – gating heuristics TBD.
- Zero-copy philosophy: prefer referencing model weight bytes directly; conversion / copies explicit.


## License

MIT License (see LICENSE file).

## Disclaimer

Not production ready. APIs will break. Intended for educational exploration and rapid iteration around minimal LLM execution paths.
