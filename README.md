# 🚀 CUDA Image Filter Accelerator

High-performance CUDA application that applies advanced image filtering using multiple GPU kernel strategies. Optimized to leverage shared memory, constant memory, and arithmetic intensity for real-time edge detection on grayscale images. Kernel 5 achieves over **2× speedup** compared to baseline and includes additional transfer optimizations using pinned memory.

> 🧠 Built as a systems-level project for GPU parallel computing (CSC367H5).
> 🛠️ Developed and profiled 5 custom kernels with increasing levels of optimization.
> ✅ Final implementation includes custom timers, modular utility functions, and high-efficiency normalization logic.

---

## 🧱 Tech Stack

| Language   | Tools & Libraries | Infrastructure    |
| ---------- | ----------------- | ----------------- |
| C++ / CUDA | nvcc, pthreads    | Linux, Make/CMake |

---

## 🔍 Project Overview

This project accelerates 2D image filtering using CUDA kernels. Each grayscale image is filtered using Laplacian kernels, followed by normalization of the output based on computed min/max values.

### Key Capabilities:

* Efficient parallel pixel computation using multiple kernel designs.
* Flexible support for 3x3, 5x5, and 9x9 Laplacian filters.
* Modular utility architecture (`kernel_common.cu`) for memory allocation, transfer, and cleanup.
* Advanced normalization kernel with smart skipping for uniform outputs.

---

## 🧪 Performance Highlights

| Optimization Technique             | Impact                           |
| ---------------------------------- | -------------------------------- |
| Shared Memory Reductions           | Improved per-block min/max speed |
| Constant Memory for Filters        | Faster repeated reads            |
| Pinned Host Memory                 | 2× faster transfer times         |
| Kernel Fusion (filter + reduction) | \~200% speedup vs baseline       |
| Removed Redundant Branching        | Reduced warp divergence          |
| Arithmetic Precomputation          | Cleaner and faster kernel logic  |
| Row-Major Access Optimization      | Improved spatial locality        |

> ✅ Kernel 5 outperformed all others with **>10% runtime improvement** for medium and large images
> 🔄 Transfer time reduced significantly with pinned memory
> 🔬 Edge cases (uniform images) handled optimally with skipped normalization

---

Command Line Flags:

* `-i`: input image file
* `-o`: output image file
* `-f`: filter type (1: 3x3, 2: 5x5, 3: 9x9, 4: identity)
* `-m`: method (1–5: various kernels)
* `-n`: threads per block (e.g., 256)
* `-t`: toggle timing (1 = on, 0 = off)


### 🔨 Compile with CMake

```bash
mkdir build && cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release
make
```

> ⚠️ Use `Release` mode (`-O3`) for performance measurements.
> ✅ Outputs: `main`, `pgm_creator`, `test_solution`

---

## 🧠 What I Learned

* Deepened understanding of CUDA execution model and GPU memory hierarchy
* Practiced reduction strategies with intra-block shared memory coordination
* Learned to balance compute and memory access patterns for GPU efficiency
* Applied software engineering best practices to C++/CUDA:

  * Functional decomposition
  * Modular utility headers
  * RAII-style custom timers
* Gained insight into GPU-specific performance tuning and measurement tooling

---

## 🧪 Testing

Run built-in tests to validate correctness:

```bash
./test_solution
```

> Add additional tests in `tests.cu` as needed to verify edge cases or new kernels.

---

## 🤝 Collaboration & Credits

> 💡 Developed independently with architectural mentoring.
> 🧠 Optimization insights and performance tuning guided by experimentation.

---

## 🔗 Links
* 🎓 Course: CSC367H5 – Parallel Programming at University of Toronto
