# cbet_cu - A GPU-accelerated simulation of Cross Beam Energy Transfer

Cross Beam Energy Transfer is an interaction of multiple high-intensity lasers in a high-density plasma in which some energy is exchanged between the lasers. This code simulates the interaction, computing the ray-based model introduced by [Follett et. al.](https://pubs.aip.org/aip/pop/article/30/4/042102/2882936/Ray-based-cross-beam-energy-transfer-modeling-for) This specific implementation of the model uses CUDA to accelerate the execution of the program.

This file contains some documentation on how the code works, in case anyone might want to use or work on it in the future. I'm not a physicist so a lot of the physics that happens goes over my head, and this is written more from a CS perspective.

Each beam is represented as a set of rays (and crossings between those rays and the mesh), and the plasma is represented as a linearly spaced mesh.

## Running it

```
$ make
$ ./cbet
```

Depends on CUDA, OpenMP, and HDF5 libraries. HDF5 must be compiled with OpenMP and C++ support.

## Files

* `main.cpp` - The main file, which is the entry point into the program. It contains functions to read the mesh profile and save the result to an HDF5 file.
* `structs.hpp`, `consts.hpp` - Header files that define structs and constants used in the program. Some of the constants in `consts.hpp` should be edited when reconfiguring the program.
* `utils.cu*` - Utility functions used throughout the program. Some of them are implemented in the header file, so that they can be used across multiple CUDA files without having to generate relocatable device code in the CUDA compilation (which can impact performance).
* `ray_trace.*` - Ray tracing code. Uses CUDA and OpenMP. Finds the trajectories of the rays and saves in memory each crossing between each ray and the mesh zones.
* `cbet.*` - CBET implementation. Uses an iterative process to get the resulting laser intensities. Uses CUDA.
* `omega_beams.cuh`, `output_*` - contain the beam configuration and mesh profile used.

## Ray Tracing

- This ray tracing implementation can use multiple GPUs. It manages each GPU with its own thread by using OpenMP. The limit on the size of the problem that can be computed is determined by main memory size.
- The main ray trace function (`ray_trace()`) allocates memory on the GPUs, initializes some values, and then calls the CUDA kernel. It uses cudaMemGetInfo to determine how much information can fit on the GPUs, and if the GPU can't store all of the laser beams' information at once, it traces rays in batches, copying and resetting the GPU memory between batches.
- In order to calculate all of the values used for CBET, each ray has two child rays, whose trajectories are computed but only need to be stored while the parent ray is being traced. Thus, after all of the other memory is allocated on each GPU, the remaining memory is filled with space for the trajectories of the child rays. The number of trajectories determines how many parent rays we can trace simultaneously. Thus, each thread corresponds to one trajectory slot in memory, and may compute multiple rays if there isn't much space.
- The ray tracing kernel itself (`trace_rays()`) initalizes the position of the rays it's been given to trace before calling the functions that trace the two child rays and then the parent ray, by calling the `launch_child_ray()` and `launch_parent_ray()` device functions.
- The `launch_child_ray()` function is very simple and may be read first to understand how the movement of the rays is calculated. Most importantly, it refers to the `deden` computed in `ray_trace()`, the change in density at a point (d eden/dx, d eden/dy, d eden/dz), in order to calculate its velocity.
- The `launch_parent_ray()` function augments the code in `launch_child_ray()`. It needs to keep track of the index of the mesh zone it is currently in, as well as a few extra values (kds, permittivity). Most importantly, it needs to save in memory each a `Crossing` each time it crosses into a different mesh zone. The closure `new_crossing` computes values related to the crossing and saves them in memory, and may be called up to three times in a single iteration of the ray tracing loop. Crossings are first saved in local memory before being sorted and saved to main memory, removing redundant crossings. The `turn` variable records when the ray turns around.
- `raystore` saves the index of one ray (and crossing) per mesh zone per beam, to be used in CBET.
- In terms of design decisions, the kernel for ray tracing is one huge monolithic kernel, which may impact performance. Furthermore, a lot of local variables are used, limiting how many threads can be called in each block. Future work may profile this code for warp divergence and see if shared memory or other tricks further improve performance.

## CBET

- This CBET implementation uses just one GPU, and the size of the problem it can compute is limited by the memory available on the GPU.
- Each crossing is paired with one `w_mult` value (`cuda_w_mult_values`), one `CmultBounds` value, and any number of crossing multipliers (`Cmult`). Each crossing multiplier corresponds to a pair of laser beams which cross, so a crossing might have multiple if multiple beams intersect the zone it's in. The crossing multipliers are allocated on the GPU after all of the other memory is allocated, with as many as possible allocated. Any crossing multipliers that don't fit on the GPU have to be calculated again at each step, rather than being calculated once at the beginning of execution.
- The kernel `get_num_cmults()` computes the number of crossing multipliers we need to allocate by taking the maximum value of crossing multipliers needed for any given ray. This may be more than are needed in general but is a good approximation that makes it easier to compute and use the crossing multipliers, because coupling multipliers are computed in parallel and each ray needs to know where to start saving them within the array. One thread per ray, using `raystore` to find out which beams cross which zones.
- The `calc_coupling_mults()` kernel computes the aforementioned coupling multipliers. The math for all of it is in the `get_coupling_mult()` device function. It also populates the `CmultBounds` values. One thread per ray, which is the limit due to the CmultBounds values.
- The main loop uses the `get_cbet_gain()` kernel, which populates the `w_mult` values for each crossing, and then the `update_intensities()` kernel, which does what it says on the box. Then, it checks if the output has converged below a given threshhold in order to exit the loop.
- `get_cbet_gain()` has one thread per ray, although in theory there could be more since there's no data dependencies between crossings in this function. It computes the coupling multipliers if they weren't already saved in order to compute `w_mult`.
- `update_intensities()` is limited at one thread per ray, since each interaction modifies all of the intensities downstream of it in a given ray. It uses `BlockReduce` from NVIDIA's `cub` library to get the maximum convergence value (`conv_max` or `updateconv`, depending on where you look) within a given block, and then saves that to one location in memory. Ideally, there'd be another reduce after the block reduces, to get the maximum value across all blocks (which would require saving all of them and running another kernel, probably, or `DeviceReduce` from `cub`)---the current solution may encounter a data race in updating the value, but in a practical sense I haven't seen it mess up yet and it's good enough.
- Future work may investigate how to enable larger problems or work on multiple GPUs. This is more difficult for CBET than for ray tracing because of data dependencies: each ray of each beam has to have the information from every other beam in order to compute its `w_mult` value. My initial idea was to do it in batches, like with ray tracing---at each `get_cbet_gain()` execution, the interactions between two batches would be multiplied before switching which batches are in memory until all pairs of batches were computed, kind of like squaring a matrix. I have not had the chance to implement this idea yet. One worry is that all of the memory operations may cause a hit to performance. However, it may be the only option for larger problem sizes.

## Acknowledgements

Many thanks to Professor Adam Sefkow for supporting this research and to Shuang Zhai for writing the code this implementation is based on. Thanks for all of the help, advice, and direction provided as well.

This material is based upon work supported by the Department of Energy [National Nuclear Security Administration] University of Rochester “National Inertial Confinement Fusion Program” under Award Number(s) DE-NA0004144.
