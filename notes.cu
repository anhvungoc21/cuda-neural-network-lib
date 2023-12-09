// SIMT: Simple Instruction, Multiple Threads

// Architecture
// Threads
//  - Lowest granualarity
//  - Execute instructions
// Warps
//  - Lowest SCHEDULABLE granularity
//  - Executes same instructions together (lock-step)
// Thread blocks
//  - Lowest PROGRAMMABLE entity
//  - Assigned to a single shader core
// Grids
//  - How a problem is mapped to the GPU
//  - Part of GPU LAUNCH PARAMETERS (#Blocks, #Threads

// Matrix Multiplication
// -> Think in terms of the resulting matrix!
//    Each thread is responsible for one cell of that matrix
//    -> One row of first matrix, One column of second matrix
//    -> Only ever need ONE for-loop for each thread

// Cache Tiling
// -> Shared Memory (Scratchpad)
//    - User-managed L1 Cache
//    - Private per block
// => Basically we copy each "tile" of the matrix to its corresponding block's
// shared memory
// => This way we always access cache instead of memory

// Coalescing
// - In terms of memory addresses, matrices are in row-major order cuz it's 1D
// of rows
// => Matrix A: Each thread accesses a different ROW => misaligned
// => Matrix B: Each thread accesses a different COLUMN => aligned (adjacent)
//              Multiple adjacent accesses can be coalesced into a single wide
//              access
// ==> Solution: Transpose the A Matrix!!.
// This doesn't help crazily though

// L1, L2 caches?


// Single vs Double precision?
// https://www.geeksforgeeks.org/difference-between-single-precision-and-double-precision/


// INSANE GUIDE:
// https://siboehm.com/articles/22/CUDA-MMM