
// Macro for error checking on CUDA API calls
#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }

// Assert function to be used by macro for error checking
void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true);