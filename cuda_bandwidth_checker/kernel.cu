
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <assert.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call.
// this is a No-op in release builds.
// inline suggests to the compiler to define this function in
// a way that it can be replaceable. This can speed up execution.
// this presents the compiler from going through the normal function
// overhead when it is called. It isn't looked up. It is compiled so
// that the instructions are just right there. This is used when the
// function has a small number of instructions.
inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n",
			cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

void profileCopies(float* h_a, float* h_b, float* d, unsigned int n, char* desc) {
	printf("\n%s transfers\n", desc);

	unsigned int bytes = n * sizeof(float);

	//events for timing
	cudaEvent_t startEvent, stopEvent;

	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));


}

int main()
{
	
    return 0;
}