
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

// Copies the data between the device and the host. It also takes 
// performance measurements.
// If the transfers are small, then it would be better in a real 
// application to do them batched transfers. You can do this by
// using a temporary array, preferably pinned, and packing all
// of the data that needs to be transferred into it. Transfer
// it when ready.
// this method can be used: (there is also a 3D version)
// cudaMemcpy2D(dest, dest_pitch, src, src_pitch, w, h, cudaMemcpyHostToDevice)
void profileCopies(float* h_a, float* h_b, float* d, unsigned int n, char* desc) {
	printf("\n%s transfers\n", desc);

	unsigned int bytes = n * sizeof(float);

	//events for timing
	cudaEvent_t startEvent, stopEvent;
	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));

	// record the time it takes for the host to copy the
	// data over to the device.
	// Note, it's better to do this kind of analysis with
	// nvprof or Nsight rather than instrument the code.
	checkCuda(cudaEventRecord(startEvent, 0));
	checkCuda(cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice));
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	// print result
	float time;
	checkCuda(cudaEventElapsedTime(&time, startEvent, stopEvent));
	printf("  Host to Deice bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

	// Do the same thing from the device back to the host.
	checkCuda(cudaEventRecord(startEvent, 0));
	checkCuda(cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost));
	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	
	checkCuda(cudaEventElapsedTime(&time, startEvent, stopEvent));
	printf("  Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

	for (int i = 0; i < n; i++) {
		if (h_a[i] != h_b[i]) {
			printf("*** %s transfers failed *** \n", desc);
			break;
		}
	}

	// clean up events
	checkCuda(cudaEventDestroy(startEvent));
	checkCuda(cudaEventDestroy(stopEvent));
}

int main()
{
	unsigned int nElements = 4 * 1024 * 1024;
	const unsigned int bytes = nElements * sizeof(float);

	//host arrays
	float *h_aPageable, *h_bPageable;
	float *h_aPinned, *h_bPinned;

	// device array
	float *d_a;

	// allocate and initialize
	h_aPageable = (float*)malloc(bytes);                   // host pageable
	h_bPageable = (float*)malloc(bytes);                   // host pageable
	checkCuda(cudaMallocHost((void**)&h_aPinned, bytes));  // host pinned
	checkCuda(cudaMallocHost((void**)&h_bPinned, bytes));  // host pinned
	checkCuda(cudaMalloc((void**)&d_a, bytes));            // device

	for (int i = 0; i < nElements; ++i) {
		h_aPageable[i] = i;
	}
	memcpy(h_aPinned, h_aPageable, bytes);
	memset(h_bPageable, 0, bytes);
	memset(h_bPageable, 0, bytes);

	// output device info and transfer size
	cudaDeviceProp prop;
	checkCuda(cudaGetDeviceProperties(&prop, 0));

	printf("\nDevice: %s\n", prop.name);
	printf("Transfer size (MB): %d\n", bytes / (1024 * 1024));

	// perform copies and report bandwidth
	profileCopies(h_aPageable, h_bPageable, d_a, nElements, "Pageable");
	profileCopies(h_aPinned, h_bPinned, d_a, nElements, "Pinned");

	printf("\n");

	// cleanup
	cudaFree(d_a);
	cudaFreeHost(h_aPinned);
	cudaFreeHost(h_bPinned);
	free(h_aPageable);
	free(h_bPageable);

	// On my machine the pinned memory is over 3 times faster.
	// This is all device dependent, however.
	// Do not overuse Pinned Memory though. It limits the memory
	// available to the operating system, etc. So, test to make
	// sure the application is working suitably.

	// Ultimately, take care to minimize the number of transfers
	// and to optomize them when they must happen. This is the 
	// bottleneck of hybrid CPU/GPU computing.
    return 0;
}