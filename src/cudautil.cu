#include <stdio.h>
#include <cuda_gl_interop.h>

#include "cudautil.cuh"
#include "compute_select.h"

cudaDeviceProp checkCUDA(const Options &options)
{
	int deviceCount;
	CUDA_SAFE_CALL_NOSYNC(cudaGetDeviceCount(&deviceCount));
	if (deviceCount == 0) {
		fprintf(stderr, "no CUDA device found!\n");
		exit(1);
	}

	int dev = options.device;

	if (dev < 0) dev = 0;
	else if (dev > deviceCount - 1)
		dev = deviceCount - 1;

	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL_NOSYNC(cudaGetDeviceProperties(&deviceProp, dev));

	if (deviceProp.major < 1) {
		fprintf(stderr, "device %d does not support CUDA!\n", dev);
		exit(1);
	}

	printf("Using device %d: %s\n", dev, deviceProp.name );
	CUDA_SAFE_CALL(cudaSetDevice(dev));
	CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	/* Check if we were compiled for the same compute capability as the device, and print
	   warning/informational messages otherwise. */
	int device_cc = deviceProp.major*10 + deviceProp.minor;
	if (device_cc != COMPUTE) {
		/* There is nothing wrong with being compiled for a different CC, except if the
		   CC of the device is _lower_ than the CC we were compiled for; prefix WARNING:
		   to the message only in this case */
		fprintf(stderr, "%sGPU has compute capability %u.%u, we are compiled for %u.%u\n",
			device_cc < COMPUTE ? "WARNING: " : "",
			deviceProp.major, deviceProp.minor, COMPUTE/10, COMPUTE-(COMPUTE/10)*10);
		/* Additionally, if the _major_ CC is different, print a specific warning.
		   This is done because things such as symbols don't seem to work well in
		   e.g. 2.0 devices when the code is written for 1.x */
		if (deviceProp.major != COMPUTE/10) {
			fprintf(stderr, "WARNING: code is compiled for a different MAJOR compute capability. Expect failures\n");
		}
	}

	return deviceProp;
}


void allocateArray(void **devPtr, size_t size)
{
	CUDA_SAFE_CALL(cudaMalloc(devPtr, size));
}


void freeArray(void *devPtr)
{
	CUDA_SAFE_CALL(cudaFree(devPtr));
}


void threadSync()
{
	CUDA_SAFE_CALL(cudaThreadSynchronize());
}


void copyArrayFromDevice(void* host, const void* device, GLuint vbo, int size)
{
	if (vbo)
		CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&device, vbo));
	CUDA_SAFE_CALL(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
	if (vbo)
		CUDA_SAFE_CALL(cudaGLUnmapBufferObject(vbo));
}


void copyArrayToDevice(void* device, const void* host, int offset, int size)
{
	CUDA_SAFE_CALL(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
}


void registerGLBufferObject(GLuint vbo)
{
	CUDA_SAFE_CALL(cudaGLRegisterBufferObject(vbo));
}


void unregisterGLBufferObject(GLuint vbo)
{
	CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(vbo));
}
