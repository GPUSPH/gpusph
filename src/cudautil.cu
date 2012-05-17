#include <stdio.h>
#include <cuda_gl_interop.h>

#include "cudautil.cuh"

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
