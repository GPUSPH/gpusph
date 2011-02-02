#ifndef _CUDAUTIL_CUH_
#define _CUDAUTIL_CUH_

#include "Options.h"
#include "cuda_call.h"

void checkCUDA(const Options&);

void allocateArray(void **devPtr, size_t size);

void freeArray(void *devPtr);

void threadSync();

void copyArrayFromDevice(void* host, const void* device, GLuint vbo, int size);

void copyArrayToDevice(void* device, const void* host, int offset, int size);

void registerGLBufferObject(GLuint vbo);

void unregisterGLBufferObject(GLuint vbo);

#endif
