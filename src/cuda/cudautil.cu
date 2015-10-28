#include <stdio.h>

#include "cudautil.cuh"
#include "compute_select.opt"

// TODO: errors should be thrown properly, the functions should not brutally terminate everything.
// Furthermore, the number of CUDA devices could be checked before the threads are started (i.e. when checking
// command line options in main()).
cudaDeviceProp checkCUDA(const GlobalData* gdata, uint devidx)
{
	int deviceCount;

	CUDA_SAFE_CALL_NOSYNC(cudaGetDeviceCount(&deviceCount));
	if (deviceCount == 0) {
		fprintf(stderr, "no CUDA device found!\n");
		exit(1);
	} else
		if (devidx==0) printf("%d CUDA devices detected\n", deviceCount);

	int cudaDevNum = gdata->device[devidx];

	// it is a semantic error to correct here the validity of the device num.
	// if a devnum is out of range, program should terminate, not try to "fix" it
	/* if (cudaDevNum < 0) cudaDevNum = 0;
	else if (cudaDevNum > deviceCount - 1)
		cudaDevNum = deviceCount - 1; */

	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL_NOSYNC(cudaGetDeviceProperties(&deviceProp, cudaDevNum));

	if (deviceProp.major < 1) {
		fprintf(stderr, "device %d does not support CUDA!\n", cudaDevNum);
		exit(1);
	}

	//printf("Using device %d: %s\n", cudaDevNum, deviceProp.name );
	CUDA_SAFE_CALL(cudaSetDevice(cudaDevNum));
	CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	/* Check if we were compiled for the same compute capability as the device, and print
	   warning/informational messages otherwise. */
	int device_cc = deviceProp.major*10 + deviceProp.minor;
	if (device_cc != COMPUTE) {
		/* There is nothing wrong with being compiled for a different CC, except if the
		   CC of the device is _lower_ than the CC we were compiled for; prefix WARNING:
		   to the message only in this case */
		fprintf(stderr, "%sDevice %d has compute capability %u.%u, we are compiled for %u.%u\n",
			device_cc < COMPUTE ? "WARNING: " : "", cudaDevNum,
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
