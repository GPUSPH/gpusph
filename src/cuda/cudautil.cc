#include <cstdio>

#include "cudautil.h"
#include "compute_select.opt"

// TODO: errors should be thrown properly, the functions should not brutally terminate everything.
// Furthermore, the number of CUDA devices could be checked before the threads are started (i.e. when checking
// command line options in main()).
cudaDeviceProp checkCUDA(const GlobalData* gdata, uint devidx)
{
	int cudaDevNum = gdata->device[devidx];
	int deviceCount;
	cudaDeviceProp deviceProp;

	CUDA_SAFE_CALL_NOSYNC(cudaSetDevice(cudaDevNum));
	CUDA_SAFE_CALL_NOSYNC(cudaGetDeviceCount(&deviceCount));
	CUDA_SAFE_CALL_NOSYNC(cudaGetDeviceProperties(&deviceProp, cudaDevNum));

	printf("thread 0x%llx device idx %d: CUDA device %d/%d, PCI device %04x:%02x:%02x.0: %s\n",
		(unsigned long long)pthread_self(), devidx, cudaDevNum, deviceCount,
		deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID,
		deviceProp.name);

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

	// We set the device to prefer L1 over shared memory on devices with compute
	// capability 2 or higher, since 1.x doesn't have L1 cache anyway so the setting
	// would be meaningless.
	// On Kepler (3.x) we prefer shared memory instead because on that architecture
	// L1 is only used for register spills (not global memory), and we are too
	// constrained by the limited amount of shared memory.
	if (deviceProp.major >= 2) {
		cudaFuncCache cacheConfig = cudaFuncCachePreferL1;
		if (deviceProp.major == 3)
			cacheConfig = cudaFuncCachePreferShared;
		CUDA_SAFE_CALL_NOSYNC(cudaDeviceSetCacheConfig(cacheConfig));
	}

	return deviceProp;
}
