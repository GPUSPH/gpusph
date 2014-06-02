#include <cuda.h>
#include <stdio.h>

int main(int, char *[]) {
	int numDevs = 0;
#define BUFSIZE 8192
	char namebuf[BUFSIZE];

	cuInit(0);

	if (cuDeviceGetCount(&numDevs) != CUDA_SUCCESS)
		return 1;
	for (int d = 0; d < numDevs; ++d) {
		CUdevice dev;
		int ccmaj, ccmin;
		if (cuDeviceGet(&dev, d) != CUDA_SUCCESS)
			return d+2;
		if (cuDeviceGetName(namebuf, BUFSIZE, dev) != CUDA_SUCCESS)
			return 250;
#if CUDA_VERSION < 5000
		if (cuDeviceComputeCapability(&ccmaj, &ccmin, dev) != CUDA_SUCCESS)
			return 251;
#else
		if (cuDeviceGetAttribute(&ccmaj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) != CUDA_SUCCESS)
			return 251;
		if (cuDeviceGetAttribute(&ccmin, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) != CUDA_SUCCESS)
			return 252;
#endif
		printf("%d\t%d.%d\t%s\n", d, ccmaj, ccmin, namebuf);
	}

	return 0;
}
