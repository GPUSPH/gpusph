#include "backend_select.opt"
#include <cstdio>

#if CUDA_BACKEND_ENABLED
#include <cuda_runtime_api.h>
#endif

int main(int, char *[]) {
	int numDevs = 0;

#if CUDA_BACKEND_ENABLED
	cudaDeviceProp dev_prop;

	if (cudaGetDeviceCount(&numDevs) != cudaSuccess)
		return 1;

	for (int d = 0; d < numDevs; ++d) {
		if (cudaGetDeviceProperties(&dev_prop, 0) != cudaSuccess)
			return d+2;
		printf("%d\t%d.%d\t%s\n", d, dev_prop.major, dev_prop.minor, dev_prop.name);
	}
#else
	printf("%d\t%d.%d\t%s\n", 0, 0, 0, "CPU");
#endif

	return 0;
}
