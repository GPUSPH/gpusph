#include <stdio.h>

#include <cuda_runtime_api.h>

int main(int, char *[]) {
	int numDevs = 0;
	cudaDeviceProp dev_prop;

	if (cudaGetDeviceCount(&numDevs) != cudaSuccess)
		return 1;

	for (int d = 0; d < numDevs; ++d) {
		if (cudaGetDeviceProperties(&dev_prop, 0) != cudaSuccess)
			return d+2;
		printf("%d\t%d.%d\t%s\n", d, dev_prop.major, dev_prop.minor, dev_prop.name);
	}

	return 0;
}
