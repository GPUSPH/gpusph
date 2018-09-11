
void
GenericProblem::initializeParticles(BufferList &buffers, const uint numParticles)
{
	// 1. warn the user if this is expected to take much time
	printf("Performing advanced user initialisation...\n");

	// 2. grab the particle arrays from the buffer list
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();
	double4 *pos_global = buffers.getData<BUFFER_POS_GLOBAL>();
	float *k = buffers.getData<BUFFER_TKE>();
	float *epsilon = buffers.getData<BUFFER_EPSILON>();

	// 3. iterate on the particles
	for (uint i = 0; i < numParticles; i++) {
		const float Ti = 0.01f;
		const float u = 1.0f;
		const float L = 1.0f;

		// Example of k and epsilon initialisation
		if (k && epsilon) {
			k[i] = fmaxf(1e-5f, 3.0f/2.0f*(u*Ti)*(u*Ti));
			epsilon[i] = fmaxf(1e-5f, 2.874944542f*k[i]*u*Ti/L);
		}
		// See src/problems/Bubble.cu for an example of mass initialisation
	}
}
