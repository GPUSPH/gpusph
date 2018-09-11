namespace cuGenericProblem
{
	using namespace cuforces;
	using namespace cubounds;

	__device__
		void
		GenericProblem_imposeBoundaryCondition(
				const	particleinfo	info,
				const	float3			absPos,
				float			waterdepth,
				const	float			t,
				float4&			vel,
				float4&			eulerVel,
				float&			tke,
				float&			eps)
		{
			// Default value for eulerVel
			// Note that this default value needs to be physically feasible, as it is used in case of boundary elements
			// without fluid particles in their support. It is also possible to use this default value to impose tangential
			// velocities for pressure outlets.
			eulerVel = make_float4(0.0f, 0.0f, 0.0f, d_rho0[fluid_num(info)]);
			vel = make_float4(0.0f);
			tke = 0.0f;
			eps = 0.0f;

			// open boundary conditions
			if (IO_BOUNDARY(info)) {
				if (!VEL_IO(info)) {
					// set waterdepth
					IMPOSE_WATER_LEVEL
					const float localdepth = fmaxf(waterdepth - absPos.z, 0.0f);
					const float pressure = 9.81e3f*localdepth;
					eulerVel.w = RHO(pressure, fluid_num(info));
				} else {
					IMPOSE_VELOCITY
				}
			}
		}

	__global__ void
		GenericProblem_imposeBoundaryConditionDevice(
				float4*		newVel,
				float4*		newEulerVel,
				float*		newTke,
				float*		newEpsilon,
				const	float4*		oldPos,
				const	uint*		IOwaterdepth,
				const	float		t,
				const	uint		numParticles,
				const	hashKey*	particleHash)
		{
			const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

			if (index >= numParticles)
				return;

			float4 vel = make_float4(0.0f);			// imposed velocity for moving objects
			float4 eulerVel = make_float4(0.0f);	// imposed velocity/pressure for open boundaries
			float tke = 0.0f;						// imposed turbulent kinetic energy for open boundaries
			float eps = 0.0f;						// imposed turb. diffusivity for open boundaries

			// Open boundary conditions
			if(index < numParticles) {
				const particleinfo info = tex1Dfetch(infoTex, index);
				// The case of a corner needs to be treated as follows:
				// - for a velocity inlet nothing is imposed (in case of k-eps newEulerVel already contains the info
				//   from the viscosity
				// - for a pressure inlet the pressure is imposed on the corners. If we are in the k-epsilon case then
				//   we need to get the viscosity info from newEulerVel (x,y,z) and add the imposed density in .w
				if ((VERTEX(info) || BOUNDARY(info)) && IO_BOUNDARY(info) && (!CORNER(info) || !VEL_IO(info))) {
					// For corners we need to get eulerVel in case of k-eps and pressure outlet
					if (CORNER(info) && newTke && !VEL_IO(info)) {
						eulerVel = newEulerVel[index];
					}
					const float3 absPos = d_worldOrigin + as_float3(oldPos[index])
						+ calcGridPosFromParticleHash(particleHash[index])*d_cellSize
						+ 0.5f*d_cellSize;
					float waterdepth = 0.0f;
					if (!VEL_IO(info) && IOwaterdepth) {
						waterdepth = ((float)IOwaterdepth[object(info)])/((float)UINT_MAX); // now between 0 and 1
						waterdepth *= d_cellSize.z*d_gridSize.z; // now between 0 and world size
						waterdepth += d_worldOrigin.z; // now absolute z position
					}
					// this now calls the virtual function that is problem specific
					GenericProblem_imposeBoundaryCondition(info, absPos, waterdepth, t, vel, eulerVel, tke, eps);
					// copy values to arrays
					newVel[index] = vel;
					newEulerVel[index] = eulerVel;
					if(newTke)
						newTke[index] = tke;
					if(newEpsilon)
						newEpsilon[index] = eps;
				}
			}
		}

} // end of cuGenericProblem namespace

	void
GenericProblem::imposeBoundaryConditionHost(
		MultiBufferList::iterator		bufwrite,
		MultiBufferList::const_iterator	bufread,
		uint*			IOwaterdepth,
		const	float			t,
		const	uint			numParticles,
		const	uint			numOpenBoundaries,
		const	uint			particleRangeEnd)
{
	float4	*newVel = bufwrite->getData<BUFFER_VEL>();
	float4	*newEulerVel = bufwrite->getData<BUFFER_EULERVEL>();
	float	*newTke = bufwrite->getData<BUFFER_TKE>();
	float	*newEpsilon = bufwrite->getData<BUFFER_EPSILON>();

	const particleinfo *info = bufread->getData<BUFFER_INFO>();
	const float4 *oldPos = bufread->getData<BUFFER_POS>();
	const hashKey *particleHash = bufread->getData<BUFFER_HASH>();

	const uint numThreads = min(BLOCK_SIZE_IOBOUND, particleRangeEnd);
	const uint numBlocks = div_up(particleRangeEnd, numThreads);

	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	int dummy_shared = 0;
#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
#endif

	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	cuGenericProblem::GenericProblem_imposeBoundaryConditionDevice<<< numBlocks, numThreads, dummy_shared >>>
		(newVel, newEulerVel, newTke, newEpsilon, oldPos, IOwaterdepth, t, numParticles, particleHash);

	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));

	// reset waterdepth calculation
	if (IOwaterdepth) {
		uint h_IOwaterdepth[numOpenBoundaries];
		for (uint i=0; i<numOpenBoundaries; i++)
			h_IOwaterdepth[i] = 0;
		CUDA_SAFE_CALL(cudaMemcpy(IOwaterdepth, h_IOwaterdepth, numOpenBoundaries*sizeof(int), cudaMemcpyHostToDevice));
	}

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}
