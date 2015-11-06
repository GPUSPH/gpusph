/*  Copyright 2011-2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

    Istituto Nazionale di Geofisica e Vulcanologia
        Sezione di Catania, Catania, Italy

    Università di Catania, Catania, Italy

    Johns Hopkins University, Baltimore, MD

    This file is part of GPUSPH.

    GPUSPH is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    GPUSPH is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with GPUSPH.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <stdexcept>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/functional.h>

#include "textures.cuh"

#include "engine_forces.h"
#include "engine_visc.h"
#include "engine_filter.h"
#include "simflags.h"

#include "utils.h"
#include "cuda_call.h"

#include "define_buffers.h"

#include "forces_params.h"

/* Important notes on block sizes:
	- a parallel reduction for adaptive dt is done inside forces, block
	size for forces MUST BE A POWER OF 2
 */
#if (__COMPUTE__ >= 20)
	#define BLOCK_SIZE_FORCES		128
	#define BLOCK_SIZE_CALCVORT		128
	#define MIN_BLOCKS_CALCVORT		6
	#define BLOCK_SIZE_CALCTEST		128
	#define MIN_BLOCKS_CALCTEST		6
	#define BLOCK_SIZE_SHEPARD		128
	#define MIN_BLOCKS_SHEPARD		6
	#define BLOCK_SIZE_MLS			128
	#define MIN_BLOCKS_MLS			6
	#define BLOCK_SIZE_SPS			128
	#define MIN_BLOCKS_SPS			6
	#define BLOCK_SIZE_FMAX			256
	#define MAX_BLOCKS_FMAX			64
#else
	#define BLOCK_SIZE_FORCES		64
	#define BLOCK_SIZE_CALCVORT		128
	#define MIN_BLOCKS_CALCVORT		1
	#define BLOCK_SIZE_CALCTEST		128
	#define MIN_BLOCKS_CALCTEST		1
	#define BLOCK_SIZE_SHEPARD		224
	#define MIN_BLOCKS_SHEPARD		1
	#define BLOCK_SIZE_MLS			128
	#define MIN_BLOCKS_MLS			1
	#define BLOCK_SIZE_SPS			128
	#define MIN_BLOCKS_SPS			1
	#define BLOCK_SIZE_FMAX			256
	#define MAX_BLOCKS_FMAX			64
#endif


cudaArray*  dDem = NULL;

/* Auxiliary data for parallel reductions */
size_t	reduce_blocks = 0;
size_t	reduce_blocksize_max = 0;
size_t	reduce_bs2 = 0;
size_t	reduce_shmem_max = 0;
void*	reduce_buffer = NULL;

#include "forces_kernel.cu"

/// static inline methods for fmax reduction

static inline void
reducefmax(	const int	size,
			const int	threads,
			const int	blocks,
			float		*d_idata,
			float		*d_odata)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

	switch (threads)
	{
		case 512:
			cuforces::fmaxDevice<512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 256:
			cuforces::fmaxDevice<256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 128:
			cuforces::fmaxDevice<128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 64:
			cuforces::fmaxDevice<64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 32:
			cuforces::fmaxDevice<32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 16:
			cuforces::fmaxDevice<16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  8:
			cuforces::fmaxDevice<8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  4:
			cuforces::fmaxDevice<4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  2:
			cuforces::fmaxDevice<2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  1:
			cuforces::fmaxDevice<1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
	}
}


static inline uint nextPow2(uint x )
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}


#define MIN(x,y) ((x < y) ? x : y)
static inline void
getNumBlocksAndThreads(	const uint	n,
						const uint	maxBlocks,
						const uint	maxThreads,
						uint		&blocks,
						uint		&threads)
{
	threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
	blocks = (n + (threads * 2 - 1)) / (threads * 2);
	blocks = MIN(maxBlocks, blocks);
}

static inline float
cflmax( const uint	n,
		float*		cfl,
		float*		tempCfl)
{
	uint numBlocks = 0;
	uint numThreads = 0;
	float max = 0.0f;

	getNumBlocksAndThreads(n, MAX_BLOCKS_FMAX, BLOCK_SIZE_FMAX, numBlocks, numThreads);

	// execute the kernel
	reducefmax(n, numThreads, numBlocks, cfl, tempCfl);

	// check if kernel execution generated an error
	KERNEL_CHECK_ERROR;

	// TODO this can be done in just two calls
	uint s = numBlocks;
	while(s > 1)
	{
		uint threads = 0, blocks = 0;
		getNumBlocksAndThreads(s, MAX_BLOCKS_FMAX, BLOCK_SIZE_FMAX, blocks, threads);

		reducefmax(s, threads, blocks, tempCfl, tempCfl);
		KERNEL_CHECK_ERROR;

		s = (s + (threads*2-1)) / (threads*2);
	}

	CUDA_SAFE_CALL(cudaMemcpy(&max, tempCfl, sizeof(float), cudaMemcpyDeviceToHost));

	return max;
}

// CUDAForcesEngine. Some methods need helper classes, which are defined below,
// right before the actual class definition starts
template<
	KernelType kerneltype,
	SPHFormulation sph_formulation,
	ViscosityType visctype,
	BoundaryType boundarytype,
	flag_t simflags>
class CUDAForcesEngine;

/// Density computation is a no-op in all cases but Grenier's. Since C++ does not
/// allow partial template specialization for methods, we rely on a CUDADensityHelper
/// auxiliary functor, that we can re-define with partial specialization as needed.

/// General case: do nothing
template<
	KernelType kerneltype,
	SPHFormulation sph_formulation,
	BoundaryType boundarytype>
struct CUDADensityHelper {
	static void
	process(MultiBufferList::const_iterator bufread,
		MultiBufferList::iterator bufwrite,
		const uint *cellStart,
		const uint numParticles,
		float slength,
		float influenceradius)
	{ /* do nothing by default */ }
};

/// Grenier
template<
	KernelType kerneltype,
	BoundaryType boundarytype>
struct CUDADensityHelper<kerneltype, SPH_GRENIER, boundarytype> {
	static void
	process(MultiBufferList::const_iterator bufread,
		MultiBufferList::iterator bufwrite,
		const uint *cellStart,
		const uint numParticles,
		float slength,
		float influenceradius)
	{
		uint numThreads = BLOCK_SIZE_FORCES;
		uint numBlocks = div_up(numParticles, numThreads);

		const float4 *pos = bufread->getData<BUFFER_POS>();
		const float4 *vol = bufread->getData<BUFFER_VOLUME>();
		const particleinfo *info = bufread->getData<BUFFER_INFO>();
		const hashKey *pHash = bufread->getData<BUFFER_HASH>();
		const neibdata *neibsList = bufread->getData<BUFFER_NEIBSLIST>();

		/* Update WRITE vel in place, caller should do a swap before and after */
		float4 *vel = bufwrite->getData<BUFFER_VEL>();
		float *sigma = bufwrite->getData<BUFFER_SIGMA>();

		cuforces::densityGrenierDevice<kerneltype, boundarytype>
			<<<numBlocks, numThreads>>>(sigma, pos, vel, info, pHash, vol, cellStart, neibsList, numParticles, slength, influenceradius);

		// check if kernel invocation generated an error
		KERNEL_CHECK_ERROR;
	}
};


/// CUDAForcesEngine

template<
	KernelType kerneltype,
	SPHFormulation sph_formulation,
	ViscosityType visctype,
	BoundaryType boundarytype,
	flag_t simflags>
class CUDAForcesEngine : public AbstractForcesEngine
{

static const bool needs_eulerVel = (boundarytype == SA_BOUNDARY &&
			(visctype == KEPSVISC || (simflags & ENABLE_INLET_OUTLET)));


void
setconstants(const SimParams *simparams, const PhysParams *physparams,
	float3 const& worldOrigin, uint3 const& gridSize, float3 const& cellSize,
	idx_t const& allocatedParticles)
{
	// Setting kernels and kernels derivative factors
	const float h = simparams->slength;
	const float h2 = h*h;
	const float h3 = h2*h;
	const float h4 = h2*h2;
	const float h5 = h4*h;
	float kernelcoeff = 1.0f/(M_PI*h3);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_wcoeff_cubicspline, &kernelcoeff, sizeof(float)));
	kernelcoeff = 15.0f/(16.0f*M_PI*h3);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_wcoeff_quadratic, &kernelcoeff, sizeof(float)));
	kernelcoeff = 21.0f/(16.0f*M_PI*h3);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_wcoeff_wendland, &kernelcoeff, sizeof(float)));
	kernelcoeff = 3.0f/(4.0f*M_PI*h4);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_fcoeff_cubicspline, &kernelcoeff, sizeof(float)));
	kernelcoeff = 15.0f/(32.0f*M_PI*h4);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_fcoeff_quadratic, &kernelcoeff, sizeof(float)));
	kernelcoeff = 105.0f/(128.0f*M_PI*h5);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_fcoeff_wendland, &kernelcoeff, sizeof(float)));

	/*	Gaussian kernel: W(r, h) (exp(-(r/h)^2) - exp(-(δ/h)^2))/const
		with δ cut-off radius (typically, 3h). For us, δ is the influence radius R*h,
		with R kernel radius
	 */
	const float R = simparams->kernelradius;
	const float R2 = R*R; // R2 = squared kernel radius = (δ/h)^2
	const float exp_R2 = exp(-R2); // exp(-(δ/h)^2)
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_wsub_gaussian, &exp_R2, sizeof(float)));
	// constant: pi^(3/2)
#define M_PI_TO_3_2 5.5683279968317078452848179821188357020136243902832439
	// -2/3 exp(-R^2) h^3 Pi R (3 + 2 R^2) + h^3 Pi^(3/2) Erf(R) <- CHECK this
	kernelcoeff = -2*exp_R2/3 * h3 * M_PI * R*(3+2*R2) + h3 * M_PI_TO_3_2 * erf(R);
#undef M_PI_TO_3_2
	kernelcoeff = 1/kernelcoeff;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_wcoeff_gaussian, &kernelcoeff, sizeof(float)));
	// the coefficient for the F is just the W coefficient times 2/h^2
	kernelcoeff *= 2/h2;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_fcoeff_gaussian, &kernelcoeff, sizeof(float)));

	const int numFluids = physparams->numFluids();
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_numfluids, &numFluids, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_rho0, &physparams->rho0[0], numFluids*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_bcoeff, &physparams->bcoeff[0], numFluids*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_gammacoeff, &physparams->gammacoeff[0], numFluids*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_sscoeff, &physparams->sscoeff[0], numFluids*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_sspowercoeff, &physparams->sspowercoeff[0], numFluids*sizeof(float)));

	// compute (and upload) square of sound speeds, needed for Ferrari
	float sqC0[numFluids];
	for (uint i = 0; i < numFluids; ++i) {
		sqC0[i]  = physparams->sscoeff[i];
		sqC0[i] *= sqC0[i];
	}
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_sqC0, sqC0, numFluids*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_visccoeff, &physparams->visccoeff[0], numFluids*sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_gravity, &physparams->gravity, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_dcoeff, &physparams->dcoeff, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_p1coeff, &physparams->p1coeff, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_p2coeff, &physparams->p2coeff, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_MK_K, &physparams->MK_K, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_MK_d, &physparams->MK_d, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_MK_beta, &physparams->MK_beta, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_r0, &physparams->r0, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_epsartvisc, &physparams->epsartvisc, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cubounds::d_ewres, &physparams->ewres, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cubounds::d_nsres, &physparams->nsres, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cubounds::d_demdx, &physparams->demdx, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cubounds::d_demdy, &physparams->demdy, sizeof(float)));
	float demdxdy = physparams->demdx*physparams->demdy;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cubounds::d_demdxdy, &demdxdy, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cubounds::d_demzmin, &physparams->demzmin, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_smagfactor, &physparams->smagfactor, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_kspsfactor, &physparams->kspsfactor, sizeof(float)));

	float partsurf = physparams->partsurf;
	if (partsurf == 0.0f)
		partsurf = physparams->r0*physparams->r0;
		// partsurf = (6.0 - M_PI)*physparams->r0*physparams->r0/4;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_partsurf, &partsurf, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_objectobjectdf, &physparams->objectobjectdf, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_objectboundarydf, &physparams->objectboundarydf, sizeof(float)));

	idx_t neiblist_end = simparams->maxneibsnum*allocatedParticles;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_neiblist_stride, &allocatedParticles, sizeof(idx_t)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_neiblist_end, &neiblist_end, sizeof(idx_t)));

	// Neibs cell to offset table
	char3 cell_to_offset[27];
	for(char z=-1; z<=1; z++) {
		for(char y=-1; y<=1; y++) {
			for(char x=-1; x<=1; x++) {
				int i = (x + 1) + (y + 1)*3 + (z + 1)*9;
				cell_to_offset[i] =  make_char3(x, y, z);
			}
		}
	}
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cubounds::d_cell_to_offset, cell_to_offset, 27*sizeof(char3)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cubounds::d_worldOrigin, &worldOrigin, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cubounds::d_gridSize, &gridSize, sizeof(uint3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cubounds::d_cellSize, &cellSize, sizeof(float3)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_ferrari, &simparams->ferrari, sizeof(float)));

	const float rhodiffcoeff = simparams->rhodiffcoeff*2*simparams->slength;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_rhodiffcoeff, &rhodiffcoeff, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_epsinterface, &physparams->epsinterface, sizeof(float)));
}


void
getconstants(PhysParams *physparams)
{
	int numFluids = -1;
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&numFluids, cuforces::d_numfluids, sizeof(int)));
	if (numFluids != physparams->numFluids())
		throw std::runtime_error("wrong number of fluids");
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->visccoeff[0], cuforces::d_visccoeff, numFluids*sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->rho0[0], cuforces::d_rho0, numFluids*sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->bcoeff[0], cuforces::d_bcoeff, numFluids*sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->gammacoeff[0], cuforces::d_gammacoeff, numFluids*sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->sscoeff[0], cuforces::d_sscoeff, numFluids*sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->sspowercoeff[0], cuforces::d_sspowercoeff, numFluids*sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->gravity, cuforces::d_gravity, sizeof(float3), 0));

	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->dcoeff, cuforces::d_dcoeff, sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->p1coeff, cuforces::d_p1coeff, sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->p2coeff, cuforces::d_p2coeff, sizeof(float), 0));

	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->MK_K, cuforces::d_MK_K, sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->MK_d, cuforces::d_MK_d, sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->MK_beta, cuforces::d_MK_beta, sizeof(float), 0));

	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->r0, cuforces::d_r0, sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->epsartvisc, cuforces::d_epsartvisc, sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->ewres, cuforces::d_ewres, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->nsres, cuforces::d_nsres, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->demdx, cuforces::d_demdx, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->demdy, cuforces::d_demdy, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->demzmin, cuforces::d_demzmin, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->smagfactor, cuforces::d_smagfactor, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->kspsfactor, cuforces::d_kspsfactor, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_epsinterface, &physparams->epsinterface, sizeof(float)));
}

void
setplanes(std::vector<plane_t> const& planes)
{
	uint numPlanes = planes.size();
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cubounds::d_numplanes, &numPlanes, sizeof(uint)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cubounds::d_plane, &planes[0], numPlanes*sizeof(plane_t)));
}

void
setgravity(float3 const& gravity)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_gravity, &gravity, sizeof(float3)));
}

void
setrbcg(const int3* cgGridPos, const float3* cgPos, int numbodies)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_rbcgGridPos, cgGridPos, numbodies*sizeof(int3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_rbcgPos, cgPos, numbodies*sizeof(float3)));
}

void
setrbstart(const int* rbfirstindex, int numbodies)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_rbstartindex, rbfirstindex, numbodies*sizeof(int)));
}

void
bind_textures(
	MultiBufferList::const_iterator bufread,
	uint	numParticles)
{
	// bind textures to read all particles, not only internal ones
	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, bufread->getData<BUFFER_POS>(), numParticles*sizeof(float4)));
	#endif
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, bufread->getData<BUFFER_VEL>(), numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, bufread->getData<BUFFER_INFO>(), numParticles*sizeof(particleinfo)));

	const float4 *eulerVel = bufread->getData<BUFFER_EULERVEL>();
	if (needs_eulerVel) {
		if (!eulerVel)
			throw std::invalid_argument("eulerVel not set but needed");
		CUDA_SAFE_CALL(cudaBindTexture(0, eulerVelTex, eulerVel, numParticles*sizeof(float4)));
	} else {
		if (eulerVel)
			cerr << "eulerVel set but not used" << endl;
	}

	if (boundarytype == SA_BOUNDARY) {
		CUDA_SAFE_CALL(cudaBindTexture(0, gamTex, bufread->getData<BUFFER_GRADGAMMA>(), numParticles*sizeof(float4)));
		CUDA_SAFE_CALL(cudaBindTexture(0, boundTex, bufread->getData<BUFFER_BOUNDELEMENTS>(), numParticles*sizeof(float4)));
	}

	if (visctype == KEPSVISC) {
		CUDA_SAFE_CALL(cudaBindTexture(0, keps_kTex, bufread->getData<BUFFER_TKE>(), numParticles*sizeof(float)));
		CUDA_SAFE_CALL(cudaBindTexture(0, keps_eTex, bufread->getData<BUFFER_EPSILON>(), numParticles*sizeof(float)));
	}
}

void
unbind_textures()
{
	// TODO FIXME why are SPS textures unbound here but bound in sps?
	// shouldn't we bind them in bind_textures() instead?
	if (visctype == SPSVISC) {
		CUDA_SAFE_CALL(cudaUnbindTexture(tau0Tex));
		CUDA_SAFE_CALL(cudaUnbindTexture(tau1Tex));
		CUDA_SAFE_CALL(cudaUnbindTexture(tau2Tex));
	}

	if (visctype == KEPSVISC) {
		CUDA_SAFE_CALL(cudaUnbindTexture(keps_kTex));
		CUDA_SAFE_CALL(cudaUnbindTexture(keps_eTex));
	}

	if (boundarytype == SA_BOUNDARY) {
		CUDA_SAFE_CALL(cudaUnbindTexture(gamTex));
		CUDA_SAFE_CALL(cudaUnbindTexture(boundTex));
	}

	if (needs_eulerVel)
		CUDA_SAFE_CALL(cudaUnbindTexture(eulerVelTex));

	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	#endif
}

// returns the number of elements in the (starting) fmax array, assuming n particles.
// this is _exactly_ the number of blocks in the grid launch for the forces kernel over n
// particles, since the forces kernel pre-reduces the cfl values, producing one value
// per block instead of one per particle
// TODO FIXME reorganize this reduction stuff
uint
getFmaxElements(const uint n)
{
	return div_up<uint>(n, BLOCK_SIZE_FORCES);
}


uint
getFmaxTempElements(const uint n)
{
	uint numBlocks, numThreads;
	getNumBlocksAndThreads(n, MAX_BLOCKS_FMAX, BLOCK_SIZE_FMAX, numBlocks, numThreads);
	return numBlocks;
}



float
dtreduce(	float	slength,
			float	dtadaptfactor,
			float	max_kinematic,
			float	*cfl,
			float	*cfl_dS,
			float	*cflTVisc,
			float	*tempCfl,
			uint	numBlocks)
{
	// cfl holds one value per block in the forces kernel call,
	// so it holds numBlocks elements
	float maxcfl = cflmax(numBlocks, cfl, tempCfl);
	float dt = dtadaptfactor*sqrtf(slength/maxcfl);

	if(simflags & ENABLE_DENSITY_SUM) {
		maxcfl = fmax(cflmax(numBlocks, cfl_dS, tempCfl), 1e-5f/dt);
		const float dt_gam = 0.001f/maxcfl;
		if (dt_gam < dt)
			dt = dt_gam;
	}

	if (visctype != ARTVISC) {
		/* Stability condition from viscosity h²/ν
		   We get the maximum kinematic viscosity from the caller, and in the KEPS case we
		   add the maximum KEPS
		 */
		float visccoeff = max_kinematic;
		if (visctype == KEPSVISC)
			visccoeff += cflmax(numBlocks, cflTVisc, tempCfl);

		float dt_visc = slength*slength/visccoeff;
		dt_visc *= 0.125; // TODO allow customization
		if (dt_visc < dt)
			dt = dt_visc;
	}

	// check if last kernel invocation generated an error
	KERNEL_CHECK_ERROR;

	return dt;
}

void
compute_density(MultiBufferList::const_iterator bufread,
	MultiBufferList::iterator bufwrite,
	const uint *cellStart,
	uint numParticles,
	float slength,
	float influenceradius)
{
	CUDADensityHelper<kerneltype, sph_formulation, boundarytype>::process(bufread,
		bufwrite, cellStart, numParticles, slength, influenceradius);
	return;
}

// Returns numBlock for delayed dt reduction in case of striping
uint
basicstep(
	MultiBufferList::const_iterator bufread,
	MultiBufferList::iterator bufwrite,
	float4	*rbforces,
	float4	*rbtorques,
	const	uint	*cellStart,
	uint	numParticles,
	uint	fromParticle,
	uint	toParticle,
	float	deltap,
	float	slength,
	float	dtadaptfactor,
	float	influenceradius,
	const	float	epsilon,
	uint	*IOwaterdepth,
	uint	cflOffset,
	const	uint	step)
{
	const float4 *pos = bufread->getData<BUFFER_POS>();
	const float4 *vel = bufread->getData<BUFFER_VEL>();
	const particleinfo *info = bufread->getData<BUFFER_INFO>();
	const hashKey *particleHash = bufread->getData<BUFFER_HASH>();
	const neibdata *neibsList = bufread->getData<BUFFER_NEIBSLIST>();

	const float2 * const *vertPos = bufread->getRawPtr<BUFFER_VERTPOS>();
	const float4 *oldGGam = bufread->getData<BUFFER_GRADGAMMA>();
	const float4 *boundelem = bufread->getData<BUFFER_BOUNDELEMENTS>();

	float4 *forces = bufwrite->getData<BUFFER_FORCES>();
	float4 *xsph = bufwrite->getData<BUFFER_XSPH>();
	float2 *contupd = bufwrite->getData<BUFFER_CONTUPD>();
	float4 *newGGam = bufwrite->getData<BUFFER_GRADGAMMA>();

	// TODO FIXME TURBVISC, TKE, EPSILON are in/out, but they are taken from the READ position
	float *turbvisc = const_cast<float*>(bufread->getData<BUFFER_TURBVISC>());
	float *keps_tke = const_cast<float*>(bufread->getData<BUFFER_TKE>());
	float *keps_eps = const_cast<float*>(bufread->getData<BUFFER_EPSILON>());

	float3 *keps_dkde = bufwrite->getData<BUFFER_DKDE>();
	float *cfl = bufwrite->getData<BUFFER_CFL>();
	float *cfl_Ds = bufwrite->getData<BUFFER_CFL_DS>();
	float *cflTVisc = bufwrite->getData<BUFFER_CFL_KEPS>();
	float *tempCfl = bufwrite->getData<BUFFER_CFL_TEMP>();

	int dummy_shared = 0;

	const uint numParticlesInRange = toParticle - fromParticle;
	// thread per particle
	uint numThreads = BLOCK_SIZE_FORCES;
	uint numBlocks = div_up(numParticlesInRange, numThreads);
	#if (__COMPUTE__ == 20)
	int dtadapt = !!(simflags & ENABLE_DTADAPT);
	if (visctype == SPSVISC)
		dummy_shared = 3328 - dtadapt*BLOCK_SIZE_FORCES*4;
	else
		dummy_shared = 2560 - dtadapt*BLOCK_SIZE_FORCES*4;
	#endif

	forces_params<kerneltype, sph_formulation, boundarytype, visctype, simflags> params(
			forces, rbforces, rbtorques,
			pos, particleHash, cellStart, neibsList, fromParticle, toParticle,
			deltap, slength, influenceradius, step,
			cfl, cfl_Ds, cflTVisc, cflOffset,
			xsph,
			bufread->getData<BUFFER_VOLUME>(),
			bufread->getData<BUFFER_SIGMA>(),
			newGGam, contupd, vertPos, epsilon,
			IOwaterdepth,
			keps_dkde, turbvisc);

	// FIXME forcesDevice should use simflags, not the neverending pile of booleans
	cuforces::forcesDevice<kerneltype, sph_formulation, boundarytype, visctype, simflags>
			<<< numBlocks, numThreads, dummy_shared >>>(params);

	return numBlocks;
}

void
setDEM(const float *hDem, int width, int height)
{
	// Allocating, reading and copying DEM
	unsigned int size = width*height*sizeof(float);
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	CUDA_SAFE_CALL( cudaMallocArray( &dDem, &channelDesc, width, height ));
	CUDA_SAFE_CALL( cudaMemcpyToArray( dDem, 0, 0, hDem, size, cudaMemcpyHostToDevice));

	cubounds::demTex.addressMode[0] = cudaAddressModeClamp;
	cubounds::demTex.addressMode[1] = cudaAddressModeClamp;
	cubounds::demTex.filterMode = cudaFilterModeLinear;
	cubounds::demTex.normalized = false;

	CUDA_SAFE_CALL( cudaBindTextureToArray(cubounds::demTex, dDem, channelDesc));
}

void
unsetDEM()
{
	CUDA_SAFE_CALL(cudaFreeArray(dDem));
}

uint
round_particles(uint numparts)
{
	return (numparts/BLOCK_SIZE_FORCES)*BLOCK_SIZE_FORCES;
}

void
reduceRbForces(	float4	*forces,
				float4	*torques,
				uint	*rbnum,
				uint	*lastindex,
				float3	*totalforce,
				float3	*totaltorque,
				uint	numforcesbodies,
				uint	numForcesBodiesParticles)
{
	thrust::device_ptr<float4> forces_devptr = thrust::device_pointer_cast(forces);
	thrust::device_ptr<float4> torques_devptr = thrust::device_pointer_cast(torques);
	thrust::device_ptr<uint> rbnum_devptr = thrust::device_pointer_cast(rbnum);
	thrust::equal_to<uint> binary_pred;
	thrust::plus<float4> binary_op;

	// For the segmented scan, we use rbnum (number of object per object particle) as key (first and second parameters
	// of inclusive_scan_by_key are the begin and the end of the array of keys); forces or torques as input and output
	// the scan is in place); equal_to as data-key operator and plus as scan operator. The sums are in the last position
	// of each segment (thus we retrieve them by using lastindex values).

	thrust::inclusive_scan_by_key(rbnum_devptr, rbnum_devptr + numForcesBodiesParticles,
				forces_devptr, forces_devptr, binary_pred, binary_op);
	thrust::inclusive_scan_by_key(rbnum_devptr, rbnum_devptr + numForcesBodiesParticles,
				torques_devptr, torques_devptr, binary_pred, binary_op);

	for (uint i = 0; i < numforcesbodies; i++) {
		float4 temp;
		void * ddata = (void *) (forces + lastindex[i]);
		CUDA_SAFE_CALL(cudaMemcpy((void *) &temp, ddata, sizeof(float4), cudaMemcpyDeviceToHost));
		totalforce[i] = as_float3(temp);

		ddata = (void *) (torques + lastindex[i]);
		CUDA_SAFE_CALL(cudaMemcpy((void *) &temp, ddata, sizeof(float4), cudaMemcpyDeviceToHost));
		totaltorque[i] = as_float3(temp);
		}
}

};

/// CUDAViscEngine class. Should be moved into its own source file
///
/// Generally, the kernel and boundary type will be passed through to the
/// process() to call the appropriate kernels, and the main selector would be
/// just the ViscosityType. We cannot have partial function/method template
/// specialization, so our CUDAViscEngine actually delegates to a helper functor,
/// which should be partially specialized as a whole class

template<ViscosityType visctype,
	KernelType kerneltype,
	BoundaryType boundarytype>
class CUDAViscEngine;

template<ViscosityType visctype,
	KernelType kerneltype,
	BoundaryType boundarytype>
struct CUDAViscEngineHelper
{
	static void
	process(		float2	*tau[],
					float	*turbvisc,
			const	float4	*pos,
			const	float4	*vel,
			const	particleinfo	*info,
			const	hashKey	*particleHash,
			const	uint	*cellStart,
			const	neibdata*neibsList,
					uint	numParticles,
					uint	particleRangeEnd,
					float	slength,
					float	influenceradius);
};


template<ViscosityType visctype,
	KernelType kerneltype,
	BoundaryType boundarytype>
void
CUDAViscEngineHelper<visctype, kerneltype, boundarytype>::process(
			float2	*tau[],
			float	*turbvisc,
	const	float4	*pos,
	const	float4	*vel,
	const	particleinfo	*info,
	const	hashKey	*particleHash,
	const	uint	*cellStart,
	const	neibdata*neibsList,
			uint	numParticles,
			uint	particleRangeEnd,
			float	slength,
			float	influenceradius)
{ /* default, does nothing */ }

/// Partial specialization for SPSVISC. Partial specializations
/// redefine the whole helper struct, not just the method, since
/// C++ does not allow partial function/method template specializations
/// (which is why we have the Helper struct in the first place
template<KernelType kerneltype,
	BoundaryType boundarytype>
struct CUDAViscEngineHelper<SPSVISC, kerneltype, boundarytype>
{
	static void
	process(		float2	*tau[],
					float	*turbvisc,
			const	float4	*pos,
			const	float4	*vel,
			const	particleinfo	*info,
			const	hashKey	*particleHash,
			const	uint	*cellStart,
			const	neibdata*neibsList,
					uint	numParticles,
					uint	particleRangeEnd,
					float	slength,
					float	influenceradius)
{
	int dummy_shared = 0;
	// bind textures to read all particles, not only internal ones
	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	#endif
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, vel, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	uint numThreads = BLOCK_SIZE_SPS;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	sps_params<kerneltype, boundarytype, (SPSK_STORE_TAU | SPSK_STORE_TURBVISC)> params(
			pos, particleHash, cellStart, neibsList, numParticles, slength, influenceradius,
			tau[0], tau[1], tau[2], turbvisc);

	cuforces::SPSstressMatrixDevice<kerneltype, boundarytype, (SPSK_STORE_TAU | SPSK_STORE_TURBVISC)>
		<<<numBlocks, numThreads, dummy_shared>>>(params);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;

	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	#endif

	CUDA_SAFE_CALL(cudaBindTexture(0, tau0Tex, tau[0], numParticles*sizeof(float2)));
	CUDA_SAFE_CALL(cudaBindTexture(0, tau1Tex, tau[1], numParticles*sizeof(float2)));
	CUDA_SAFE_CALL(cudaBindTexture(0, tau2Tex, tau[2], numParticles*sizeof(float2)));
}
};

/// Actual CUDAVicEngine
template<ViscosityType visctype,
	KernelType kerneltype,
	BoundaryType boundarytype>
class CUDAViscEngine : public AbstractViscEngine
{
	// TODO when we will be in a separate namespace from forces
	void setconstants() {}
	void getconstants() {}

	void
	process(		float2	*tau[],
					float	*turbvisc,
			const	float4	*pos,
			const	float4	*vel,
			const	particleinfo	*info,
			const	hashKey	*particleHash,
			const	uint	*cellStart,
			const	neibdata*neibsList,
					uint	numParticles,
					uint	particleRangeEnd,
					float	slength,
					float	influenceradius)
	{
		CUDAViscEngineHelper<visctype, kerneltype, boundarytype>::process
		(tau, turbvisc, pos, vel, info, particleHash, cellStart, neibsList, numParticles,
		 particleRangeEnd, slength, influenceradius);
	}

};

/// Preprocessing engines (Shepard, MLS)

// As with the viscengine, we need a helper struct for the partial
// specialization of process
template<FilterType filtertype, KernelType kerneltype, BoundaryType boundarytype>
struct CUDAFilterEngineHelper
{
	static void process(
		const	float4	*pos,
		const	float4	*oldVel,
				float4	*newVel,
		const	particleinfo	*info,
		const	hashKey	*particleHash,
		const	uint	*cellStart,
		const	neibdata*neibsList,
				uint	numParticles,
				uint	particleRangeEnd,
				float	slength,
				float	influenceradius);
};

/* Shepard Filter specialization */
template<KernelType kerneltype, BoundaryType boundarytype>
struct CUDAFilterEngineHelper<SHEPARD_FILTER, kerneltype, boundarytype>
{
	static void process(
		const	float4	*pos,
		const	float4	*oldVel,
				float4	*newVel,
		const	particleinfo	*info,
		const	hashKey	*particleHash,
		const	uint	*cellStart,
		const	neibdata*neibsList,
				uint	numParticles,
				uint	particleRangeEnd,
				float	slength,
				float	influenceradius)
{
	int dummy_shared = 0;
	// thread per particle
	uint numThreads = BLOCK_SIZE_SHEPARD;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	#endif
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, oldVel, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	// execute the kernel
	#if (__COMPUTE__ >= 20)
	dummy_shared = 2560;
	#endif

	cuforces::shepardDevice<kerneltype, boundarytype><<< numBlocks, numThreads, dummy_shared >>>
		(pos, newVel, particleHash, cellStart, neibsList, particleRangeEnd, slength, influenceradius);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;

	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	#endif
	CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
}
};

/* MLS Filter specialization */
template<KernelType kerneltype, BoundaryType boundarytype>
struct CUDAFilterEngineHelper<MLS_FILTER, kerneltype, boundarytype>
{
	static void process(
		const	float4	*pos,
		const	float4	*oldVel,
				float4	*newVel,
		const	particleinfo	*info,
		const	hashKey	*particleHash,
		const	uint	*cellStart,
		const	neibdata*neibsList,
				uint	numParticles,
				uint	particleRangeEnd,
				float	slength,
				float	influenceradius)
{
	int dummy_shared = 0;
	// thread per particle
	uint numThreads = BLOCK_SIZE_MLS;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	#endif
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, oldVel, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	// execute the kernel
	#if (__COMPUTE__ >= 20)
	dummy_shared = 2560;
	#endif

	cuforces::MlsDevice<kerneltype, boundarytype><<< numBlocks, numThreads, dummy_shared >>>
		(pos, newVel, particleHash, cellStart, neibsList, particleRangeEnd, slength, influenceradius);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;

	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	#endif
	CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
}
};

template<FilterType filtertype, KernelType kerneltype, BoundaryType boundarytype>
class CUDAFilterEngine : public AbstractFilterEngine
{
public:
	CUDAFilterEngine(uint _frequency) : AbstractFilterEngine(_frequency)
	{}

	void setconstants() {} // TODO
	void getconstants() {} // TODO

	void
	process(
		const	float4	*pos,
		const	float4	*oldVel,
				float4	*newVel,
		const	particleinfo	*info,
		const	hashKey	*particleHash,
		const	uint	*cellStart,
		const	neibdata*neibsList,
				uint	numParticles,
				uint	particleRangeEnd,
				float	slength,
				float	influenceradius)
	{
		CUDAFilterEngineHelper<filtertype, kerneltype, boundarytype>::process
			(pos, oldVel, newVel, info, particleHash, cellStart, neibsList,
			 numParticles, particleRangeEnd, slength, influenceradius);
	}
};


/// Boundary conditions engines

// TODO FIXME at this time this is just a horrible hack to group the boundary-conditions
// methods needed for SA, it needs a heavy-duty refactoring of course

template<KernelType kerneltype, ViscosityType visctype,
	BoundaryType boundarytype, flag_t simflags>
class CUDABoundaryConditionsEngine : public AbstractBoundaryConditionsEngine
{
public:

void
updateNewIDsOffset(const uint &newIDsOffset)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cubounds::d_newIDsOffset, &newIDsOffset, sizeof(uint)));
}

/// Disables particles that went through boundaries when open boundaries are used
void
disableOutgoingParts(		float4*			pos,
							vertexinfo*		vertices,
					const	particleinfo*	info,
					const	uint			numParticles,
					const	uint			particleRangeEnd)
{
	uint numThreads = BLOCK_SIZE_FORCES;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	//execute kernel
	cuforces::disableOutgoingPartsDevice<<<numBlocks, numThreads>>>
		(	pos,
			vertices,
			numParticles);

	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}

/// Computes the boundary conditions on segments using the information from the fluid (on solid walls used for Neumann boundary conditions).
void
saSegmentBoundaryConditions(
			float4*			oldPos,
			float4*			oldVel,
			float*			oldTKE,
			float*			oldEps,
			float4*			oldEulerVel,
			float4*			oldGGam,
			vertexinfo*		vertices,
	const	uint*			vertIDToIndex,
	const	float2	* const vertPos[],
	const	float4*			boundelement,
	const	particleinfo*	info,
	const	hashKey*		particleHash,
	const	uint*			cellStart,
	const	neibdata*		neibsList,
	const	uint			numParticles,
	const	uint			particleRangeEnd,
	const	float			deltap,
	const	float			slength,
	const	float			influenceradius,
	const	bool			initStep,
	const	uint			step)
{
	uint numThreads = BLOCK_SIZE_FORCES;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	int dummy_shared = 0;
	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	CUDA_SAFE_CALL(cudaBindTexture(0, boundTex, boundelement, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	// execute the kernel
	cuforces::saSegmentBoundaryConditions<kerneltype><<< numBlocks, numThreads, dummy_shared >>>
		(oldPos, oldVel, oldTKE, oldEps, oldEulerVel, oldGGam, vertices, vertIDToIndex, vertPos[0], vertPos[1], vertPos[2], particleHash, cellStart, neibsList, particleRangeEnd, deltap, slength, influenceradius, initStep, step, simflags & ENABLE_INLET_OUTLET);

	CUDA_SAFE_CALL(cudaUnbindTexture(boundTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}

/// Apply boundary conditions to vertex particles.
// There is no need to use two velocity arrays (read and write) and swap them after.
// Computes the boundary conditions on vertex particles using the values from the segments associated to it. Also creates particles for inflow boundary conditions.
// Data is only read from fluid and segments and written only on vertices.
void
saVertexBoundaryConditions(
			float4*			oldPos,
			float4*			oldVel,
			float*			oldTKE,
			float*			oldEps,
			float4*			oldGGam,
			float4*			oldEulerVel,
			float4*			forces,
			float2*			contupd,
	const	float4*			boundelement,
			vertexinfo*		vertices,
	const	float2			* const vertPos[],
	const	uint*			vertIDToIndex,
			particleinfo*	info,
			hashKey*		particleHash,
	const	uint*			cellStart,
	const	neibdata*		neibsList,
	const	uint			numParticles,
			uint*			newNumParticles,
	const	uint			particleRangeEnd,
	const	float			dt,
	const	int				step,
	const	float			deltap,
	const	float			slength,
	const	float			influenceradius,
	const	bool			initStep,
	const	bool			resume,
	const	uint			deviceId,
	const	uint			numDevices)
{
	int dummy_shared = 0;

	uint numThreads = BLOCK_SIZE_SHEPARD;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	CUDA_SAFE_CALL(cudaBindTexture(0, boundTex, boundelement, numParticles*sizeof(float4)));

	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	// execute the kernel
	cuforces::saVertexBoundaryConditions<kerneltype><<< numBlocks, numThreads, dummy_shared >>>
		(oldPos, oldVel, oldTKE, oldEps, oldGGam, oldEulerVel, forces, contupd, vertices, vertPos[0], vertPos[1], vertPos[2], vertIDToIndex, info, particleHash, cellStart, neibsList,
		 particleRangeEnd, newNumParticles, dt, step, deltap, slength, influenceradius, initStep, resume, deviceId, numDevices);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;

	CUDA_SAFE_CALL(cudaUnbindTexture(boundTex));

}

/// Computes the initial value of gamma using a Gauss quadrature formula
void
initGamma(
	MultiBufferList::iterator bufwrite,
	MultiBufferList::const_iterator bufread,
	const	uint			numParticles,
	const	float			slength,
	const	float			deltap,
	const	float			influenceradius,
	const	float			epsilon,
	const	uint*			cellStart,
	const	uint			particleRangeEnd)
{
	uint numThreads = BLOCK_SIZE_FORCES;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	int dummy_shared = 0;
	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	const float4 *pos = bufread->getData<BUFFER_POS>();
	const particleinfo *info = bufread->getData<BUFFER_INFO>();
	const hashKey *pHash = bufread->getData<BUFFER_HASH>();
	const neibdata *neibsList = bufread->getData<BUFFER_NEIBSLIST>();
	const float2 * const *vertPos = bufread->getRawPtr<BUFFER_VERTPOS>();
	const float4 *oldGGam = bufread->getData<BUFFER_GRADGAMMA>();
	const vertexinfo *vertices = bufread->getData<BUFFER_VERTICES>();
	const uint *vertIDToIndex = bufread->getData<BUFFER_VERTIDINDEX>();
	float4 *newGGam = bufwrite->getData<BUFFER_GRADGAMMA>();
	float4 *boundelement = bufwrite->getData<BUFFER_BOUNDELEMENTS>();

	// execute the kernel
	cuforces::initGamma<kerneltype><<< numBlocks, numThreads, dummy_shared >>>
		(newGGam, boundelement, pos, oldGGam, vertices, vertIDToIndex, vertPos[0], vertPos[1], vertPos[2], pHash, info, cellStart, neibsList, particleRangeEnd, slength, deltap, influenceradius, epsilon);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}

// Downloads the per device waterdepth from the GPU
void
downloadIOwaterdepth(
			uint*	h_IOwaterdepth,
	const	uint*	d_IOwaterdepth,
	const	uint	numOpenBoundaries)
{
	CUDA_SAFE_CALL(cudaMemcpy(h_IOwaterdepth, d_IOwaterdepth, numOpenBoundaries*sizeof(int), cudaMemcpyDeviceToHost));
}

// Upload the global waterdepth to the GPU
void
uploadIOwaterdepth(
	const	uint*	h_IOwaterdepth,
			uint*	d_IOwaterdepth,
	const	uint	numOpenBoundaries)
{
	CUDA_SAFE_CALL(cudaMemcpy(d_IOwaterdepth, h_IOwaterdepth, numOpenBoundaries*sizeof(int), cudaMemcpyHostToDevice));
}

// Identifies vertices at the corners of open boundaries
void
saIdentifyCornerVertices(
	const	float4*			oldPos,
	const	float4*			boundelement,
			particleinfo*	info,
	const	hashKey*		particleHash,
	const	vertexinfo*		vertices,
	const	uint*			cellStart,
	const	neibdata*		neibsList,
	const	uint			numParticles,
	const	uint			particleRangeEnd,
	const	float			deltap,
	const	float			eps)
{
	int dummy_shared = 0;

	uint numThreads = BLOCK_SIZE_SHEPARD;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	CUDA_SAFE_CALL(cudaBindTexture(0, boundTex, boundelement, numParticles*sizeof(float4)));

	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif
	// execute the kernel
	cuforces::saIdentifyCornerVertices<<< numBlocks, numThreads, dummy_shared >>> (
		oldPos,
		info,
		particleHash,
		vertices,
		cellStart,
		neibsList,
		numParticles,
		deltap,
		eps);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;

	CUDA_SAFE_CALL(cudaUnbindTexture(boundTex));

}
};

