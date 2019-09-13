/*  Copyright (c) 2011-2019 INGV, EDF, UniCT, JHU

    Istituto Nazionale di Geofisica e Vulcanologia, Sezione di Catania, Italy
    Électricité de France, Paris, France
    Università di Catania, Catania, Italy
    Johns Hopkins University, Baltimore (MD), USA

    This file is part of GPUSPH. Project founders:
        Alexis Hérault, Giuseppe Bilotta, Robert A. Dalrymple,
        Eugenio Rustico, Ciro Del Negro
    For a full list of authors and project partners, consult the logs
    and the project website <https://www.gpusph.org>

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

/*! \file
 * Template implementation of ForceEngine in CUDA
 */

#include <cstdio>
#include <stdexcept>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/functional.h>

#include "textures.cuh"

#include "engine_forces.h"
#include "engine_filter.h"
#include "simflags.h"

#include "utils.h"
#include "cuda_call.h"

#include "define_buffers.h"

#include "forces_params.h"
#include "density_diffusion_params.h"

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
	#define BLOCK_SIZE_FMAX			256
	#define MAX_BLOCKS_FMAX			64
#endif

// We want to always have at least two warps per block in the reductions
#if BLOCK_SIZE_FMAX <= 32
#error "BLOCK_SIZE_FMAX must be larger than 32"
#endif


thread_local cudaArray*  dDem = NULL;

/* Auxiliary data for parallel reductions */
thread_local size_t	reduce_blocks = 0;
thread_local size_t	reduce_blocksize_max = 0;
thread_local size_t	reduce_bs2 = 0;
thread_local size_t	reduce_shmem_max = 0;
thread_local void*	reduce_buffer = NULL;

#include "forces_kernel.cu"

/// Run a single reduction step
/** \return the number of blocks used to reduce the given elements
 * If the function is invoked with NULL input, no reduction is used. This can be used
 * to compute the number of blocks without doing an actual reduction
 */
static inline int
reducefmax(
			float	*output,
	const	float	*input,
			int		nels)
{
	if (nels & 3) {
		std::cerr << nels << std::endl;
		throw std::runtime_error("number of elements to reduce is not a multiple of 4");
	}

	int numquarts = nels/4;

	uint numBlocks = div_up(numquarts, BLOCK_SIZE_FMAX);
	// we want to be able to complete the reduction in at most two steps. This means that:
	// * we must be able to feed the output back to this function as input; since we produce
	//   an output value per block, we need the number of blocks to be a multiple of 4
	//   (unless it's 1);
	// * the second time this function is invoked, we must produce a single result,
	//   so on the first run we must not produce more than BLOCK_SIZE_FMAX*4 elements;
	if (numBlocks > 1) {
		numBlocks = round_up(numBlocks, 4U);
		numBlocks = min(numBlocks, BLOCK_SIZE_FMAX*4);
	}

	// Only run the actual reduction if there's anything to reduce
	if (input) {
		const float4 *input4 = reinterpret_cast<const float4*>(input);

		cuforces::fmaxDevice<BLOCK_SIZE_FMAX><<<numBlocks, BLOCK_SIZE_FMAX>>>(output, input4, numquarts);
	}

	return numBlocks;
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

static inline float
cflmax( const uint	n,
const	float*		cfl,
		float*		tempCfl)
{
	const int numBlocks = reducefmax(tempCfl, cfl, n);
	float max = NAN;

	// check if kernel execution generated an error
	KERNEL_CHECK_ERROR;

	// Second reduction step, if necessary
	if (numBlocks > 1) {
		const int numBlocks2 = reducefmax(tempCfl, tempCfl, numBlocks);
		KERNEL_CHECK_ERROR;
		// The second run should have produced a single result, if it didn't
		// we busted something in the logic
		if (numBlocks2 > 1)
			throw std::runtime_error("reduction numBlocks error!");
	}

	CUDA_SAFE_CALL(cudaMemcpy(&max, tempCfl, sizeof(float), cudaMemcpyDeviceToHost));

	return max;
}

// CUDAForcesEngine. Some methods need helper classes, which are defined below,
// right before the actual class definition starts
template<
	KernelType kerneltype,
	SPHFormulation sph_formulation,
	DensityDiffusionType densitydiffusiontype,
	typename ViscSpec,
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
	process(BufferList const& bufread,
		BufferList& bufwrite,
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
	process(BufferList const& bufread,
		BufferList& bufwrite,
		const uint numParticles,
		float slength,
		float influenceradius)
	{
		uint numThreads = BLOCK_SIZE_FORCES;
		uint numBlocks = div_up(numParticles, numThreads);

		const float4 *pos = bufread.getData<BUFFER_POS>();
		const float4 *vol = bufread.getData<BUFFER_VOLUME>();
		const particleinfo *info = bufread.getData<BUFFER_INFO>();
		const hashKey *pHash = bufread.getData<BUFFER_HASH>();
		const uint *cellStart = bufread.getData<BUFFER_CELLSTART>();
		const neibdata *neibsList = bufread.getData<BUFFER_NEIBSLIST>();

		/* Update WRITE vel in place, caller should do a swap before and after */
		float4 *vel = bufwrite.getData<BUFFER_VEL>();
		float *sigma = bufwrite.getData<BUFFER_SIGMA>();

#if !PREFER_L1
		CUDA_SAFE_CALL(cudaBindTexture(0, posTex, bufread.getData<BUFFER_POS>(), numParticles*sizeof(float4)));
#endif

		cuforces::densityGrenierDevice<kerneltype, boundarytype>
			<<<numBlocks, numThreads>>>(sigma, pos, vel, info, pHash, vol, cellStart, neibsList, numParticles, slength, influenceradius);

#if !PREFER_L1
		CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
#endif

		// check if kernel invocation generated an error
		KERNEL_CHECK_ERROR;
	}
};

/// CUDAForcesEngine

template<
	KernelType kerneltype,
	SPHFormulation sph_formulation,
	DensityDiffusionType densitydiffusiontype,
	typename ViscSpec,
	BoundaryType boundarytype,
	flag_t simflags>
class CUDAForcesEngine : public AbstractForcesEngine
{
	static const RheologyType rheologytype = ViscSpec::rheologytype;
	static const TurbulenceModel turbmodel = ViscSpec::turbmodel;

	static const bool needs_eulerVel = (boundarytype == SA_BOUNDARY &&
			(turbmodel == KEPSILON || (simflags & ENABLE_INLET_OUTLET)));


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
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cusph::d_wcoeff_cubicspline, &kernelcoeff, sizeof(float)));
	kernelcoeff = 15.0f/(16.0f*M_PI*h3);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cusph::d_wcoeff_quadratic, &kernelcoeff, sizeof(float)));
	kernelcoeff = 21.0f/(16.0f*M_PI*h3);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cusph::d_wcoeff_wendland, &kernelcoeff, sizeof(float)));
	kernelcoeff = 3.0f/(4.0f*M_PI*h4);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cusph::d_fcoeff_cubicspline, &kernelcoeff, sizeof(float)));
	kernelcoeff = 15.0f/(32.0f*M_PI*h4);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cusph::d_fcoeff_quadratic, &kernelcoeff, sizeof(float)));
	kernelcoeff = 105.0f/(128.0f*M_PI*h5);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cusph::d_fcoeff_wendland, &kernelcoeff, sizeof(float)));

	/*	Gaussian kernel: W(r, h) (exp(-(r/h)^2) - exp(-(δ/h)^2))/const
		with δ cut-off radius (typically, 3h). For us, δ is the influence radius R*h,
		with R kernel radius
	 */
	const float R = simparams->kernelradius;
	const float R2 = R*R; // R2 = squared kernel radius = (δ/h)^2
	const float exp_R2 = exp(-R2); // exp(-(δ/h)^2)
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cusph::d_wsub_gaussian, &exp_R2, sizeof(float)));
	// constant: pi^(3/2)
#define M_PI_TO_3_2 5.5683279968317078452848179821188357020136243902832439
	// -2/3 exp(-R^2) h^3 Pi R (3 + 2 R^2) + h^3 Pi^(3/2) Erf(R) <- CHECK this
	kernelcoeff = -2*exp_R2/3 * h3 * M_PI * R*(3+2*R2) + h3 * M_PI_TO_3_2 * erf(R);
#undef M_PI_TO_3_2
	kernelcoeff = 1/kernelcoeff;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cusph::d_wcoeff_gaussian, &kernelcoeff, sizeof(float)));
	// the coefficient for the F is just the W coefficient times 2/h^2
	kernelcoeff *= 2/h2;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cusph::d_fcoeff_gaussian, &kernelcoeff, sizeof(float)));

	const uint numFluids = physparams->numFluids();
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_numfluids, &numFluids, sizeof(uint)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_rho0, &physparams->rho0[0], numFluids*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_bcoeff, &physparams->bcoeff[0], numFluids*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_gammacoeff, &physparams->gammacoeff[0], numFluids*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_sscoeff, &physparams->sscoeff[0], numFluids*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_sspowercoeff, &physparams->sspowercoeff[0], numFluids*sizeof(float)));

	// compute (and upload) square of sound speeds, needed for Ferrari
	float sqC0[numFluids];
	for (uint i = 0; i < numFluids; ++i) {
		sqC0[i]  = physparams->sscoeff[i];
		sqC0[i] *= sqC0[i];
	}
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_sqC0, sqC0, numFluids*sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_visccoeff, &physparams->visccoeff[0], numFluids*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_visc2coeff, &physparams->visc2coeff[0], numFluids*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_yield_strength, &physparams->yield_strength[0], numFluids*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_visc_nonlinear_param, &physparams->visc_nonlinear_param[0], numFluids*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_visc_regularization_param, &physparams->visc_regularization_param[0], numFluids*sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_limiting_kinvisc, &physparams->limiting_kinvisc, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_monaghan_visc_coeff, &physparams->monaghan_visc_coeff, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_artvisccoeff, &physparams->artvisccoeff, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_gravity, &physparams->gravity, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_dcoeff, &physparams->dcoeff, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_p1coeff, &physparams->p1coeff, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_p2coeff, &physparams->p2coeff, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_MK_K, &physparams->MK_K, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_MK_d, &physparams->MK_d, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_MK_beta, &physparams->MK_beta, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_r0, &physparams->r0, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_epsartvisc, &physparams->epsartvisc, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_sinpsi, &physparams->sinpsi[0], numFluids*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_cohesion, &physparams->cohesion[0], numFluids*sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cugeom::d_ewres, &physparams->ewres, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cugeom::d_nsres, &physparams->nsres, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cugeom::d_demdx, &physparams->demdx, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cugeom::d_demdy, &physparams->demdy, sizeof(float)));
	float demdxdy = physparams->demdx*physparams->demdy;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cugeom::d_demdxdy, &demdxdy, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cugeom::d_demzmin, &physparams->demzmin, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_smagfactor, &physparams->smagfactor, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_kspsfactor, &physparams->kspsfactor, sizeof(float)));

	float partsurf = physparams->partsurf;
	if (partsurf == 0.0f)
		partsurf = physparams->r0*physparams->r0;
		// partsurf = (6.0 - M_PI)*physparams->r0*physparams->r0/4;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_partsurf, &partsurf, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_cosconeanglefluid, &physparams->cosconeanglefluid, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuphys::d_cosconeanglenonfluid, &physparams->cosconeanglenonfluid, sizeof(float)));

	idx_t neiblist_end = simparams->neiblistsize*allocatedParticles;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_neiblist_end, &neiblist_end, sizeof(idx_t)));

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
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_cell_to_offset, cell_to_offset, 27*sizeof(char3)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_worldOrigin, &worldOrigin, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_gridSize, &gridSize, sizeof(uint3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_cellSize, &cellSize, sizeof(float3)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_densityDiffCoeff, &simparams->densityDiffCoeff, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_epsinterface, &physparams->epsinterface, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_repack_alpha, &simparams->repack_alpha, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_repack_a, &simparams->repack_a, sizeof(float)));

}


void
getconstants(PhysParams *physparams)
{
	uint numFluids = UINT_MAX;
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&numFluids, cuforces::d_numfluids, sizeof(uint)));
	if (numFluids != physparams->numFluids())
		throw std::runtime_error("wrong number of fluids");

	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->visccoeff[0], cuforces::d_visccoeff, numFluids*sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->yield_strength[0], cuforces::d_yield_strength, numFluids*sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->visc_nonlinear_param[0], cuforces::d_visc_nonlinear_param, numFluids*sizeof(float), 0));

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
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cugeom::d_numplanes, &numPlanes, sizeof(uint)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cugeom::d_plane, &planes[0], numPlanes*sizeof(plane_t)));
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
setfeastart(const int2* feanodefirstindex, const int2 *feapartsfirstindex, int numfeabodies)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_feanodesstartindex, feanodefirstindex, numfeabodies*sizeof(int2)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_feapartsstartindex, feapartsfirstindex, numfeabodies*sizeof(int2)));
}

void
setfeanatcoords(const float4* natcoords, const uint4* nodes, int numfeaparticles)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_feapartsnatcoords, natcoords, numfeaparticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_feapartsownnodes, nodes, numfeaparticles*sizeof(uint4)));
}


void
bind_textures(
	BufferList const& bufread,
	uint	numParticles,
	RunMode	run_mode)
{
	// bind textures to read all particles, not only internal ones
	#if !PREFER_L1
	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, bufread.getData<BUFFER_POS>(), numParticles*sizeof(float4)));
	#endif
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, bufread.getData<BUFFER_VEL>(), numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, bufread.getData<BUFFER_INFO>(), numParticles*sizeof(particleinfo)));

	const float4 *eulerVel = bufread.getData<BUFFER_EULERVEL>();
	if (run_mode != REPACK && needs_eulerVel) {
		if (!eulerVel)
			throw std::invalid_argument("eulerVel not set but needed");
		CUDA_SAFE_CALL(cudaBindTexture(0, eulerVelTex, eulerVel, numParticles*sizeof(float4)));
	} else {
		if (eulerVel)
			std::cerr << "eulerVel set but not used" << std::endl;
	}

	if (boundarytype == SA_BOUNDARY) {
		CUDA_SAFE_CALL(cudaBindTexture(0, gamTex, bufread.getData<BUFFER_GRADGAMMA>(), numParticles*sizeof(float4)));
		CUDA_SAFE_CALL(cudaBindTexture(0, boundTex, bufread.getData<BUFFER_BOUNDELEMENTS>(), numParticles*sizeof(float4)));
	}

	if (run_mode != REPACK && turbmodel == KEPSILON) {
		CUDA_SAFE_CALL(cudaBindTexture(0, keps_kTex, bufread.getData<BUFFER_TKE>(), numParticles*sizeof(float)));
		CUDA_SAFE_CALL(cudaBindTexture(0, keps_eTex, bufread.getData<BUFFER_EPSILON>(), numParticles*sizeof(float)));
	}
}

void
unbind_textures(RunMode run_mode)
{
	// TODO FIXME why are SPS textures unbound here but bound in sps?
	// shouldn't we bind them in bind_textures() instead?
	if (run_mode != REPACK && turbmodel == SPS) {
		CUDA_SAFE_CALL(cudaUnbindTexture(tau0Tex));
		CUDA_SAFE_CALL(cudaUnbindTexture(tau1Tex));
		CUDA_SAFE_CALL(cudaUnbindTexture(tau2Tex));
	}

	if (run_mode != REPACK && turbmodel == KEPSILON) {
		CUDA_SAFE_CALL(cudaUnbindTexture(keps_kTex));
		CUDA_SAFE_CALL(cudaUnbindTexture(keps_eTex));
	}

	if (boundarytype == SA_BOUNDARY) {
		CUDA_SAFE_CALL(cudaUnbindTexture(gamTex));
		CUDA_SAFE_CALL(cudaUnbindTexture(boundTex));
	}

	if (run_mode != REPACK && needs_eulerVel)
		CUDA_SAFE_CALL(cudaUnbindTexture(eulerVelTex));

	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
	#if !PREFER_L1
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
	return round_up(div_up<uint>(n, BLOCK_SIZE_FORCES), 4U);
}


// returns the number of elements in the intermediate reduction step, assuming n
// elements in the original array
uint
getFmaxTempElements(const uint n)
{
	return reducefmax(NULL, NULL, n);
}



float
dtreduce(	float	slength,
			float	dtadaptfactor,
			float	sspeed_cfl,
			float	max_kinematic,
			BufferList const& bufread,
			BufferList& bufwrite,
			uint	numBlocks,
			uint	numParticles)
{
	const float *cfl_forces = bufread.getData<BUFFER_CFL>();
	const float *cfl_gamma = bufread.getData<BUFFER_CFL_GAMMA>();
	const float *cfl_keps = bufread.getData<BUFFER_CFL_KEPS>();
	float *tempCfl = bufwrite.getData<BUFFER_CFL_TEMP>();

	// cfl holds one value per block in the forces kernel call,
	// so it holds numBlocks elements
	float maxcfl = cflmax(numBlocks, cfl_forces, tempCfl);
	float dt = dtadaptfactor*fminf(sqrtf(slength/maxcfl), slength/sspeed_cfl);

	if (boundarytype == SA_BOUNDARY && USING_DYNAMIC_GAMMA(simflags)) {
		// TODO FIXME cfl_gamma is handled differently from the other arrays,
		// because of the need to carry information across split forces kernel invokations.
		// The "pre-reduced” numBlocks elements are thus found after numParticles elements.
		size_t cfl_gamma_offset = round_up(numParticles, 4U);
		maxcfl = fmaxf(cflmax(numBlocks, cfl_gamma + cfl_gamma_offset, tempCfl), 1e-5f/dt);
		const float dt_gam = 0.001f/maxcfl;
		if (dt_gam < dt)
			dt = dt_gam;
	}

	if (rheologytype != INVISCID || turbmodel > ARTIFICIAL) {
		/* Stability condition from viscosity h²/ν
		   We get the maximum kinematic viscosity from the caller, and in the KEPS case we
		   add the maximum KEPS
		 */
		float visccoeff = max_kinematic;
		if (turbmodel == KEPSILON)
			visccoeff += cflmax(numBlocks, cfl_keps, tempCfl);

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
compute_density(BufferList const& bufread,
	BufferList& bufwrite,
	uint numParticles,
	float slength,
	float influenceradius)
{
	CUDADensityHelper<kerneltype, sph_formulation, boundarytype>::process(bufread,
		bufwrite, numParticles, slength, influenceradius);
	return;
}

void
compute_density_diffusion(
	BufferList const& bufread,
	BufferList& bufwrite,
	const	uint	numParticles,
	const	uint	particleRangeEnd,
	const	float	deltap,
	const	float	slength,
	const	float	influenceRadius,
	const	float	dt)
{
	uint numThreads = BLOCK_SIZE_FORCES;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	if (boundarytype == SA_BOUNDARY)
		CUDA_SAFE_CALL(cudaBindTexture(0, boundTex, bufread.getData<BUFFER_BOUNDELEMENTS>(), numParticles*sizeof(float4)));

	auto params = density_diffusion_params<kerneltype, sph_formulation, densitydiffusiontype, boundarytype, PT_FLUID>(
			bufwrite.getData<BUFFER_FORCES>(),
			bufread.getData<BUFFER_POS>(),
			bufread.getData<BUFFER_VEL>(),
			bufread.getData<BUFFER_INFO>(),
			bufread.getData<BUFFER_HASH>(),
			bufread.getData<BUFFER_CELLSTART>(),
			bufread.getData<BUFFER_NEIBSLIST>(),
			bufread.getData<BUFFER_GRADGAMMA>(),
			bufread.getRawPtr<BUFFER_VERTPOS>(),
			particleRangeEnd,
			deltap, slength, influenceRadius, dt);

	cuforces::computeDensityDiffusionDevice
		<kerneltype, sph_formulation, densitydiffusiontype, boundarytype,
		 ViscSpec, simflags, PT_FLUID>
		<<<numBlocks, numThreads>>>(params);

	// check if last kernel invocation generated an error
	KERNEL_CHECK_ERROR;

	if (boundarytype == SA_BOUNDARY)
		CUDA_SAFE_CALL(cudaUnbindTexture(boundTex));

}

/* forcesDevice kernel calls that involve vertex particles
 * are factored out here in this separate member function, that
 * does nothing in the non-SA_BOUNDARY case
 */
template<
	typename FluidVertexParams,
	typename VertexFluidParams>
enable_if_t<FluidVertexParams::boundarytype == SA_BOUNDARY>
vertex_forces(
	uint numBlocks, uint numThreads, int dummy_shared,
	FluidVertexParams const& params_fv,
	VertexFluidParams const& params_vf)
{
	cuforces::forcesDevice<<< numBlocks, numThreads, dummy_shared >>>(params_fv);

	// Fluid contributions to vertices is only needed to compute water depth
	// and for turbulent viscosity with the k-epsilon model
	const bool waterdepth =
		QUERY_ALL_FLAGS(simflags, ENABLE_INLET_OUTLET | ENABLE_WATER_DEPTH);
	const bool keps = (turbmodel == KEPSILON);
	if (waterdepth || keps) {
		cuforces::forcesDevice<<< numBlocks, numThreads, dummy_shared >>>(params_vf);
	}
}
template<
	typename FluidVertexParams,
	typename VertexFluidParams>
enable_if_t<FluidVertexParams::boundarytype != SA_BOUNDARY>
vertex_forces(
	uint numBlocks, uint numThreads, int dummy_shared,
	FluidVertexParams const& params_fv,
	VertexFluidParams const& params_vf)
{ /* do nothing */ }

/* forcesDevice kernel calls where the central type is boundary
 * are factored out here in this separate member function, that
 * does nothing in the SA_BOUNDARY case
 */
template<typename BoundaryFluidParams>
enable_if_t<BoundaryFluidParams::boundarytype == SA_BOUNDARY>
boundary_forces(
	uint numBlocks, uint numThreads, int dummy_shared,
	BoundaryFluidParams const& params_bf)
{ /* do nothing */ }
template<typename BoundaryFluidParams>
enable_if_t<BoundaryFluidParams::boundarytype != SA_BOUNDARY>
boundary_forces(
	uint numBlocks, uint numThreads, int dummy_shared,
	BoundaryFluidParams const& params_bf)
{
	cuforces::forcesDevice<<< numBlocks, numThreads, dummy_shared >>>(params_bf);
}

// Returns numBlock for delayed dt reduction in case of striping
uint
run_forces(
	BufferList const& bufread,
	BufferList& bufwrite,
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
	const	uint	step,
	const	float	dt,
	const	bool compute_object_forces)
{
	int dummy_shared = 0;

	const uint numParticlesInRange = toParticle - fromParticle;

	// thread per particle
	uint numThreads = BLOCK_SIZE_FORCES;
	// number of blocks, rounded up to next multiple of 4 to improve reductions
	uint numBlocks = round_up(div_up(numParticlesInRange, numThreads), 4U);
	#if (__COMPUTE__ == 20)
	int dtadapt = !!(simflags & ENABLE_DTADAPT);
	if (turbmodel == SPS)
		dummy_shared = 3328 - dtadapt*BLOCK_SIZE_FORCES*4;
	else
		dummy_shared = 2560 - dtadapt*BLOCK_SIZE_FORCES*4;
	#endif

	forces_params<kerneltype, sph_formulation, densitydiffusiontype, boundarytype, ViscSpec, simflags, PT_FLUID, PT_FLUID> params_ff(
		bufread, bufwrite,
		fromParticle, toParticle,
		deltap, slength, influenceradius, step, dt,
		epsilon,
		IOwaterdepth);

	cuforces::forcesDevice<<< numBlocks, numThreads, dummy_shared >>>(params_ff);
	{
		forces_params<kerneltype, sph_formulation, densitydiffusiontype, boundarytype, ViscSpec, simflags, PT_FLUID, PT_VERTEX> params_fv(
			bufread, bufwrite,
			fromParticle, toParticle,
			deltap, slength, influenceradius, step, dt,
			epsilon,
			IOwaterdepth);

		forces_params<kerneltype, sph_formulation, densitydiffusiontype, boundarytype, ViscSpec, simflags, PT_VERTEX, PT_FLUID> params_vf(
			bufread, bufwrite,
			fromParticle, toParticle,
			deltap, slength, influenceradius, step, dt,
			epsilon,
			IOwaterdepth);

		vertex_forces(numBlocks, numThreads, dummy_shared, params_fv, params_vf);
	}

	forces_params<kerneltype, sph_formulation, densitydiffusiontype, boundarytype, ViscSpec, simflags, PT_FLUID, PT_BOUNDARY> params_fb(
		bufread, bufwrite,
		fromParticle, toParticle,
		deltap, slength, influenceradius, step, dt,
		epsilon,
		IOwaterdepth);

	cuforces::forcesDevice<<< numBlocks, numThreads, dummy_shared >>>(params_fb);

	if (compute_object_forces || (boundarytype == DYN_BOUNDARY)) {
		forces_params<kerneltype, sph_formulation, densitydiffusiontype, boundarytype, ViscSpec, simflags, PT_BOUNDARY, PT_FLUID> params_bf(
			bufread, bufwrite,
			fromParticle, toParticle,
			deltap, slength, influenceradius, step, dt,
			epsilon,
			IOwaterdepth);

		boundary_forces(numBlocks, numThreads, dummy_shared, params_bf);
	}

	finalize_forces_params<sph_formulation, boundarytype, ViscSpec, simflags> params_finalize(
			bufread, bufwrite,
			numParticles, fromParticle, toParticle, slength, deltap,
			cflOffset,
			IOwaterdepth);

	cuforces::finalizeforcesDevice<<< numBlocks, numThreads, dummy_shared >>>(params_finalize);

	return numBlocks;
}

/* repackDevice kernel calls that involve vertex particles
 * are factored out here in this separate member function, that
 * does nothing in the non-SA_BOUNDARY case
 */
template<typename FluidVertexParams>
enable_if_t<FluidVertexParams::boundarytype == SA_BOUNDARY>
vertex_repack(
	uint numBlocks, uint numThreads, int dummy_shared,
	FluidVertexParams const& params_fv)
{
	cuforces::repackDevice<<< numBlocks, numThreads, dummy_shared >>>(params_fv);
}

template<typename FluidVertexParams>
enable_if_t<FluidVertexParams::boundarytype != SA_BOUNDARY>
vertex_repack(
	uint numBlocks, uint numThreads, int dummy_shared,
	FluidVertexParams const& params_fv)
{ /* do nothing */ }

uint
run_repack(
		BufferList const& bufread,
		BufferList& bufwrite,
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
		const	uint	step,
		const	float	dt)
{
	int dummy_shared = 0;

	const uint numParticlesInRange = toParticle - fromParticle;

	// thread per particle
	uint numThreads = BLOCK_SIZE_FORCES;
	// number of blocks, rounded up to next multiple of 4 to improve reductions
	uint numBlocks = round_up(div_up(numParticlesInRange, numThreads), 4U);
#if (__COMPUTE__ == 20)
	int dtadapt = !!(simflags & ENABLE_DTADAPT);
	dummy_shared = 2560 - dtadapt*BLOCK_SIZE_FORCES*4;
#endif

	repack_params<kerneltype, boundarytype, simflags, PT_FLUID, PT_FLUID> params_ff(
		bufread, bufwrite,
		fromParticle, toParticle,
		deltap, slength, influenceradius, step, dt,
		epsilon,
		IOwaterdepth);

	cuforces::repackDevice<<< numBlocks, numThreads, dummy_shared >>>(params_ff);

	{
		repack_params<kerneltype, boundarytype, simflags, PT_FLUID, PT_VERTEX> params_fv(
			bufread, bufwrite,
			fromParticle, toParticle,
			deltap, slength, influenceradius, step, dt,
			epsilon,
			IOwaterdepth);

		vertex_repack(numBlocks, numThreads, dummy_shared, params_fv);
	}

	repack_params<kerneltype, boundarytype, simflags, PT_FLUID, PT_BOUNDARY> params_fb(
		bufread, bufwrite,
		fromParticle, toParticle,
		deltap, slength, influenceradius, step, dt,
		epsilon,
		IOwaterdepth);

	cuforces::repackDevice<<< numBlocks, numThreads, dummy_shared >>>(params_fb);

	finalize_repack_params<boundarytype, simflags> params_finalize(
		bufread, bufwrite,
		numParticles, fromParticle, toParticle, slength, deltap,
		cflOffset,
		IOwaterdepth);

	cuforces::finalizeRepackDevice<<< numBlocks, numThreads, dummy_shared >>>(params_finalize);

	return numBlocks;
}


// Returns numBlock for delayed dt reduction in case of striping
uint
basicstep(
	BufferList const& bufread,
	BufferList& bufwrite,
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
	const	RunMode	run_mode,
	const	int	step,
	const	float	dt,
	const	bool compute_object_forces)
{
	if (run_mode == REPACK)
		return run_repack(bufread, bufwrite,
			numParticles, fromParticle, toParticle,
			deltap, slength, dtadaptfactor,
			influenceradius, epsilon,
			IOwaterdepth,
			cflOffset,
			step, dt);
	else
		return run_forces(bufread, bufwrite,
			numParticles, fromParticle, toParticle,
			deltap, slength, dtadaptfactor,
			influenceradius, epsilon,
			IOwaterdepth,
			cflOffset,
			step, dt, compute_object_forces);
}

void
setDEM(const float *hDem, int width, int height)
{
	// Allocating, reading and copying DEM
	unsigned int size = width*height*sizeof(float);
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	CUDA_SAFE_CALL( cudaMallocArray( &dDem, &channelDesc, width, height ));
	CUDA_SAFE_CALL( cudaMemcpyToArray( dDem, 0, 0, hDem, size, cudaMemcpyHostToDevice));

	cugeom::demTex.addressMode[0] = cudaAddressModeClamp;
	cugeom::demTex.addressMode[1] = cudaAddressModeClamp;
	cugeom::demTex.filterMode = cudaFilterModeLinear;
	cugeom::demTex.normalized = false;

	CUDA_SAFE_CALL( cudaBindTextureToArray(cugeom::demTex, dDem, channelDesc));
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
reduceRbForces(	BufferList& bufwrite,
				uint	*lastindex,
				float3	*totalforce,
				float3	*totaltorque,
				uint	numforcesbodies,
				uint	numForcesBodiesParticles)
{
	float4 *forces = bufwrite.getData<BUFFER_RB_FORCES>();
	float4 *torques = bufwrite.getData<BUFFER_RB_TORQUES>();
	const uint *rbnum = bufwrite.getConstData<BUFFER_RB_KEYS>();

	thrust::device_ptr<float4> forces_devptr = thrust::device_pointer_cast(forces);
	thrust::device_ptr<float4> torques_devptr = thrust::device_pointer_cast(torques);
	const thrust::device_ptr<const uint> rbnum_devptr = thrust::device_pointer_cast(rbnum);
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

/// Preprocessing engines (Shepard, MLS)

// As with the viscengine, we need a helper struct for the partial
// specialization of process
template<FilterType filtertype, KernelType kerneltype, BoundaryType boundarytype>
struct CUDAFilterEngineHelper
{
	static void process(
		const	BufferList& bufread,
				BufferList& bufwrite,
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
		const	BufferList& bufread,
				BufferList& bufwrite,
				uint	numParticles,
				uint	particleRangeEnd,
				float	slength,
				float	influenceradius)
{
	const float4 *pos = bufread.getData<BUFFER_POS>();
	const float4 *oldVel = bufread.getData<BUFFER_VEL>();
	float4 *newVel = bufwrite.getData<BUFFER_VEL>();
	const particleinfo *info = bufread.getData<BUFFER_INFO>();
	const hashKey *particleHash = bufread.getData<BUFFER_HASH>();
	const uint *cellStart = bufread.getData<BUFFER_CELLSTART>();
	const neibdata*neibsList = bufread.getData<BUFFER_NEIBSLIST>();

	int dummy_shared = 0;
	// thread per particle
	uint numThreads = BLOCK_SIZE_SHEPARD;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	#if !PREFER_L1
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

	#if !PREFER_L1
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
		const	BufferList& bufread,
				BufferList& bufwrite,
				uint	numParticles,
				uint	particleRangeEnd,
				float	slength,
				float	influenceradius)
{
	const float4 *pos = bufread.getData<BUFFER_POS>();
	const float4 *oldVel = bufread.getData<BUFFER_VEL>();
	float4 *newVel = bufwrite.getData<BUFFER_VEL>();
	const particleinfo *info = bufread.getData<BUFFER_INFO>();
	const hashKey *particleHash = bufread.getData<BUFFER_HASH>();
	const uint *cellStart = bufread.getData<BUFFER_CELLSTART>();
	const neibdata*neibsList = bufread.getData<BUFFER_NEIBSLIST>();

	int dummy_shared = 0;
	// thread per particle
	uint numThreads = BLOCK_SIZE_MLS;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	#if !PREFER_L1
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

	#if !PREFER_L1
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
		const	BufferList& bufread,
				BufferList& bufwrite,
				uint	numParticles,
				uint	particleRangeEnd,
				float	slength,
				float	influenceradius)
	{
		CUDAFilterEngineHelper<filtertype, kerneltype, boundarytype>::process
			(bufread, bufwrite,
			 numParticles, particleRangeEnd, slength, influenceradius);
	}
};


