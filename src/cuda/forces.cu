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

#include "engine_forces.h"
#include "engine_filter.h"
#include "simflags.h"

#include "utils.h"
#include "safe_call.h"

#include "define_buffers.h"

#include "forces_params.h"
#include "density_diffusion_params.h"

#include "posvel_struct.h"

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
#if CUDA_BACKEND_ENABLED
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
		numBlocks = std::min(numBlocks, BLOCK_SIZE_FMAX*4U);
	}

	// Only run the actual reduction if there's anything to reduce
	if (input) {
		const float4 *input4 = reinterpret_cast<const float4*>(input);

		execute_kernel(cuforces::fmaxDevice<BLOCK_SIZE_FMAX>(output, input4, numquarts),
			numBlocks, BLOCK_SIZE_FMAX);
	}

	return numBlocks;
}
#endif

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
#if CPU_BACKEND_ENABLED
	float max = cfl[0];
#else // CUDA_BACKEND_ENABLED
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

	SAFE_CALL(cudaMemcpy(&max, tempCfl, sizeof(float), cudaMemcpyDeviceToHost));
#endif

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

		execute_kernel(
			cuforces::densityGrenierDevice<kerneltype, boundarytype>(
				bufread, bufwrite, numParticles, slength, influenceradius),
			numBlocks, numThreads);

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

void
setconstants(const SimParams *simparams, const PhysParams *physparams,
	float3 const& worldOrigin, uint3 const& gridSize, float3 const& cellSize,
	idx_t const& allocatedParticles)
{
	const int dim = space_dimensions_for(simparams->dimensions);

	// TODO FIXME we don't have all combos defined yet
	std::string unsupported_dim_kernel_combo = "coefficients for " +
		std::string(KernelName[kerneltype]) + " not defined for " +
		std::string(DimensionalityName[simparams->dimensions]) + " problems (" +
		std::to_string(dim) + " dimensions)";

	// Setting kernels and kernels derivative factors
	const float h = simparams->slength;
	const float h2 = h*h;
	const float h3 = h2*h;
	const float h4 = h2*h2;
	const float h5 = h4*h;

	// TODO why do we bother setting them up for all kernels, rather than relying on the
	// kerneltype information from simparams?

	float kernelcoeff, gradcoeff;

	// CUBICSPLINE
	switch (dim) {
	case 1:
		kernelcoeff = 4.0f/(3.0f*h);
		gradcoeff = 1.0f/h2;
		break;
	case 2:
		kernelcoeff = 10.f/(7.0f*M_PI*h2);
		gradcoeff = 15.f/(14.0f*M_PI*h3);
		break;
	case 3:
		kernelcoeff = 1.0f/(M_PI*h3);
		gradcoeff = 3.0f/(4.0f*M_PI*h4);
		break;
	default:
		if (kerneltype == CUBICSPLINE)
			throw std::invalid_argument(unsupported_dim_kernel_combo);
	}
	COPY_TO_SYMBOL(cusph::d_wcoeff_cubicspline, kernelcoeff, 1);
	COPY_TO_SYMBOL(cusph::d_fcoeff_cubicspline, gradcoeff, 1);

	// QUADRATIC
	switch (dim) {
	case 3:
		kernelcoeff = 15.0f/(16.0f*M_PI*h3);
		gradcoeff = 15.0f/(32.0f*M_PI*h4);
		break;
	default:
		if (kerneltype == QUADRATIC)
			throw std::invalid_argument(unsupported_dim_kernel_combo);
	}
	COPY_TO_SYMBOL(cusph::d_wcoeff_quadratic, kernelcoeff, 1);
	COPY_TO_SYMBOL(cusph::d_fcoeff_quadratic, gradcoeff, 1);

	// WENDLAND
	switch (dim) {
	case 1:
		kernelcoeff = 3.0f/(2.0f*h);
		gradcoeff = 15.0f/(16.0f*M_PI*h3);
		break;
	case 2:
		kernelcoeff = 7.0f/(4.0f*M_PI*h2);
		gradcoeff = 35.0f/(32.0f*M_PI*h4);
		break;
	case 3:
		kernelcoeff = 21.0f/(16.0f*M_PI*h3);
		gradcoeff = 105.0f/(128.0f*M_PI*h5);
		break;
	default:
		if (kerneltype == WENDLAND)
			throw std::invalid_argument(unsupported_dim_kernel_combo);
	}
	COPY_TO_SYMBOL(cusph::d_wcoeff_wendland, kernelcoeff, 1);
	COPY_TO_SYMBOL(cusph::d_fcoeff_wendland, gradcoeff, 1);

	// GAUSSIAN
	if (kerneltype == GAUSSIAN && dim != 3)
		throw std::invalid_argument(unsupported_dim_kernel_combo);

	/*	Gaussian kernel: W(r, h) (exp(-(r/h)^2) - exp(-(δ/h)^2))/const
		with δ cut-off radius (typically, 3h). For us, δ is the influence radius R*h,
		with R kernel radius
	 */
	const float R = simparams->kernelradius;
	const float R2 = R*R; // R2 = squared kernel radius = (δ/h)^2
	const float exp_R2 = exp(-R2); // exp(-(δ/h)^2)
	COPY_TO_SYMBOL(cusph::d_wsub_gaussian, exp_R2, 1);
	// constant: pi^(3/2)
#define M_PI_TO_3_2 5.5683279968317078452848179821188357020136243902832439
	// -2/3 exp(-R^2) h^3 Pi R (3 + 2 R^2) + h^3 Pi^(3/2) Erf(R) <- CHECK this
	kernelcoeff = -2*exp_R2/3 * h3 * M_PI * R*(3+2*R2) + h3 * M_PI_TO_3_2 * erf(R);
#undef M_PI_TO_3_2
	kernelcoeff = 1/kernelcoeff;
	COPY_TO_SYMBOL(cusph::d_wcoeff_gaussian, kernelcoeff, 1);
	// the coefficient for the F is just the W coefficient times 2/h^2
	kernelcoeff *= 2/h2;
	COPY_TO_SYMBOL(cusph::d_fcoeff_gaussian, kernelcoeff, 1);

	const uint numFluids = physparams->numFluids();
	COPY_TO_SYMBOL(cuphys::d_numfluids, numFluids, 1);
	COPY_TO_SYMBOL(cuphys::d_rho0, physparams->rho0[0], numFluids);
	COPY_TO_SYMBOL(cuphys::d_bcoeff, physparams->bcoeff[0], numFluids);
	COPY_TO_SYMBOL(cuphys::d_gammacoeff, physparams->gammacoeff[0], numFluids);
	COPY_TO_SYMBOL(cuphys::d_sscoeff, physparams->sscoeff[0], numFluids);
	COPY_TO_SYMBOL(cuphys::d_sspowercoeff, physparams->sspowercoeff[0], numFluids);

	// compute (and upload) square of sound speeds, needed for Ferrari
	float sqC0[numFluids];
	for (uint i = 0; i < numFluids; ++i) {
		sqC0[i]  = physparams->sscoeff[i];
		sqC0[i] *= sqC0[i];
	}
	COPY_TO_SYMBOL(cuphys::d_sqC0, sqC0[0], numFluids);

	COPY_TO_SYMBOL(cuphys::d_visccoeff, physparams->visccoeff[0], numFluids);
	COPY_TO_SYMBOL(cuphys::d_visc2coeff, physparams->visc2coeff[0], numFluids);
	COPY_TO_SYMBOL(cuphys::d_yield_strength, physparams->yield_strength[0], numFluids);
	COPY_TO_SYMBOL(cuphys::d_visc_nonlinear_param, physparams->visc_nonlinear_param[0], numFluids);
	COPY_TO_SYMBOL(cuphys::d_visc_regularization_param, physparams->visc_regularization_param[0], numFluids);

	COPY_TO_SYMBOL(cuphys::d_limiting_kinvisc, physparams->limiting_kinvisc, 1);
	COPY_TO_SYMBOL(cuphys::d_monaghan_visc_coeff, physparams->monaghan_visc_coeff, 1);

	COPY_TO_SYMBOL(cuphys::d_artvisccoeff, physparams->artvisccoeff, 1);

	COPY_TO_SYMBOL(cuphys::d_gravity, physparams->gravity, 1);
	COPY_TO_SYMBOL(cuphys::d_dcoeff, physparams->dcoeff, 1);
	COPY_TO_SYMBOL(cuphys::d_p1coeff, physparams->p1coeff, 1);
	COPY_TO_SYMBOL(cuphys::d_p2coeff, physparams->p2coeff, 1);

	COPY_TO_SYMBOL(cuphys::d_MK_K, physparams->MK_K, 1);
	COPY_TO_SYMBOL(cuphys::d_MK_d, physparams->MK_d, 1);
	COPY_TO_SYMBOL(cuphys::d_MK_beta, physparams->MK_beta, 1);

	COPY_TO_SYMBOL(cuphys::d_r0, physparams->r0, 1);
	COPY_TO_SYMBOL(cuphys::d_epsartvisc, physparams->epsartvisc, 1);

	COPY_TO_SYMBOL(cuphys::d_sinpsi, physparams->sinpsi[0], numFluids);
	COPY_TO_SYMBOL(cuphys::d_cohesion, physparams->cohesion[0], numFluids);

	// DEM uploads
	{
		const float ewres = physparams->ewres;
		const float nsres = physparams->nsres;
		const float dx = physparams->demdx;
		const float dy = physparams->demdy;

		const float2 scaled_cellSize = make_float2(cellSize.x/ewres, cellSize.y/nsres);
		const float2 scaled_dx = make_float2(dx/ewres, dy/nsres);
		const float demdxdy = dx*dy;

		COPY_TO_SYMBOL(cugeom::d_dem_pos_fixup, physparams->dem_pos_fixup, 1);
		COPY_TO_SYMBOL(cugeom::d_dem_scaled_cellSize, scaled_cellSize, 1);
		COPY_TO_SYMBOL(cugeom::d_dem_scaled_dx, scaled_dx, 1);

		COPY_TO_SYMBOL(cugeom::d_ewres, ewres, 1);
		COPY_TO_SYMBOL(cugeom::d_nsres, nsres, 1);
		COPY_TO_SYMBOL(cugeom::d_demdx, dx, 1);
		COPY_TO_SYMBOL(cugeom::d_demdy, dy, 1);
		COPY_TO_SYMBOL(cugeom::d_demdxdy, demdxdy, 1);
		COPY_TO_SYMBOL(cugeom::d_demzmin, physparams->demzmin, 1);
	}

	COPY_TO_SYMBOL(cuphys::d_smagfactor, physparams->smagfactor, 1);
	COPY_TO_SYMBOL(cuphys::d_kspsfactor, physparams->kspsfactor, 1);

	float partsurf = physparams->partsurf;
	if (partsurf == 0.0f)
		partsurf = physparams->r0*physparams->r0;
		// partsurf = (6.0 - M_PI)*physparams->r0*physparams->r0/4;
	COPY_TO_SYMBOL(cuphys::d_partsurf, partsurf, 1);

	COPY_TO_SYMBOL(cuphys::d_cosconeanglefluid, physparams->cosconeanglefluid, 1);
	COPY_TO_SYMBOL(cuphys::d_cosconeanglenonfluid, physparams->cosconeanglenonfluid, 1);

	idx_t neiblist_end = simparams->neiblistsize*allocatedParticles;
	COPY_TO_SYMBOL(cuneibs::d_neiblist_end, neiblist_end, 1);

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
	COPY_TO_SYMBOL(cuneibs::d_cell_to_offset, cell_to_offset[0], 27);

	COPY_TO_SYMBOL(cuneibs::d_worldOrigin, worldOrigin, 1);
	COPY_TO_SYMBOL(cuneibs::d_gridSize, gridSize, 1);
	COPY_TO_SYMBOL(cuneibs::d_cellSize, cellSize, 1);

	COPY_TO_SYMBOL(cuforces::d_densityDiffCoeff, simparams->densityDiffCoeff, 1);

	COPY_TO_SYMBOL(cuforces::d_epsinterface, physparams->epsinterface, 1);

	COPY_TO_SYMBOL(cuforces::d_repack_alpha, simparams->repack_alpha, 1);
	COPY_TO_SYMBOL(cuforces::d_repack_a, simparams->repack_a, 1);

	COPY_TO_SYMBOL(cuforces::d_ccsph_min_det, simparams->ccsph_min_det, 1);
}


void
getconstants(PhysParams *physparams)
{
	uint numFluids = UINT_MAX;
	COPY_FROM_SYMBOL(numFluids, cuforces::d_numfluids, 1);
	if (numFluids != physparams->numFluids())
		throw std::runtime_error("wrong number of fluids");

	COPY_FROM_SYMBOL(physparams->visccoeff[0], cuforces::d_visccoeff, numFluids);
	COPY_FROM_SYMBOL(physparams->yield_strength[0], cuforces::d_yield_strength, numFluids);
	COPY_FROM_SYMBOL(physparams->visc_nonlinear_param[0], cuforces::d_visc_nonlinear_param, numFluids);

	COPY_FROM_SYMBOL(physparams->rho0[0], cuforces::d_rho0, numFluids);
	COPY_FROM_SYMBOL(physparams->bcoeff[0], cuforces::d_bcoeff, numFluids);
	COPY_FROM_SYMBOL(physparams->gammacoeff[0], cuforces::d_gammacoeff, numFluids);
	COPY_FROM_SYMBOL(physparams->sscoeff[0], cuforces::d_sscoeff, numFluids);
	COPY_FROM_SYMBOL(physparams->sspowercoeff[0], cuforces::d_sspowercoeff, numFluids);
	COPY_FROM_SYMBOL(physparams->gravity, cuforces::d_gravity, 1);

	COPY_FROM_SYMBOL(physparams->dcoeff, cuforces::d_dcoeff, 1);
	COPY_FROM_SYMBOL(physparams->p1coeff, cuforces::d_p1coeff, 1);
	COPY_FROM_SYMBOL(physparams->p2coeff, cuforces::d_p2coeff, 1);

	COPY_FROM_SYMBOL(physparams->MK_K, cuforces::d_MK_K, 1);
	COPY_FROM_SYMBOL(physparams->MK_d, cuforces::d_MK_d, 1);
	COPY_FROM_SYMBOL(physparams->MK_beta, cuforces::d_MK_beta, 1);

	COPY_FROM_SYMBOL(physparams->r0, cuforces::d_r0, 1);
	COPY_FROM_SYMBOL(physparams->epsartvisc, cuforces::d_epsartvisc, 1);
	COPY_FROM_SYMBOL(physparams->ewres, cuforces::d_ewres, 1);
	COPY_FROM_SYMBOL(physparams->nsres, cuforces::d_nsres, 1);
	COPY_FROM_SYMBOL(physparams->demdx, cuforces::d_demdx, 1);
	COPY_FROM_SYMBOL(physparams->demdy, cuforces::d_demdy, 1);
	COPY_FROM_SYMBOL(physparams->demzmin, cuforces::d_demzmin, 1);
	COPY_FROM_SYMBOL(physparams->smagfactor, cuforces::d_smagfactor, 1);
	COPY_FROM_SYMBOL(physparams->kspsfactor, cuforces::d_kspsfactor, 1);

	COPY_TO_SYMBOL(cuforces::d_epsinterface, physparams->epsinterface, 1);
}

void
setplanes(std::vector<plane_t> const& planes)
{
	uint numPlanes = planes.size();
	COPY_TO_SYMBOL(cugeom::d_numplanes, numPlanes, 1);
	COPY_TO_SYMBOL(cugeom::d_plane, planes[0], numPlanes);
}

void
setgravity(float3 const& gravity)
{
	COPY_TO_SYMBOL(cuforces::d_gravity, gravity, 1);
}

void
setrbcg(const int3* cgGridPos, const float3* cgPos, int numbodies)
{
	COPY_TO_SYMBOL(cuforces::d_rbcgGridPos, cgGridPos[0], numbodies);
	COPY_TO_SYMBOL(cuforces::d_rbcgPos, cgPos[0], numbodies);
}

void
setrbstart(const int* rbfirstindex, int numbodies)
{
	COPY_TO_SYMBOL(cuforces::d_rbstartindex, rbfirstindex[0], numbodies);
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
#if CPU_BACKEND_ENABLED
	return 1;
#else // CUDA_BACKEND_ENABLED
	return reducefmax(NULL, NULL, n);
#endif
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

	auto params = density_diffusion_params<kerneltype, sph_formulation, densitydiffusiontype, boundarytype, PT_FLUID>
		(bufread, bufwrite, particleRangeEnd, deltap, slength, influenceRadius, dt);

	execute_kernel(
		cuforces::computeDensityDiffusionDevice<kerneltype, sph_formulation, densitydiffusiontype, boundarytype,
			ViscSpec, simflags, PT_FLUID>
			(bufread, bufwrite, particleRangeEnd, deltap, slength, influenceRadius, dt),
		numBlocks, numThreads);

	// check if last kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}

// computing the CSPM coefficients for CCSPH and ANTUONO / DELTA_SPH
void
compute_cspm_coeff(
	BufferList const& bufread,
	BufferList& bufwrite,
	const	uint	numParticles,
	const	uint	particleRangeEnd,
	const	float	deltap, //FIXME check if actually needed
	const	float	slength,
	const	float	influenceRadius)
{
	uint numThreads = BLOCK_SIZE_FORCES;
	uint numBlocks = div_up(numParticles, numThreads);

	execute_kernel(
		cuforces::cspmCoeffDevice<kerneltype, boundarytype, densitydiffusiontype, simflags>
			(bufread, bufwrite, particleRangeEnd, slength, influenceRadius),
		numBlocks, numThreads);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
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
	execute_kernel(
		cuforces::forcesDevice<FluidVertexParams>(params_fv),
		numBlocks, numThreads, dummy_shared);

	// Fluid contributions to vertices is only needed to compute water depth
	// and for turbulent viscosity with the k-epsilon model
	static constexpr bool waterdepth =
		QUERY_ALL_FLAGS(simflags, ENABLE_INLET_OUTLET | ENABLE_WATER_DEPTH);
	static constexpr bool keps = (turbmodel == KEPSILON);
	if (waterdepth || keps) {
		execute_kernel(
			cuforces::forcesDevice<VertexFluidParams>(params_vf),
			numBlocks, numThreads, dummy_shared);
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
	execute_kernel(cuforces::forcesDevice<BoundaryFluidParams>(params_bf),
		numBlocks, numThreads, dummy_shared);
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
	static constexpr bool dtadapt = HAS_DTADAPT(simflags);
	if (turbmodel == SPS)
		dummy_shared = 3328 - dtadapt*BLOCK_SIZE_FORCES*4;
	else
		dummy_shared = 2560 - dtadapt*BLOCK_SIZE_FORCES*4;
	#endif

	using FluidFluidParams    = forces_params<kerneltype, sph_formulation, densitydiffusiontype, boundarytype, ViscSpec, simflags, PT_FLUID, PT_FLUID>;
	using FluidVertexParams   = forces_params<kerneltype, sph_formulation, densitydiffusiontype, boundarytype, ViscSpec, simflags, PT_FLUID, PT_VERTEX>;
	using VertexFluidParams   = forces_params<kerneltype, sph_formulation, densitydiffusiontype, boundarytype, ViscSpec, simflags, PT_VERTEX, PT_FLUID>;
	using FluidBoundaryParams = forces_params<kerneltype, sph_formulation, densitydiffusiontype, boundarytype, ViscSpec, simflags, PT_FLUID, PT_BOUNDARY>;
	using BoundaryFluidParams = forces_params<kerneltype, sph_formulation, densitydiffusiontype, boundarytype, ViscSpec, simflags, PT_BOUNDARY, PT_FLUID>;

	FluidFluidParams params_ff(
		bufread, bufwrite,
		fromParticle, toParticle,
		deltap, slength, influenceradius, step, dt,
		epsilon,
		IOwaterdepth);

	execute_kernel(cuforces::forcesDevice<FluidFluidParams>(params_ff), numBlocks, numThreads, dummy_shared);

	{
		FluidVertexParams params_fv(
			bufread, bufwrite,
			fromParticle, toParticle,
			deltap, slength, influenceradius, step, dt,
			epsilon,
			IOwaterdepth);

		VertexFluidParams params_vf(
			bufread, bufwrite,
			fromParticle, toParticle,
			deltap, slength, influenceradius, step, dt,
			epsilon,
			IOwaterdepth);

		vertex_forces(numBlocks, numThreads, dummy_shared, params_fv, params_vf);
	}

	FluidBoundaryParams params_fb(
		bufread, bufwrite,
		fromParticle, toParticle,
		deltap, slength, influenceradius, step, dt,
		epsilon,
		IOwaterdepth);

	execute_kernel(cuforces::forcesDevice<FluidBoundaryParams>(params_fb), numBlocks, numThreads, dummy_shared);

	if (compute_object_forces || (boundarytype == DYN_BOUNDARY)) {
		BoundaryFluidParams params_bf(
			bufread, bufwrite,
			fromParticle, toParticle,
			deltap, slength, influenceradius, step, dt,
			epsilon,
			IOwaterdepth);

		boundary_forces(numBlocks, numThreads, dummy_shared, params_bf);
	}

	using FinalizeForcesParams = finalize_forces_params<sph_formulation, boundarytype, ViscSpec, simflags>;

	FinalizeForcesParams params_finalize(
			bufread, bufwrite,
			numParticles, fromParticle, toParticle, slength, deltap,
			cflOffset,
			IOwaterdepth);

	execute_kernel(cuforces::finalizeforcesDevice<FinalizeForcesParams>(params_finalize),
		numBlocks, numThreads, dummy_shared);

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
	execute_kernel(cuforces::repackDevice<FluidVertexParams>(params_fv), numBlocks, numThreads, dummy_shared);
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
	static constexpr bool dtadapt = HAS_DTADAPT(simflags);
	dummy_shared = 2560 - dtadapt*BLOCK_SIZE_FORCES*4;
#endif

	using FluidFluidParams    = repack_params<kerneltype, boundarytype, simflags, PT_FLUID, PT_FLUID>;
	using FluidVertexParams   = repack_params<kerneltype, boundarytype, simflags, PT_FLUID, PT_VERTEX>;
	using FluidBoundaryParams = repack_params<kerneltype, boundarytype, simflags, PT_FLUID, PT_BOUNDARY>;

	FluidFluidParams params_ff(
		bufread, bufwrite,
		fromParticle, toParticle,
		deltap, slength, influenceradius, step, dt,
		epsilon,
		IOwaterdepth);

	execute_kernel(cuforces::repackDevice<FluidFluidParams>(params_ff), numBlocks, numThreads, dummy_shared);

	{
		FluidVertexParams params_fv(
			bufread, bufwrite,
			fromParticle, toParticle,
			deltap, slength, influenceradius, step, dt,
			epsilon,
			IOwaterdepth);

		vertex_repack(numBlocks, numThreads, dummy_shared, params_fv);
	}

	FluidBoundaryParams params_fb(
		bufread, bufwrite,
		fromParticle, toParticle,
		deltap, slength, influenceradius, step, dt,
		epsilon,
		IOwaterdepth);

	execute_kernel(cuforces::repackDevice<FluidBoundaryParams>(params_fb), numBlocks, numThreads, dummy_shared);

	using FinalizeRepackParams = finalize_repack_params<boundarytype, simflags>;
	FinalizeRepackParams params_finalize(
		bufread, bufwrite,
		numParticles, fromParticle, toParticle, slength, deltap,
		cflOffset,
		IOwaterdepth);

	execute_kernel(cuforces::finalizeRepackDevice<FinalizeRepackParams>(params_finalize),
		numBlocks, numThreads, dummy_shared);

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
		space_w_t temp;
		void * ddata = (void *) (forces + lastindex[i]);
		COPY_FROM_DEVICE(&temp, ddata, 1);
		totalforce[i] = temp.xyz;

		ddata = (void *) (torques + lastindex[i]);
		COPY_FROM_DEVICE(&temp, ddata, 1);
		totaltorque[i] = temp.xyz;
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
	int dummy_shared = 0;
	// thread per particle
	uint numThreads = BLOCK_SIZE_SHEPARD;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	// execute the kernel
	#if (__COMPUTE__ >= 20)
	dummy_shared = 2560;
	#endif

	if (boundarytype == SA_BOUNDARY)
		throw std::runtime_error("Shepard filtering is not supported with SA_BOUNDARY");

	execute_kernel(cuforces::shepardDevice<kerneltype, boundarytype>(
		bufread, bufwrite, numParticles, slength, influenceradius),
		numBlocks, numThreads, dummy_shared);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
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
	int dummy_shared = 0;
	// thread per particle
	uint numThreads = BLOCK_SIZE_MLS;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	// execute the kernel
	#if (__COMPUTE__ >= 20)
	dummy_shared = 2560;
	#endif

	if (boundarytype == SA_BOUNDARY)
		throw std::runtime_error("MLS filtering is not supported with SA_BOUNDARY");

	execute_kernel(cuforces::MlsDevice<kerneltype, boundarytype>
		(bufread, bufwrite, numParticles, slength, influenceradius),
		numBlocks, numThreads, dummy_shared);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
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


