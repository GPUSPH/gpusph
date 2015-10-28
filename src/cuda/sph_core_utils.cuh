/* IMPORTANT NOTE: this header should be included _inside_ the namespace,
 * of each _kernel.cu file, because of the __constant__s defined below
 */

/* This header file contains core functions of SPH such as the weighting functions,
 * its derivative, as well as the EOS (including inverse) and the methods to compute
 * Riemann invariants
 */

////////////////////
// V A R I A B L E S
////////////////////

//< Kernel normalization constants
__constant__ float	d_wcoeff_cubicspline;			//< coeff = 1/(Pi h^3)
__constant__ float	d_wcoeff_quadratic;				//< coeff = 15/(16 Pi h^3)
__constant__ float	d_wcoeff_wendland;				//< coeff = 21/(16 Pi h^3)

/*! Gaussian kernel is in the form (exp(-r/h)^2 - S)/K, so we have two constants:
  d_wsub_gaussian = S which is exp(-R^2), and the normalization constant
  d_wcoeff_gaussian = 1/K
 */
__constant__ float	d_wsub_gaussian;
__constant__ float	d_wcoeff_gaussian;

//< Kernel derivative normalization constants
__constant__ float	d_fcoeff_cubicspline;			//< coeff = 3/(4Pi h^4)
__constant__ float	d_fcoeff_quadratic;				//< coeff = 15/(32Pi h^4)
__constant__ float	d_fcoeff_wendland;				//< coeff = 105/(128Pi h^5)
__constant__ float	d_fcoeff_gaussian;				//< coeff = wcoeff * 2/h^2

// physical constants
__constant__ float	d_rho0[MAX_FLUID_TYPES];		// rest density of fluids
__constant__ float3	d_gravity;						// gravity (vector)
// speed of sound constants
__constant__ float	d_bcoeff[MAX_FLUID_TYPES];		// \rho_0 c_0^2 / \gamma
__constant__ float	d_gammacoeff[MAX_FLUID_TYPES];	// \gamma
__constant__ float	d_sscoeff[MAX_FLUID_TYPES];		// c_0
__constant__ float	d_sspowercoeff[MAX_FLUID_TYPES];// (\gamma - 1)/2

////////////////////
// F U N C T I O N S
////////////////////

/********************************************* SPH kernels **************************************************/
// Return kernel value at distance r, for a given smoothing length
template<KernelType kerneltype>
__device__ __forceinline__ float
W(const float r, const float slength);


// Cubic Spline kernel
template<>
__device__ __forceinline__ float
W<CUBICSPLINE>(const float r, const float slength)
{
	float val = 0.0f;
	const float R = r/slength;

	if (R < 1)
		val = 1.0f - 1.5f*R*R + 0.75f*R*R*R;			// val = 1 - 3/2 R^2 + 3/4 R^3
	else
		val = 0.25f*(2.0f - R)*(2.0f - R)*(2.0f - R);	// val = 1/4 (2 - R)^3

	val *= d_wcoeff_cubicspline;						// coeff = 1/(Pi h^3)

	return val;
}


// Qudratic kernel
template<>
__device__ __forceinline__ float
W<QUADRATIC>(const float r, const float slength)
{
	float val = 0.0f;
	const float R = r/slength;

	val = 0.25f*R*R - R + 1.0f;		// val = 1/4 R^2 -  R + 1
	val *= d_wcoeff_quadratic;		// coeff = 15/(16 Pi h^3)

	return val;
}


// Wendland kernel
template<>
__device__ __forceinline__ float
W<WENDLAND>(const float r, const float slength)
{
	const float R = r/slength;

	float val = 1.0f - 0.5f*R;
	val *= val;
	val *= val;						// val = (1 - R/2)^4
	val *= 1.0f + 2.0f*R;			// val = (2R + 1)(1 - R/2)^4*
	val *= d_wcoeff_wendland;		// coeff = 21/(16 Pi h^3)
	return val;
}


// Gaussia kernel
// W(r, h) = (exp(-(r/h)^2) - exp(-(δ/h)^2))*const
// with δ cut-off radius (i.e. influence radius) (typically, 3h),
// and const normalization constant
template<>
__device__ __forceinline__ float
W<GAUSSIAN>(float r, float slength)
{
	const float R = r/slength;

	float val = expf(-R*R);
	val -= d_wsub_gaussian;
	val *= d_wcoeff_gaussian;
	return val;
}


// Return 1/r dW/dr at distance r, for a given smoothing length
template<KernelType kerneltype>
__device__ __forceinline__ float
F(const float r, const float slength);


template<>
__device__ __forceinline__ float
F<CUBICSPLINE>(const float r, const float slength)
{
	float val = 0.0f;
	const float R = r/slength;

	if (R < 1.0f)
		val = (-4.0f + 3.0f*R)/slength;		// val = (-4 + 3R)/h
	else
		val = -(-2.0f + R)*(-2.0f + R)/r;	// val = -(-2 + R)^2/r
	val *= d_fcoeff_cubicspline;			// coeff = 3/(4Pi h^4)

	return val;
}


template<>
__device__ __forceinline__ float
F<QUADRATIC>(const float r, const float slength)
{
	const float R = r/slength;

	float val = (-2.0f + R)/r;		// val = (-2 + R)/r
	val *= d_fcoeff_quadratic;		// coeff = 15/(32Pi h^4)

	return val;
}


template<>
__device__ __forceinline__ float
F<WENDLAND>(const float r, const float slength)
{
	const float qm2 = r/slength - 2.0f;	// val = (-2 + R)^3
	float val = qm2*qm2*qm2*d_fcoeff_wendland;
	return val;
}


template<>
__device__ __forceinline__ float
F<GAUSSIAN>(const float r, const float slength)
{
	const float R = r/slength;
	float val = -expf(-R*R)*d_fcoeff_gaussian;
	return val;
}

/************************************************************************************************************/


/********************** Equation of state, speed of sound, repulsive force **********************************/
// Equation of state: pressure from density, where i is the fluid kind, not particle_id
__device__ __forceinline__ float
P(const float rho, const ushort i)
{
	return d_bcoeff[i]*(__powf(rho/d_rho0[i], d_gammacoeff[i]) - 1.0f);
}

// Inverse equation of state: density from pressure, where i is the fluid kind, not particle_id
__device__ __forceinline__ float
RHO(const float p, const ushort i)
{
	return __powf(p/d_bcoeff[i] + 1.0f, 1.0f/d_gammacoeff[i])*d_rho0[i];
}

// Riemann celerity
__device__ float
R(const float rho, const ushort i)
{
	return 2.0f/(d_gammacoeff[i]-1.0f)*d_sscoeff[i]*__powf(rho/d_rho0[i], 0.5f*d_gammacoeff[i]-0.5f);
}

// Density from Riemann celerity
__device__ __forceinline__ float
RHOR(const float r, const ushort i)
{
	return d_rho0[i]*__powf((d_gammacoeff[i]-1.)*r/(2.*d_sscoeff[i]), 2./(d_gammacoeff[i]-1.));
}

// Sound speed computed from density
__device__ __forceinline__ float
soundSpeed(const float rho, const ushort i)
{
	return d_sscoeff[i]*__powf(rho/d_rho0[i], d_sspowercoeff[i]);
}
