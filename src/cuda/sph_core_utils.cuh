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

template<KernelType kerneltype>
__device__ __forceinline__ float
gradGamma(
	const	float3	&segmentRelPos,
	const	float3	*vertexRelPos,
	const	float3	&ns,
	const	float	&slength)
{ return -1.0f; } // TODO throw not implemented error

template<>
__device__ __forceinline__ float
gradGamma<WENDLAND>(
	const	float3	&segmentRelPos,		// r_{as} vector s to a (dimensional)
	const	float3	*vertexRelPos,		// r_{v_i,s} vector s to v_i (dimensional)
	const	float3	&ns,				// n_s normal
	const	float	&slength)			// smoothing length
{
	// Sigma is the point a projected onto the plane spanned by the edge
	// pas: is the algebraic distance of the particle a to the plane
	// qas: is the distance of the particle a to the plane
	float pas = dot(ns, segmentRelPos)/slength;
	float qas = fabs(pas);

	if (qas >= 2.f)
		return 0.f;

	float qas2 = qas*qas;
	float qas3 = qas2*qas;
	float qas4 = qas2*qas2;
	float qas5 = qas3*qas2;

	// Indirection sorted array of vertex index
	uint sIdx[2];

	float gradGamma_as = 0.f;
	float totalSumAngles = 0.f;
	float sumAngles = 0.f;

	// general formula (also used if particle is on vertex / edge to compute remaining edges)

	// loop over all three edges
	for (uint e = 0; e < 3; e++) {
		sIdx[0] = e%3;
		sIdx[1] = (e+1)%3;

		float3 v01 = normalize(vertexRelPos[sIdx[0]] - vertexRelPos[sIdx[1]]);

		// ne is the vector pointing outward from segment, normal to segment and normal and v_{10}
		// this is only possible because initConnectivity makes sure that the segments are ordered correctly
		// i.e. anticlokewise sens when in the plane of the boundary segment with the normal pointing
		// in our direction.
		// NB: (ns, v01, ne) is an orthotropic reference fram
		float3 ne = normalize(cross(ns, v01));

		// algebraic distance of projection in the plane s to the edge
		// (negative if the projection is inside the triangle).
		float pae = dot(ne, segmentRelPos - vertexRelPos[sIdx[0]])/slength;
		// NB: in the reference frame (ns, v01, ne) it reads:
		//     r_av0/h = pas * ns + pae * ne + pav0 * v01
		//     r_av1/h = pas * ns + pae * ne + pav1 * v01
		//     qae^2 = pas^2 + pae^2
		//     qav0^2 = qae^2 + pav0^2

		// distance from the particle a to the edge
		float qae = length(pas * ns + pae * ne);

		float pav0 = -dot(segmentRelPos - vertexRelPos[sIdx[0]], v01)/slength;
		float pav1 = -dot(segmentRelPos - vertexRelPos[sIdx[1]], v01)/slength;

		// This is -2*pi if inside the segment, 0 otherwise
		totalSumAngles += copysign(atan2(pav1, fabs(pae))-atan2(pav0, fabs(pae)), pae);

		// if qae is greater than 2 the kernel support does not intersect the edge
		if (qae < 2.0f) {
			// Clip the point on the edge to a maximum distance of 2. if the vertex is
			// not in the influence radius
			pav0 = copysign(fmin(fabs(pav0), sqrt(4.0f - qae*qae)), pav0);
			float pav02 = pav0*pav0;
			pav1 = copysign(fmin(fabs(pav1), sqrt(4.0f - qae*qae)), pav1);
			float pav12 = pav1*pav1;

			// Distance from particle a to the vertices OR the points on the edge
			// which are at a distance of the influence radius
			float qav0 = fmin(sqrt(qae*qae + pav0*pav0), 2.0f);
			float qav1 = fmin(sqrt(qae*qae + pav1*pav1), 2.0f);

			float pae2 = pae*pae;
			float pae4 = pae2*pae2;
			float pae6 = pae4*pae2;

			gradGamma_as += 1.0f/2048.0f/M_PI*(
				+ 48.0f*qas5*(28.0f+qas2)*(
						  atan2(qas*pav1, pae*qav1) - atan2(pav1, pae)
						-(atan2(qas*pav0, pae*qav0) - atan2(pav0, pae)))
				+ pae*(

					 pav1*(3.0f*qas4*(-420.0f+29.0f*qav1)
						+pae4*(-420.0f+33.0f*qav1)
						+2.0f*qas2*(-210.0f*(8.0f+pav12)+756.0f*qav1+19.0f*pav12*qav1)
						+4.0f*(336.0f+pav12*(pav12*(-21.0f+2.0f*qav1)+28.0f*(-5.0f+3.0f*qav1)))
						+2.0f*pae2*(420.0f*(-2.0f+qav1)+6.0f*qas2*(-105.0f+8.0f*qav1)+pav12*(-140.0f+13.0f*qav1))
						)
					-pav0*(3.0f*qas4*(-420.0f+29.0f*qav0)
						+pae4*(-420.0f+33.0f*qav0)
						+2.0f*qas2*(-210.0f*(8.0f+pav02)+756.0f*qav0+19.0f*pav02*qav0)
						+4.0f*(336.0f+pav02*(pav02*(-21.0f+2.0f*qav0)+28.0f*(-5.0f+3.0f*qav0)))
						+2.0f*pae2*(420.0f*(-2.0f+qav0)+6.0f*qas2*(-105.0f+8.0f*qav0)+pav02*(-140.0f+13.0f*qav0))
						)

					+3.0f*(5.0f*pae6+21.0f*pae4*(8.0f+qas2)+35.0f*pae2*qas2*(16.0f+qas2)+35.0f*qas4*(24.0f+qas2))
					*(
						 copysign(1.f, pav1)*acoshf(fmax(qav1/fmax(qae, 1e-7f), 1.f))
						-copysign(1.f, pav0)*acoshf(fmax(qav0/fmax(qae, 1e-7f), 1.f))
						)
					)
				);

			sumAngles += copysign(atan2(pav1, fabs(pae))-atan2(pav0, fabs(pae)), pae);

		}
	}

	// Additional term (with no singularity!) due to the possible presence of a clipping of a vertex
	// to a point on the edge, or even a non intersected edge
	gradGamma_as += (sumAngles-totalSumAngles)*3.0f/16.0f/M_PI*__powf(1.0f - qas/2.0f, 5.0f)*(2.0f+5.0f*qas+4.0f*qas2);

	return gradGamma_as/slength;
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
