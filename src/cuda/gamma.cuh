/*  Copyright (c) 2015-2018 INGV, EDF, UniCT, JHU

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
/************************************************************************************************************/
/*		Gamma calculations																					*/
/************************************************************************************************************/

// Single-precision M_PI
// FIXME : ah, ah ! Single precision with 976896587958795795 decimals ....
#define M_PIf 3.141592653589793238462643383279502884197169399375105820974944f

////////////////////////////////
// Gaussian quadrature constants
////////////////////////////////

/* We can help Clang unroll loops by exposing the constants as constexpr values,
 * but nvcc doesn't like to use them in device code
 */
#if CLANG_CUDA
#define QUADRATURE_CONSTANT static constexpr
#else
#define QUADRATURE_CONSTANT __constant__
#endif


// 5th order
////////////

//! Gaussian quadrature 5th order: weights
QUADRATURE_CONSTANT float GQ_O5_weights[3] = {0.225f, 0.132394152788506f, 0.125939180544827f};

//! Gaussian quadrature 5th order: points, in barycentric coordinates
QUADRATURE_CONSTANT float GQ_O5_points[3][3] = {
	{0.333333333333333f, 0.333333333333333f, 0.333333333333333f},
	{0.059715871789770f, 0.470142064105115f, 0.470142064105115f},
	{0.797426985353087f, 0.101286507323456f, 0.101286507323456f}
};

//! Gaussian quadrature 5th order: multiplicity of each quadrature point
QUADRATURE_CONSTANT int GQ_O5_mult[3] = {1, 3, 3};

// 14th order
/////////////

//! Gaussian quadrature 14th order: weights
QUADRATURE_CONSTANT float GQ_O14_weights[10] = {
	0.021883581369429f,
	0.032788353544125f,
	0.051774104507292f,
	0.042162588736993f,
	0.014433699669777f,
	0.004923403602400f,
	0.024665753212564f,
	0.038571510787061f,
	0.014436308113534f,
	0.005010228838501f
};

//! Gaussian quadrature 14th order: points, in barycentric coordinates
QUADRATURE_CONSTANT float GQ_O14_points[10][3] = {
	{0.022072179275643f,0.488963910362179f,0.488963910362179f},
	{0.164710561319092f,0.417644719340454f,0.417644719340454f},
	{0.453044943382323f,0.273477528308839f,0.273477528308839f},
	{0.645588935174913f,0.177205532412543f,0.177205532412543f},
	{0.876400233818255f,0.061799883090873f,0.061799883090873f},
	{0.961218077502598f,0.019390961248701f,0.019390961248701f},
	{0.057124757403648f,0.172266687821356f,0.770608554774996f},
	{0.092916249356972f,0.336861459796345f,0.570222290846683f},
	{0.014646950055654f,0.298372882136258f,0.686980167808088f},
	{0.001268330932872f,0.118974497696957f,0.879757171370171f}
};

//! Gaussian quadrature 14th order: multiplicity of each quadrature point
QUADRATURE_CONSTANT int GQ_O14_mult[10] = {1,3,3,3,3,3,6,6,6,6};

//! This function returns the function value of the integrated wendland kernel
__device__ __forceinline__
float
wendlandOnSegment(const float q)
{
	float intKernel = 0.0f;

	if (q < 2.0f) {
		float tmp = (1.0f-q/2.0f);
		float tmp4 = tmp*tmp;
		tmp4 *= tmp4;

// Integrated Wendland coefficient: 1/(32 π)
#define WENDLAND_I_COEFF 0.009947183943243458485555235210782147627153727858778528046729f

		// integrated Wendland kernel
		const float uq = 1.0f/q;
		intKernel = WENDLAND_I_COEFF*tmp4*tmp*((((8.0f*uq + 20.0f)*uq + 30.0f)*uq) + 21.0f);
	}

	return intKernel;
}

/*
 * Gaussian quadrature
 */

//! Function that computes the surface integral of a function on a triangle using a 1st order Gaussian quadrature rule
__device__ __forceinline__
float
gaussQuadratureO1(	const	float3	vPos0,
					const	float3	vPos1,
					const	float3	vPos2,
					const	float3	relPos)
{
	float val = 0.0f;
	// perform the summation
	float3 pa =	vPos0/3.0f +
				vPos1/3.0f +
				vPos2/3.0f  ;
	pa -= relPos;
	val += 1.0f*wendlandOnSegment(length(pa));
	// compute the triangle volume
	const float vol = length(cross(vPos1-vPos0,vPos2-vPos0))/2.0f;
	// return the summed values times the volume
	return val*vol;
}

//! Function that computes the surface integral of a function on a triangle using a 5th order Gaussian quadrature rule
__device__ __forceinline__
float
gaussQuadratureO5(	const	float3	vPos0,
					const	float3	vPos1,
					const	float3	vPos2,
					const	float3	relPos)
{
	float val = 0.0f;
	// perform the summation
#pragma unroll
	for (int i=0; i<3; i++) {
#pragma unroll
		for (int j=0; j<3; j++) {
			float3 pa =	vPos0*GQ_O5_points[i][j]       +
						vPos1*GQ_O5_points[i][(j+1)%3] +
						vPos2*GQ_O5_points[i][(j+2)%3]  ;
			pa -= relPos;
			val += GQ_O5_weights[i]*wendlandOnSegment(length(pa));
			if (j >= GQ_O5_mult[i])
				break;
		}
	}
	// compute the triangle volume
	const float vol = length(cross(vPos1-vPos0,vPos2-vPos0))/2.0f;
	// return the summed values times the volume
	return val*vol;
}


//! Function that computes the surface integral of a function on a triangle using a 14th order Gaussian quadrature rule
__device__ __forceinline__
float
gaussQuadratureO14(	const	float3	vPos0,
					const	float3	vPos1,
					const	float3	vPos2,
					const	float3	relPos)
{
	float val = 0.0f;
	// perform the summation
#pragma unroll
	for (int i=0; i<10; i++) {
#pragma unroll
		for (int j=0; j<6; j++) {
			float3 pa =	vPos0*GQ_O14_points[i][j%3]       +
						vPos1*GQ_O14_points[i][(j+1+j/3)%3] +
						vPos2*GQ_O14_points[i][(j+2-j/3)%3]  ;
			pa -= relPos;
			val += GQ_O14_weights[i]*wendlandOnSegment(length(pa));
			if (j >= GQ_O14_mult[i])
				break;
		}
	}
	// compute the triangle volume
	const float vol = length(cross(vPos1-vPos0,vPos2-vPos0))/2.0f;
	// return the summed values times the volume
	return val*vol;
}

/// Compute the relative position of vertices to element barycenter
template<typename NormalT>
__device__ __forceinline__ void
calcVertexRelPos(
	float3 q_vb[3], /// Array where the results will be written
	const NormalT ns, /// normal of the segment (float3) or segment (float4, with normal in .xyz)
	const float2 vPos0,
	const float2 vPos1,
	const float2 vPos2,
	float slength)
{
	// local coordinate system for relative positions to vertices
	// vectors r_{v_i,s}
	uint j = 0;
	// Get index j for which n_s is minimal
	if (fabs(ns.x) > fabs(ns.y))
		j = 1;
	if ((1-j)*fabs(ns.x) + j*fabs(ns.y) > fabs(ns.z))
		j = 2;
	// compute the first coordinate which is a 2-D rotated version of the normal
	const float3 coord1 = normalize(make_float3(
			// switch over j to give: 0 -> (0, z, -y); 1 -> (-z, 0, x); 2 -> (y, -x, 0)
			-((j==1)*ns.z) +  (j == 2)*ns.y ,  // -z if j == 1, y if j == 2
			(j==0)*ns.z  - ((j == 2)*ns.x),  // z if j == 0, -x if j == 2
			-((j==0)*ns.y) +  (j == 1)*ns.x ));// -y if j == 0, x if j == 1
	// the second coordinate is the cross product between the normal and the first coordinate
	const float3 coord2 = cross3(ns, coord1);
	// relative positions of vertices with respect to the segment
	q_vb[0] = -(vPos0.x*coord1 + vPos0.y*coord2)/slength; // e.g. v0 = r_{v0} - r_s
	q_vb[1] = -(vPos1.x*coord1 + vPos1.y*coord2)/slength;
	q_vb[2] = -(vPos2.x*coord1 + vPos2.y*coord2)/slength;
}


/// Computes \f$ ||\nabla \gamma_{as}|| \f$
/*! Computes \f$ ||\nabla \gamma_{as}|| based on an analytical formula.
 *  This function is specialized only Wendland kernel due to the lack of analytical
 *  formula for other kernbels.
 *
 *	\param[in] slength : smoothing length
 *	\param[in] q : relative position of particle to the boundary element barycenter divided by slength
 *	\param[in] q_vb : relative position of vertices to element barycenter divided by slength
 *	\param[in] ns : inward (respect to fluid) normal vector of the boundary element
 *	\return \f$ ||\nabla \gamma_{as}|| \f$
  */
template<KernelType kerneltype>
__device__ __forceinline__ float
gradGamma(	const	float		slength,
		const	float3		&q,
		const	float3		*q_vb,
		const	float3		&ns);

template<>
__device__ __forceinline__ float
gradGamma<WENDLAND>(
		const	float		slength,
		const	float3		&q,
		const	float3		*q_vb,
		const	float3		&ns)
{
	// Sigma is the point a projected onto the plane spanned by the edge
	// pas: is the algebraic distance of the particle a to the plane
	// qas: is the distance of the particle a to the plane
	float pas = dot(ns, q);
	float qas = fabsf(pas);

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

	// loop over all three edges
	for (uint e = 0; e < 3; e++) {
		sIdx[0] = e%3;
		sIdx[1] = (e+1)%3;

		float3 v01 = normalize(q_vb[sIdx[0]] - q_vb[sIdx[1]]);
		// ne is the vector pointing outward from segment, normal to segment and normal and v_{10}
		// this is only possible because initConnectivity makes sure that the segments are ordered correctly
 		// i.e. anticlokewise sens when in the plane of the boundary segment with the normal pointing
		// in our direction.
 		// NB: (ns, v01, ne) is an orthotropic reference fram
		float3 ne = normalize(cross(ns, v01));

		// algebraic distance of projection in the plane s to the edge
 		// (negative if the projection is inside the triangle).
		float pae = dot(ne, q - q_vb[sIdx[0]]);

		// NB: in the reference frame (ns, v01, ne) it reads:
 		//     r_av0/h = pas * ns + pae * ne + pav0 * v01
 		//     r_av1/h = pas * ns + pae * ne + pav1 * v01
 		//     qae^2 = pas^2 + pae^2
 		//     qav0^2 = qae^2 + pav0^2

		// distance from the particle a to the edge
		float qae = length(pas * ns + pae * ne);

		float pav0 = -dot(q - q_vb[sIdx[0]], v01);
		float pav1 = -dot(q - q_vb[sIdx[1]], v01);

		// This is -2*pi if inside the segment, 0 otherwise
		totalSumAngles += copysignf(atan2f(pav1, fabsf(pae))-atan2f(pav0, fabsf(pae)), pae);

		// if qae is greater than 2 the kernel support does not intersect the edge
		if (qae < 2.0f) {
			// Clip the point on the edge to a maximum distance of 2. if the vertex is
			// not in the influence radius
			pav0 = copysignf(fminf(fabsf(pav0), sqrtf(4.0f - qae*qae)), pav0);
			float pav02 = pav0*pav0;
			pav1 = copysignf(fminf(fabsf(pav1), sqrtf(4.0f - qae*qae)), pav1);
			float pav12 = pav1*pav1;

			// Distance from particle a to the vertices OR the points on the edge
			// which are at a distance of the influence radius
			float qav0 = fminf(sqrtf(qae*qae + pav0*pav0), 2.0f);
			float qav1 = fminf(sqrtf(qae*qae + pav1*pav1), 2.0f);

			float pae2 = pae*pae;
			float pae4 = pae2*pae2;
			float pae6 = pae4*pae2;

#define COEFF 0.00015542474911f  // 1.0f/2048.0f/M_PI
			gradGamma_as += COEFF*(
				+ 48.0f*qas5*(28.0f+qas2)*(
						  atan2f(qas*pav1, pae*qav1) - atan2f(pav1, pae)
						-(atan2f(qas*pav0, pae*qav0) - atan2f(pav0, pae)))
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
						 copysignf(1.f, pav1)*acoshf(fmaxf(qav1/fmaxf(qae, 1e-7f), 1.f))
						-copysignf(1.f, pav0)*acoshf(fmaxf(qav0/fmaxf(qae, 1e-7f), 1.f))
						)
					)
				);
#undef COEFF

			sumAngles += copysignf(atan2f(pav1, fabsf(pae))-atan2f(pav0, fabsf(pae)), pae);

		}
	}

	// Additional term (with no singularity!) due to the possible presence of a clipping of a vertex
	// to a point on the edge, or even a non intersected edge
	const float tmp1 = 1.0f - qas/2.0f;
	float tmp2 = tmp1*tmp1;
	tmp2 *= tmp2*tmp1; // Now tmp2 = powf(1.0f - qas/2.0f, 5.0f)*
#define COEFF 0.05968310365947f // 3.0f/16.0f/M_PI
	gradGamma_as += (sumAngles-totalSumAngles)*COEFF*tmp2*(2.0f+5.0f*qas+4.0f*qas2);
#undef COEFF
	return gradGamma_as/slength;
}


/// Computes \f$ \gamma_{as} \f$
/*! Computes \f$ \gamma_{as} \f$ with a numerical quadrature.
 *  This function is specialized for Wendland kernel (the only one for which we have
 *  an analytical solution for \f$ \nabla \gamma_{as} \f$). In case we want to use
 *  other kernels we need to compute an analytical solution for the gradient and
 *  take care of the constant used for vertices that match the Wendland kernel.
 *  This function is also specialized on the particle type on which we compute
 *  \f$ \gamma_{as} \f$, the vertex case requiring the old \f$ \nabla \gamma_{as} \f$.
 *
 *	\param[in] slength : smoothing length
 *	\param[in] q : relative position of particle to the boundary element barycenter divided by slength
 *	\param[in] q_vb : relative position of vertices to element barycenter divided by slength
 *	\param[in] ns : inward (respect to fluid) normal vector of the boundary element
 *	\param[in] oldGGam : previous gradient of gamma
 *	\param[in] epsilon : threshold value used for singularity avoidance
 *	\return \f$ \gamma_{as} \f$
 *
 *	TODO:
 *	 - oldGGam is used only for PT_VERTEX specialization
 *	 - in the PT_VERTEX specialization the neihgboring particle is always a vertex or a centroid
 *	 so the tests should be simplified
 */
template<KernelType kerneltype, ParticleType cptype>
__device__ __forceinline__ float
Gamma(	const	float		slength,
		const	float3		&q,
				float3		*q_vb,
		const	float3		&ns,
		const	float3		&oldGGam,
		const	float		epsilon);

template<>
__device__ __forceinline__ float
Gamma<WENDLAND, PT_FLUID>(
		const	float		slength,
		const	float3		&q,
				float3		*q_vb,
		const	float3		&ns,
		const	float3		&oldGGam,
		const	float		epsilon)
{
	// Sigma is the point a projected onto the plane spanned by the edge
	// r_aSigma is the non-dimensionalized vector between this plane and the particle
	// q_aSigma is the clipped non-dimensionalized distance between this plane and the particle
	float3 r_aSigma = ns*dot(ns, q);
	float q_aSigma = fminf(length(r_aSigma), 2.0f);

	float gamma_as = 0.0f;

	// general formula (also used if particle is on vertex / edge to compute remaining edges)
	if (q_aSigma < 2.0f && q_aSigma > epsilon) {
		// To use Gaussian quadrature of 1st order
		// call gaussQuadratureO1
		// To use Gaussian quadrature of 14th order
		// call gaussQuadratureO14
		// To use Gaussian quadrature of 5th order
		// call gaussQuadratureO5
		const float intVal = gaussQuadratureO5(-q_vb[0], -q_vb[1], -q_vb[2], q);
		gamma_as = intVal*dot(ns, r_aSigma);
	}

	return gamma_as;
}

template<>
__device__ __forceinline__ float
Gamma<WENDLAND, PT_VERTEX>(
		const	float		slength,
		const	float3		&q,
				float3		*q_vb,
		const	float3		&ns,
		const	float3		&oldGGam,
		const	float		epsilon)
{
	// Sigma is the point a projected onto the plane spanned by the edge
	// r_aSigma is the non-dimensionalized vector between this plane and the particle
	// q_aSigma is the clipped non-dimensionalized distance between this plane and the particle
	float3 r_aSigma = ns*dot(ns, q);
	float q_aSigma = fminf(length(r_aSigma), 2.0f);

	// calculate if the projection of a (with respect to n) is inside the segment
	const float3 ba = q_vb[1] - q_vb[0]; // vector from v0 to v1
	const float3 ca = q_vb[2] - q_vb[0]; // vector from v0 to v2
	const float3 pa = q - q_vb[0]; // vector from v0 to the particle
	const float uu = sqlength(ba);
	const float uv = dot(ba, ca);
	const float vv = sqlength(ca);
	const float wu = dot(ba, pa);
	const float wv = dot(ca, pa);
	const float invdet = 1.0f/(uv*uv-uu*vv);
	const float u = (uv*wv-vv*wu)*invdet;
	const float v = (uv*wu-uu*wv)*invdet;

	float gamma_as = 0.0f;
	float gamma_vs = 0.0f;
	// check if the particle is on a vertex
	if ((	(fabsf(u-1.0f) < epsilon && fabsf(v) < epsilon) ||
			(fabsf(v-1.0f) < epsilon && fabsf(u) < epsilon) ||
			(     fabsf(u) < epsilon && fabsf(v) < epsilon)   ) && q_aSigma < epsilon) {
		// set touching vertex to v0
		if (fabsf(u-1.0f) < epsilon && fabsf(v) < epsilon) {
			const float3 tmp = q_vb[1];
			q_vb[1] = q_vb[2];
			q_vb[2] = q_vb[0];
			q_vb[0] = tmp;
		}
		else if (fabsf(v-1.0f) < epsilon && fabsf(u) < epsilon) {
			const float3 tmp = q_vb[2];
			q_vb[2] = q_vb[1];
			q_vb[1] = q_vb[0];
			q_vb[0] = tmp;
		}
		// compute the sum of all solid angles of the tetrahedron spanned by v1-v0, v2-v0 and -gradgamma
		// the minus is due to the fact that initially gamma is equal to one, so we want to subtract the outside
		const float3 inward_normal = -oldGGam/fmaxf(length(oldGGam), slength*1e-3f);
		const float l1 = length(q_vb[1] - q_vb[0]);
		const float l2 = length(q_vb[2] - q_vb[0]);
		const float abc = dot(q_vb[1] - q_vb[0] ,inward_normal)/l1
					+ dot(q_vb[2] - q_vb[0], inward_normal)/l2
					+ dot(q_vb[1] - q_vb[0], q_vb[2] - q_vb[0])/l1/l2;
		const float d = dot(inward_normal, cross(q_vb[1] - q_vb[0], q_vb[2] - q_vb[0]))/l1/l2;

		// formula by A. Van Oosterom and J. Strackee “The Solid Angle of a Plane Triangle”, IEEE Trans. Biomed. Eng. BME-30(2), 125-126 (1983)
		const float SolidAngle = fabsf(2.0f*atan2f(d, 1.0f + abc));
		gamma_vs = SolidAngle*0.079577471545947667884441881686257181017229822870228224373833f; // 1/(4π)
	}

	// general formula (also used if particle is on vertex / edge to compute remaining edges)
	if (q_aSigma < 2.0f && q_aSigma > epsilon) {
		// To use Gaussian quadrature of 1st order
		// call gaussQuadratureO1
		// To use Gaussian quadrature of 14th order
		// call gaussQuadratureO14
		// To use Gaussian quadrature of 5th order
		// call gaussQuadratureO5
		const float intVal = gaussQuadratureO5(-q_vb[0], -q_vb[1], -q_vb[2], q);
		gamma_as += intVal*dot(ns, r_aSigma);
	}
	gamma_as = gamma_vs + gamma_as;
	return gamma_as;
}

/* vim: set ft=cuda: */
