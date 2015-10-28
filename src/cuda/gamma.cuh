/************************************************************************************************************/
/*		Gamma calculations																					*/
/************************************************************************************************************/

// Single-precision M_PI
// FIXME : ah, ah ! Single precision with 976896587958795795 decimals ....
#define M_PIf 3.141592653589793238462643383279502884197169399375105820974944f

////////////////////////////////
// Gaussian quadrature constants
////////////////////////////////

// 5th order
////////////

//! Gaussian quadrature 5th order: weights
__constant__ float GQ_O5_weights[3] = {0.225f, 0.132394152788506f, 0.125939180544827f};

//! Gaussian quadrature 5th order: points, in barycentric coordinates
__constant__ float GQ_O5_points[3][3] = {
	{0.333333333333333f, 0.333333333333333f, 0.333333333333333f},
	{0.059715871789770f, 0.470142064105115f, 0.470142064105115f},
	{0.797426985353087f, 0.101286507323456f, 0.101286507323456f}
};

//! Gaussian quadrature 5th order: multiplicity of each quadrature point
__constant__ int GQ_O5_mult[3] = {1, 3, 3};

// 14th order
/////////////

//! Gaussian quadrature 14th order: weights
__constant__ float GQ_O14_weights[10] = {
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
__constant__ float GQ_O14_points[10][3] = {
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
__constant__ int GQ_O14_mult[10] = {1,3,3,3,3,3,6,6,6,6};

//! Obtains old (grad)gamma value
/*
 Load old gamma value.
 If computeGamma was false, it means the caller wants us to check gam.w against epsilon
 to see if the new gamma is to be computed
*/
__device__ __forceinline__
float4
fetchOldGamma(const uint index, const float epsilon, bool &computeGamma)
{
	float4 gam = tex1Dfetch(gamTex, index);
	if (!computeGamma)
		computeGamma = (gam.w < epsilon);
	return gam;
}

//! This function returns the function value of the wendland kernel and of the integrated wendland kernel
__device__ __forceinline__
float2
wendlandOnSegment(const float q)
{
	float kernel = 0.0f;
	float intKernel = 0.0f;

	if (q < 2.0f) {
		float tmp = (1.0f-q/2.0f);
		float tmp4 = tmp*tmp;
		tmp4 *= tmp4;

// Wendland coefficient: 21/(16 π)
#define WENDLAND_K_COEFF 0.417781725616225256393319878852850200340456570068698177962626f
// Integrated Wendland coefficient: 1/(32 π)
#define WENDLAND_I_COEFF 0.009947183943243458485555235210782147627153727858778528046729f

		// Wendland kernel
		kernel = WENDLAND_K_COEFF*tmp4*(1.0f+2.0f*q);

		// integrated Wendland kernel
		const float uq = 1.0f/q;
		intKernel = WENDLAND_I_COEFF*tmp4*tmp*((((8.0f*uq + 20.0f)*uq + 30.0f)*uq) + 21.0f);
	}

	return make_float2(kernel, intKernel);
}

/*
 * Gaussian quadrature
 */

//! Function that computes the surface integral of a function on a triangle using a 1st order Gaussian quadrature rule
__device__ __forceinline__
float2
gaussQuadratureO1(	const	float3	vPos0,
					const	float3	vPos1,
					const	float3	vPos2,
					const	float3	relPos)
{
	float2 val = make_float2(0.0f);
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
float2
gaussQuadratureO5(	const	float3	vPos0,
					const	float3	vPos1,
					const	float3	vPos2,
					const	float3	relPos)
{
	float2 val = make_float2(0.0f);
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
float2
gaussQuadratureO14(	const	float3	vPos0,
					const	float3	vPos1,
					const	float3	vPos2,
					const	float3	relPos)
{
	float2 val = make_float2(0.0f);
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

template<KernelType kerneltype>
__device__ __forceinline__ float
gradGamma(	const	float		&slength,
		const	float3		&relPos,
		const	float3		*vertexRelPos,
		const	float3		&ns)
{ return -1.0f; } // TODO throw not implemented error

template<>
__device__ __forceinline__ float
gradGamma<WENDLAND>(
		const	float		&slength,
		const	float3		&relPos,
		const	float3		*vertexRelPos,
		const	float3		&ns)
{
	// Sigma is the point a projected onto the plane spanned by the edge
	// pas: is the algebraic distance of the particle a to the plane
	// qas: is the distance of the particle a to the plane
	float pas = dot(ns, relPos)/slength;
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
		float pae = dot(ne, relPos - vertexRelPos[sIdx[0]])/slength;

		// NB: in the reference frame (ns, v01, ne) it reads:
 		//     r_av0/h = pas * ns + pae * ne + pav0 * v01
 		//     r_av1/h = pas * ns + pae * ne + pav1 * v01
 		//     qae^2 = pas^2 + pae^2
 		//     qav0^2 = qae^2 + pav0^2

		// distance from the particle a to the edge
		float qae = length(pas * ns + pae * ne);

		float pav0 = -dot(relPos - vertexRelPos[sIdx[0]], v01)/slength;
		float pav1 = -dot(relPos - vertexRelPos[sIdx[1]], v01)/slength;

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

//! Computes (grad)gamma_{as}
/*!
 gamma_{as} is computed for fluid and vertex particles using a Gaussian quadrature rule.
 grad gamma_{as} is computed using an analytical formula.
 returns grad gamma_{as} as x coordinate, gamma_{as} as y coordinate.
*/
template<KernelType kerneltype>
__device__ __forceinline__
float
Gamma(	const	float		&slength,
				float4		relPos,
		const	float2		&vPos0,
		const	float2		&vPos1,
		const	float2		&vPos2,
		const	float4		&boundElement,
				float4		oldGGam,
		const	float		&epsilon,
		const	float		&deltap,
		const	bool		&computeGamma,
		const	uint		&nIndex,
				float		&minlRas)
{
	// normalize the distance r_{as} with h
	relPos.x /= slength;
	relPos.y /= slength;
	relPos.z /= slength;
	// Sigma is the point a projected onto the plane spanned by the edge
	// q_aSigma is the non-dimensionalized distance between this plane and the particle
	float4 q_aSigma = boundElement*dot3(boundElement,relPos);
	q_aSigma.w = fmin(length3(q_aSigma),2.0f);
	// local coordinate system for relative positions to vertices
	uint j = 0;
	// Get index j for which n_s is minimal
	if (fabs(boundElement.x) > fabs(boundElement.y))
		j = 1;
	if ((1-j)*fabs(boundElement.x) + j*fabs(boundElement.y) > fabs(boundElement.z))
		j = 2;

	// compute the first coordinate which is a 2-D rotated version of the normal
	const float4 coord1 = normalize(make_float4(
		// switch over j to give: 0 -> (0, z, -y); 1 -> (-z, 0, x); 2 -> (y, -x, 0)
		-((j==1)*boundElement.z) +  (j == 2)*boundElement.y , // -z if j == 1, y if j == 2
		  (j==0)*boundElement.z  - ((j == 2)*boundElement.x), // z if j == 0, -x if j == 2
		-((j==0)*boundElement.y) +  (j == 1)*boundElement.x , // -y if j == 0, x if j == 1
		0));
	// the second coordinate is the cross product between the normal and the first coordinate
	const float4 coord2 = cross3(boundElement, coord1);

	// relative positions of vertices with respect to the segment, normalized by h
	float4 v0 = -(vPos0.x*coord1 + vPos0.y*coord2)/slength; // e.g. v0 = r_{v0} - r_s
	float4 v1 = -(vPos1.x*coord1 + vPos1.y*coord2)/slength;
	float4 v2 = -(vPos2.x*coord1 + vPos2.y*coord2)/slength;
	// calculate if the projection of a (with respect to n) is inside the segment
	const float4 ba = v1 - v0; // vector from v0 to v1
	const float4 ca = v2 - v0; // vector from v0 to v2
	const float4 pa = relPos - v0; // vector from v0 to the particle
	const float uu = sqlength3(ba);
	const float uv = dot3(ba,ca);
	const float vv = sqlength3(ca);
	const float wu = dot3(ba,pa);
	const float wv = dot3(ca,pa);
	const float invdet = 1.0f/(uv*uv-uu*vv);
	const float u = (uv*wv-vv*wu)*invdet;
	const float v = (uv*wu-uu*wv)*invdet;
	//const float w = 1.0f - u - v;
	// set minlRas only if the projection is close enough to the triangle and if the normal
	// distance is close
	if (q_aSigma.w < 0.5f && (u > -0.5f && v > -0.5f && 1.0f - u - v > -0.5f && u < 1.5f && v < 1.5f && 1.0f - u - v < 1.5f)) {
		minlRas = min(minlRas, q_aSigma.w);
	}
	float gamma_as = 0.0f;
	float gamma_vs = 0.0f;
	// check if the particle is on a vertex
	if ((	(fabs(u-1.0f) < epsilon && fabs(v) < epsilon) ||
			(fabs(v-1.0f) < epsilon && fabs(u) < epsilon) ||
			(     fabs(u) < epsilon && fabs(v) < epsilon)   ) && q_aSigma.w < epsilon) {
		// set touching vertex to v0
		if (fabs(u-1.0f) < epsilon && fabs(v) < epsilon) {
			const float4 tmp = v1;
			v1 = v2;
			v2 = v0;
			v0 = tmp;
		}
		else if (fabs(v-1.0f) < epsilon && fabs(u) < epsilon) {
			const float4 tmp = v2;
			v2 = v1;
			v1 = v0;
			v0 = tmp;
		}
		// compute the sum of all solid angles of the tetrahedron spanned by v1-v0, v2-v0 and -gradgamma
		// the minus is due to the fact that initially gamma is equal to one, so we want to subtract the outside
		oldGGam /= -fmax(length3(oldGGam),slength*1e-3f);
		float l1 = length3(v1-v0);
		float l2 = length3(v2-v0);
		float abc = dot3((v1-v0),oldGGam)/l1 + dot3((v2-v0),oldGGam)/l2 + dot3((v1-v0),(v2-v0))/l1/l2;
		float d = dot3(oldGGam,cross3((v1-v0),(v2-v0)))/l1/l2;

		// formula by A. Van Oosterom and J. Strackee “The Solid Angle of a Plane Triangle”, IEEE Trans. Biomed. Eng. BME-30(2), 125-126 (1983)
		float SolidAngle = fabs(2.0f*atan2(d,(1.0f+abc)));
		gamma_vs = SolidAngle*0.079577471545947667884441881686257181017229822870228224373833f; // 1/(4π)
	}
	// check if particle is on an edge
	else if ((	(fabs(u) < epsilon && v > -epsilon && v < 1.0f+epsilon) ||
				(fabs(v) < epsilon && u > -epsilon && u < 1.0f+epsilon) ||
				(fabs(u+v-1.0f) < epsilon && u > -epsilon && u < 1.0f+epsilon && v > -epsilon && v < 1.0f+epsilon)
			 ) && q_aSigma.w < epsilon) {
		oldGGam /= -length3(oldGGam);

		// compute the angle between a segment and -gradgamma
		const float theta0 = acos(dot3(boundElement,oldGGam)); // angle of the norms between 0 and pi
		const float4 refDir = cross3(boundElement, relPos); // this defines a reference direction
		const float4 normDir = cross3(boundElement, oldGGam); // this is the sin between the two norms
		const float theta = M_PIf + copysign(theta0, dot3(refDir, normDir)); // determine the actual angle based on the orientation of the sin

		// this is actually two times gamma_as:
		gamma_vs = theta*0.1591549430918953357688837633725143620344596457404564f; // 1/(2π)
	}
	// general formula (also used if particle is on vertex / edge to compute remaining edges)
	if (q_aSigma.w < 2.0f && q_aSigma.w > epsilon) {
		// Gaussian quadrature of 14th order
		//float2 intVal = gaussQuadratureO1(-as_float3(v0), -as_float3(v1), -as_float3(v2), as_float3(relPos));
		// Gaussian quadrature of 14th order
		//float2 intVal = gaussQuadratureO14(-as_float3(v0), -as_float3(v1), -as_float3(v2), as_float3(relPos));
		// Gaussian quadrature of 5th order
		const float2 intVal = gaussQuadratureO5(-as_float3(v0), -as_float3(v1), -as_float3(v2), as_float3(relPos));
		gamma_as += intVal.y*dot3(boundElement,q_aSigma);
	}
	gamma_as = gamma_vs + gamma_as;
	const float3 vertexRelPos[3] = {as_float3(v0), as_float3(v1), as_float3(v2)};
	return gamma_as;
}
