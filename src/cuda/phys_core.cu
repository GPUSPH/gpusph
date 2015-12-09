namespace cuphys {
__constant__ int	d_numfluids;			///< number of different fluids

__constant__ float	d_sqC0[MAX_FLUID_TYPES];	///< square of sound speed for at-rest density for each fluid

__constant__ float	d_ferrari;				///< coefficient for Ferrari correction
__constant__ float	d_rhodiffcoeff;			///< coefficient for density diffusion

__constant__ float	d_epsinterface;			///< interface epsilon for simplified surface tension in Grenier

// LJ boundary repusion force comuting
__constant__ float	d_dcoeff;
__constant__ float	d_p1coeff;
__constant__ float	d_p2coeff;
__constant__ float	d_r0;

// Monaghan-Kaijar boundary repulsion force constants
__constant__ float	d_MK_K;		///< This is typically the square of the maximum velocity, or gravity times the maximum height
__constant__ float	d_MK_d;		///< This is typically the distance between boundary particles
__constant__ float	d_MK_beta;	///< This is typically the ration between h and the distance between boundary particles

__constant__ float	d_visccoeff[MAX_FLUID_TYPES];	///< viscous coefficient
__constant__ float	d_epsartvisc;					///< epsilon of artificial viscosity

__constant__ float	d_partsurf;		///< particle surface (typically particle spacing suared)

// Sub-Particle Scale (SPS) Turbulence parameters
__constant__ float	d_smagfactor;
__constant__ float	d_kspsfactor;

// Free surface detection
__constant__ float	d_cosconeanglefluid;
__constant__ float	d_cosconeanglenonfluid;

// physical constants
__constant__ float	d_rho0[MAX_FLUID_TYPES];		// rest density of fluids
__constant__ float3	d_gravity;						// gravity (vector)
// speed of sound constants
__constant__ float	d_bcoeff[MAX_FLUID_TYPES];		// \rho_0 c_0^2 / \gamma
__constant__ float	d_gammacoeff[MAX_FLUID_TYPES];	// \gamma
__constant__ float	d_sscoeff[MAX_FLUID_TYPES];		// c_0
__constant__ float	d_sspowercoeff[MAX_FLUID_TYPES];// (\gamma - 1)/2


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
}
