
#include <iostream>
#include "TurbulentPoiseuilleFlowSA.h"
#include "GlobalData.h"
#include "cudasimframework.cu"
#include "textures.cuh"
#include "utils.h"
#include <string>

TurbulentPoiseuilleFlowSA::TurbulentPoiseuilleFlowSA(GlobalData *_gdata) : XProblem(_gdata)
{
	SETUP_FRAMEWORK(
		kernel<WENDLAND>,
		formulation<SPH_F1>,
		viscosity<KEPSVISC>,
		boundary<SA_BOUNDARY>,
		periodicity<PERIODIC_XY>,
		densitydiffusion<BREZZI>,
		add_flags<ENABLE_DTADAPT | ENABLE_DENSITY_SUM>
	);

	// *** Initialization of minimal physical parameters
	set_deltap(0.025);
	set_smoothing(1.3);
	set_gravity(1.0, 0.0, 0.0);

	// *** Initialization of minimal simulation parameters
	resize_neiblist(128+128, 128);
	simparams()->dtadaptfactor = 0.3;
	// *** ferrari correction
	simparams()->densityDiffCoeff = 0.1;

	// *** Other parameters and settings
	simparams()-> tend = 100.0f;
	add_writer(VTKWRITER, 1.0f);
	m_name = "TurbulentPoiseuilleFlowSA";

	m_origin = make_double3(-0.25, -0.25, -1.0);
	m_size = make_double3(0.5, 0.5, 2);

	//*** Fluid and Thermodynamic Properties
	add_fluid(1000.0);
	set_kinematic_visc(0, 0.0015625f);
	set_equation_of_state(0, 7.0f,40.0 );

	//*** Add the Fluid
	addHDF5File(GT_FLUID, Point(0,0,0), "./data_files/TurbulentPoiseuilleFlowSA/0.TurbulentPoiseuilleFlowSA.fluid.h5sph", NULL);

	//*** Add the Main Container
	GeometryID container = addHDF5File(GT_FIXED_BOUNDARY, Point(0,0,0), "./data_files/TurbulentPoiseuilleFlowSA/0.TurbulentPoiseuilleFlowSA.boundary.kent0.h5sph", NULL);
	disableCollisions(container);

}

void
TurbulentPoiseuilleFlowSA::initializeParticles(BufferList &buffers, const uint numParticles)
{
	printf("k and epsilon initialization...\n");

	float4 *vel = buffers.getData<BUFFER_VEL>();
	float4 *eulerVel = buffers.getData<BUFFER_EULERVEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();
	double4 *pos = buffers.getData<BUFFER_POS_GLOBAL>();
	float *k = buffers.getData<BUFFER_TKE>();
	float *epsilon = buffers.getData<BUFFER_EPSILON>();
	float r0 = physparams()->r0;
	float nu = physparams()->visccoeff[0];
	for (uint i = 0; i < numParticles; i++) {
		if (FLUID(info[i])) {
			vel[i].x = logf(fmax(1.0-fabs(pos[i].z), 0.5*r0)/nu)/0.41+5.2; // kappa=0.41
			vel[i].y = 0.f;
			vel[i].z = 0.f;
		} else if (VERTEX(info[i]) || BOUNDARY(info[i])) {
			eulerVel[i].x = logf(fmax(1.0-fabs(pos[i].z), 0.5*r0)/nu)/0.41+5.2; // kappa=0.41
			eulerVel[i].y = 0.f;
			eulerVel[i].z = 0.f;
		}
		vel[i].w = atrest_density(0);
		if (k && epsilon) {
			k[i] = 1./sqrtf(0.09f); // C_mu = 0.09
			epsilon[i] = 1./0.41/max(1.- abs(pos[i].z), 0.5*r0);
		}
	}
}

