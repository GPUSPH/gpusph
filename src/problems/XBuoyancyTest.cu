#include "XBuoyancyTest.h"
#include <cmath>
#include <iostream>

#include "GlobalData.h"
#include "cudasimframework.cu"
#include "Cube.h"
#include "Sphere.h"
#include "Point.h"
#include "Vector.h"


XBuoyancyTest::XBuoyancyTest(GlobalData *_gdata) : XProblem(_gdata)
{
	// Size and origin of the simulation domain
	double lx = 1.0;
	double ly = 1.0;
	double lz = 1.0;
	double H = 0.7;

	//m_size = make_double3(lx, ly, lz);
	//m_origin = make_double3(0.0, 0.0, 0.0);

	SETUP_FRAMEWORK(
		kernel<WENDLAND>,
		viscosity<ARTVISC>,
		//viscosity<SPSVISC>,
		//viscosity<KINEMATICVISC>,
		boundary<DYN_BOUNDARY>
	);

	// let the number of layers be autocomputed
	//setDynamicBoundariesLayers(4);

	// SPH parameters
	set_deltap(0.02); //0.008
	simparams()->dt = 0.0003f;
	simparams()->dtadaptfactor = 0.3;
	simparams()->buildneibsfreq = 10;
	simparams()->tend = 20.0f; //0.00036f

	// Physical parameters
	H = 0.6f;
	physparams()->gravity = make_float3(0.0, 0.0, -9.81f);
	double g = length(physparams()->gravity);
	add_fluid(1000.0);
	set_equation_of_state(0,  7.0f, 20.f);

	//set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5
	physparams()->dcoeff = 5.0f*g*H;
	physparams()->r0 = m_deltap;

	set_kinematic_visc(0, 1.0e-6f);
	physparams()->artvisccoeff = 0.3f;
	physparams()->epsartvisc = 0.01*simparams()->slength*simparams()->slength;

	add_writer(VTKWRITER, 0.01);

	setPositioning(PP_CORNER);

	const int layers = 4;

	GeometryID cube = addBox(GT_FIXED_BOUNDARY, FT_BORDER, Point(0,0,0), lx, ly, lz);
	disableCollisions(cube);

	const double offs = m_deltap * layers;
	//addExtraWorldMargin(2*offs);

	GeometryID fluid = addBox(GT_FLUID, FT_SOLID, Point(offs, offs, offs),
		lx - 2.0 * offs, ly - 2.0 * offs, H);

	// TODO
	/*
	switch (object_type) {
		case 0: {
			olx, oly, olz = 10.0*m_deltap;
			cube  = Cube(Point(lx/2.0 - olx/2.0, ly/2.0 - oly/2.0, H/2.0 - olz/2.0), olx, oly, olz);
		case 1: {
			double r = 6.0*m_deltap;
			sphere = Sphere(Point(lx/2.0, ly/2.0, H/2.0 - r/4.0), r);
		case 2: // TORUS
	*/
	double R = lx * 0.2;
	double r = 4.0 * m_deltap;
	GeometryID torus = addTorus(GT_FLOATING_BODY, FT_BORDER, Point(lx/2.0, ly/2.0, H/2.0), R, r);
	setMassByDensity(torus, physparams()->rho0[0]*0.5);

	// Name of problem used for directory creation
	m_name = "XBuoyancyTest";
}

void XBuoyancyTest::ODE_near_callback(void *data, dGeomID o1, dGeomID o2)
{
	const int N = 10;
	dContact contact[N];

	int n = dCollide(o1, o2, N, &contact[0].geom, sizeof(dContact));
	for (int i = 0; i < n; i++) {
		contact[i].surface.mode = dContactBounce;
		contact[i].surface.mu   = dInfinity;
		contact[i].surface.bounce     = 0.0; // (0.0~1.0) restitution parameter
		contact[i].surface.bounce_vel = 0.0; // minimum incoming velocity for bounce
		dJointID c = dJointCreateContact(m_ODEWorld, m_ODEJointGroup, &contact[i]);
		dJointAttach (c, dGeomGetBody(contact[i].geom.g1), dGeomGetBody(contact[i].geom.g2));
	}
}
