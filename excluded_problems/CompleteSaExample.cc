#include <math.h>
#include <string>
#include <iostream>

#include "CompleteSaExample.h"
#include "GlobalData.h"

#define USE_PLANES 0

CompleteSaExample::CompleteSaExample(GlobalData *_gdata) : Problem(_gdata)
{
	h5File.setFilename("sa/0.complete_sa_example.h5sph");

	container = STLMesh::load_stl("./meshes/CompleteSaExample_container_coarse.stl");
	cube = STLMesh::load_stl("./meshes/CompleteSaExample_cube_coarse.stl");

	SETUP_FRAMEWORK(
		boundary<SA_BOUNDARY>,
		viscosity<DYNAMICVISC>,
		flags<	ENABLE_DTADAPT |
			ENABLE_INLET_OUTLET |
			/* ENABLE_MOVING_BODIES | */
			ENABLE_FLOATING_BODIES>);

	m_simparams.numObjects = 2;

	m_simparams.sfactor=1.3f;
	set_deltap(0.02f);

	m_physparams.kinematicvisc = 1.0e-2f;
	m_physparams.gravity = make_float3(0.0, 0.0, -9.81);
	m_physparams.set_density(0, 1000.0, 7.0f, 70.0f);

	// ugh (cit.)
	m_simparams.maxneibsnum = 384;

	m_simparams.tend = 10.0;
	m_simparams.testpoints = false;
	m_simparams.surfaceparticle = false;
	m_simparams.savenormals = false;
	initial_water_level = 0.5;
	expected_final_water_level = INLET_WATER_LEVEL;
	// extra margin around the domain size
	const double MARGIN = 0.1;
	const double INLET_BOX_LENGTH = 0.25;
	// size of the main cube, exlcuding the inlet and any margin
	box_l = box_w = box_h = 1.0;
	// world size
	world_l = box_l + INLET_BOX_LENGTH + 2 * MARGIN; // length is 1 (box) + 0.2 (inlet box length)
	world_w = box_w + 2 * MARGIN;
	world_h = box_h + 2 * MARGIN;
	m_origin = make_double3(- INLET_BOX_LENGTH - MARGIN, - MARGIN, - MARGIN);
	m_simparams.calcPrivate = false;

	// SPH parameters
	// let the dt be autocomputed
	//m_simparams.dt = 0.00004f;
	m_simparams.xsph = false;
	m_simparams.dtadaptfactor = 0.3;
	m_simparams.buildneibsfreq = 1;
	m_simparams.ferrariLengthScale = 0.25f;
	m_simparams.mbcallback = false;
	m_simparams.nlexpansionfactor = 1.1;

	// Size and origin of the simulation domain
	m_size = make_double3(world_l, world_w ,world_h);

	// Drawing and saving times
	add_writer(VTKWRITER, 1e-2f);

	// will use only 1 ODE body: the floating/moving cube (container is fixed)
	allocate_ODE_bodies(1);
	dInitODE();
	// world setup
	m_ODEWorld = dWorldCreate(); // ODE world for dynamics
	m_ODESpace = dHashSpaceCreate(0); // ODE world for collisions
	m_ODEJointGroup = dJointGroupCreate(0);  // Joint group for collision detection
	// Set gravity（x, y, z)
	dWorldSetGravity(m_ODEWorld,
		m_physparams.gravity.x, m_physparams.gravity.y, m_physparams.gravity.z);

	// Name of problem used for directory creation
	m_name = "CompleteSaExample";
}

// for an exaple ODE_nearCallback see
// http://ode-wiki.org/wiki/index.php?title=Manual:_Collision_Detection#Collision_detection
void CompleteSaExample::ODE_near_callback(void * data, dGeomID o1, dGeomID o2)
{
	// ODE generates multiple candidate contact points. We should use at least 3 for cube-plane
	// interaction, the more the better (probably).
	// CHECK: any significant correlation between performance and MAX_CONTACTS?
	const int MAX_CONTACTS = 10;
	dContact contact[MAX_CONTACTS];

	// offset between dContactGeom-s of consecutive dContact-s in contact araray
	const uint skip_offset = sizeof(dContact);

	// Do no handle collisions between planes. With 1 floatinb cube, "skip collisions not involving the cube"
	if (o1 != cube->m_ODEGeom && o2 != cube->m_ODEGeom)
		return;

	// collide the candidate pair o1, o2
	int num_contacts = dCollide(o1, o2, MAX_CONTACTS, &contact[0].geom, skip_offset);

	// resulting collision points are treated by ODE as joints. We use them all
	for (int i = 0; i < num_contacts; i++) {
		contact[i].surface.mode = dContactBounce;
		contact[i].surface.mu   = dInfinity; //ER TODO friction coefficient
		contact[i].surface.bounce     = 0.0; // (0.0~1.0) restitution parameter
		contact[i].surface.bounce_vel = 0.0; // minimum incoming velocity for bounce
		dJointID c = dJointCreateContact(m_ODEWorld, m_ODEJointGroup, &contact[i]);
		dJointAttach (c, dGeomGetBody(contact[i].geom.g1), dGeomGetBody(contact[i].geom.g2));
	}
}

CompleteSaExample::~CompleteSaExample()
{
	dSpaceDestroy(m_ODESpace);
	dWorldDestroy(m_ODEWorld);
	dCloseODE();
}

int CompleteSaExample::fill_parts()
{
	/* If we needed an accurate collision detection between the container and the cube (i.e.
	 * using the actual container mesh instead of modelling it with 5 infinite planes), we'd
	 * have to create a geometry for the container as well. However, in this case the mesh
	 * should be "thick" (2 layers, with volume inside) and not flat. With a flat mesh, ODE
	 * detects the contained cube as penetrated into the container and thus colliding.
	 * Long story short: don't uncomment the following.
	 */
	//container->ODEGeomCreate(m_ODESpace, m_deltap);

	// cube density half water density
	const double water_density_fraction = 0.5F;
	const double cube_density = m_physparams.rho0[0] * water_density_fraction; // 1000 = water
	// setting mass after merge; see 0158cbc1
	cube->SetMass(m_deltap, cube_density);
	//cube->ODEBodyCreate(m_ODEWorld, m_deltap); // only dynamics
	//cube->ODEGeomCreate(m_ODESpace, m_deltap); // only collisions
	cube->ODEBodyCreate(m_ODEWorld, m_deltap, m_ODESpace); // dynamics + collisions
	// particles with object(info)-1==1 are associated with ODE object number 0
	m_ODEobjectId[2-1] = 0;
	add_ODE_body(cube);

	// planes modelling the tank, for the interaction with the cube
	m_box_planes[0] = dCreatePlane(m_ODESpace, 0.0, 0.0, 1.0, 0); // floor
	m_box_planes[1] = dCreatePlane(m_ODESpace, 1.0, 0.0, 0.0, 0); // YZ plane, lower X
	m_box_planes[2] = dCreatePlane(m_ODESpace, -1.0, 0.0, 0.0, - box_l); // YZ plane, higher X
	m_box_planes[3] = dCreatePlane(m_ODESpace, 0.0, 1.0, 0.0, 0); // XZ plane, lower y
	m_box_planes[4] = dCreatePlane(m_ODESpace, 0.0, -1.0, 0.0, - box_w); // XZ plane, higher Y

	// if for debug reason we need to test the position and verse of a plane, we can ask ODE to
	// compute the distance of a probe point from a plane (positive if penetrated, negative out)
	/*
	double3 probe_point = make_double3 (0.5, 0.5, 0.5);
	printf("Test: probe point is distant %g from the bottom plane.\n,
		dGeomPlanePointDepth(m_box_planes[0], probe_point.x, probe_point.y, probe_point.z)
	*/

	return h5File.getNParts();
}

void CompleteSaExample::copy_to_array(BufferList &buffers)
{
	float4 *pos = buffers.getData<BUFFER_POS>();
	hashKey *hash = buffers.getData<BUFFER_HASH>();
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();
	vertexinfo *vertices = buffers.getData<BUFFER_VERTICES>();
	float4 *boundelm = buffers.getData<BUFFER_BOUNDELEMENTS>();
	float4 *eulerVel = buffers.getData<BUFFER_EULERVEL>();

	h5File.read();

	uint n_parts = 0;
	uint n_vparts = 0;
	uint n_bparts = 0;

	for (uint i = 0; i<h5File.getNParts(); i++) {
		switch(h5File.buf[i].ParticleType) {
			case CRIXUS_FLUID:
				n_parts++;
				break;
			case CRIXUS_VERTEX:
				n_vparts++;
				break;
			case CRIXUS_BOUNDARY:
				n_bparts++;
				break;
		}
	}

	std::cout << "Fluid parts: " << n_parts << "\n";
	for (uint i = 0; i < n_parts; i++) {
		float rho = density(initial_water_level - h5File.buf[i].Coords_2, 0);
		//float rho = m_physparams.rho0[0];
		vel[i] = make_float4(0, 0, 0, rho);
		if (eulerVel)
			eulerVel[i] = make_float4(0);
		info[i] = make_particleinfo(FLUIDPART, 0, i);
		calc_localpos_and_hash(Point(h5File.buf[i].Coords_0, h5File.buf[i].Coords_1, h5File.buf[i].Coords_2,
			m_physparams.rho0[0]*h5File.buf[i].Volume), info[i], pos[i], hash[i]);
	}
	uint j = n_parts;
	std::cout << "Fluid part mass: " << pos[j-1].w << "\n";

	if(n_vparts) {
		std::cout << "Vertex parts: " << n_vparts << "\n";
		for (uint i = j; i < j + n_vparts; i++) {
			float rho = density(initial_water_level - h5File.buf[i].Coords_2, 0);
			vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
			if (eulerVel)
				eulerVel[i] = make_float4(0);
			int specialBoundType = h5File.buf[i].KENT;
			// count the number of different objects
			// note that we assume all objects to be sorted from 1 to n. Not really a problem if this
			// is not true it simply means that the IOwaterdepth object is bigger than it needs to be
			// in cases of ODE objects this array is allocated as well, even though it is not needed.
			m_simparams.numObjects = max(specialBoundType, m_simparams.numObjects);
			info[i] = make_particleinfo(VERTEXPART, specialBoundType, i);
			// Define the type of boundaries
			if (specialBoundType == 1) {
				// this vertex is part of an open boundary
				SET_FLAG(info[i], IO_PARTICLE_FLAG);
				// if you need to impose the velocity uncomment the following
				//// open boundary imposes velocity
				//SET_FLAG(info[i], VEL_IO_PARTICLE_FLAG);
			} else if (specialBoundType == 2) {
				// this vertex is part of a moving object
				SET_FLAG(info[i], MOVING_PARTICLE_FLAG);
				// this moving object is also floating
				SET_FLAG(info[i], FLOATING_PARTICLE_FLAG);
			}
			calc_localpos_and_hash(Point(h5File.buf[i].Coords_0, h5File.buf[i].Coords_1, h5File.buf[i].Coords_2,
				m_physparams.rho0[0]*h5File.buf[i].Volume), info[i], pos[i], hash[i]);
		}
		j += n_vparts;
		std::cout << "Vertex part mass: " << pos[j-1].w << "\n";
	}

	// NOTE: this actually counts only boundary parts
	uint numOdeObjParts = 0;

	if(n_bparts) {
		std::cout << "Boundary parts: " << n_bparts << "\n";
		for (uint i = j; i < j + n_bparts; i++) {
			vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
			if (eulerVel)
				eulerVel[i] = make_float4(0);
			int specialBoundType = h5File.buf[i].KENT;
			info[i] = make_particleinfo(BOUNDPART, specialBoundType, i);
			// Define the type of boundaries
			if (specialBoundType == 1) {
				// this vertex is part of an open boundary
				SET_FLAG(info[i], IO_PARTICLE_FLAG);
				// if you need to impose the velocity uncomment the following
				//// open boundary imposes velocity
				//SET_FLAG(info[i], VEL_IO_PARTICLE_FLAG);
			} else if (specialBoundType == 2) {
				// this vertex is part of a moving object
				SET_FLAG(info[i], MOVING_PARTICLE_FLAG);
				// this moving object is also floating
				SET_FLAG(info[i], FLOATING_PARTICLE_FLAG);
				if (numOdeObjParts == 0)
					// first part of floating body, write it as offset for rbforces
					gdata->s_hRbFirstIndex[2-1] = - i;
				numOdeObjParts++;
			}
			calc_localpos_and_hash(Point(h5File.buf[i].Coords_0, h5File.buf[i].Coords_1, h5File.buf[i].Coords_2, 0.0), info[i], pos[i], hash[i]);
			vertices[i].x = h5File.buf[i].VertexParticle1;
			vertices[i].y = h5File.buf[i].VertexParticle2;
			vertices[i].z = h5File.buf[i].VertexParticle3;
			boundelm[i].x = h5File.buf[i].Normal_0;
			boundelm[i].y = h5File.buf[i].Normal_1;
			boundelm[i].z = h5File.buf[i].Normal_2;
			boundelm[i].w = h5File.buf[i].Surface;
		}
		j += n_bparts;
		std::cout << "Boundary part mass: " << pos[j-1].w << "\n";
	}
	// Make sure that fluid + vertex + boundaries are done in that order
	// before adding any other items like testpoints, etc.
	cube->SetNumParts(numOdeObjParts);
	// set last index for rbforces
	gdata->s_hRbLastIndex[0] = numOdeObjParts - 1;

	//Testpoints
	if (test_points.size()) {
		std::cout << "\nTest points: " << test_points.size() << "\n";
		for (uint i = j; i < j+test_points.size(); i++) {
			vel[i] = make_float4(0, 0, 0, 0.0);
			info[i]= make_particleinfo(TESTPOINTSPART, 0, i);
			calc_localpos_and_hash(test_points[i-j], info[i], pos[i], hash[i]);
		}
		j += test_points.size();
		std::cout << "Test point mass:" << pos[j-1].w << "\n";
	}

	std::flush(std::cout);

	h5File.empty();
}

void
CompleteSaExample::init_keps(float* k, float* e, uint numpart, particleinfo* info, float4* pos, hashKey* hash)
{
	const float k0 = 1.0f/sqrtf(0.09f);

	for (uint i = 0; i < numpart; i++) {
		k[i] = k0;
		e[i] = 2.874944542f*k0*0.01f;
	}
}

uint
CompleteSaExample::max_parts(uint numpart)
{
	return (uint)((float)numpart*2.0f);
}

void CompleteSaExample::fillDeviceMap()
{
	fillDeviceMapByAxis(Y_AXIS);
}

void CompleteSaExample::imposeForcedMovingObjects(
			float3	&centerOfGravity,
			float3	&translation,
			float*	rotationMatrix,
	const	uint	ob,
	const	double	t,
	const	float	dt)
{
	switch (ob) {
		case 2:
			centerOfGravity = make_float3(0.0f, 0.0f, 0.0f);
			translation = make_float3(0.2f*dt, 0.0f, 0.0f);
			for (uint i=0; i<9; i++)
				rotationMatrix[i] = (i%4==0) ? 1.0f : 0.0f;
			break;
		default:
			break;
	}
}
