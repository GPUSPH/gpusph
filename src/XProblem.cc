#include <math.h>
#include <string>
#include <iostream>

#include "Rect.h"
#include "Disk.h"
#include "Cube.h"
#include "Cylinder.h"
#include "Cone.h"
#include "Sphere.h"
#include "Torus.h"

#include "XProblem.h"
#include "GlobalData.h"

//#define USE_PLANES 0

typedef std::vector<int>::size_type vsize_t;

XProblem::XProblem(const GlobalData *_gdata) : Problem(_gdata)
{
	/*h5File.setFilename("meshes/0.complete_sa_example.h5sph");

	container = STLMesh::load_stl("./meshes/CompleteSaExample_container_coarse.stl");
	cube = STLMesh::load_stl("./meshes/CompleteSaExample_cube_coarse.stl");

	m_simparams.numObjects = 2;

	m_simparams.sfactor=1.3f;
	set_deltap(0.02f);

	m_physparams.kinematicvisc = 1.0e-2f;
	m_simparams.visctype = DYNAMICVISC;
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
	m_simparams.inoutBoundaries = true;
	m_simparams.movingBoundaries = true;
	//m_simparams.floatingObjects = true;

	// SPH parameters
	m_simparams.dt = 0.00004f;
	m_simparams.xsph = false;
	m_simparams.dtadapt = true;
	m_simparams.dtadaptfactor = 0.3;
	m_simparams.buildneibsfreq = 1;
	m_simparams.shepardfreq = 0;
	m_simparams.mlsfreq = 0;
	m_simparams.ferrari = 0.1;
	m_simparams.mbcallback = false;
	m_simparams.boundarytype = SA_BOUNDARY;
	m_simparams.nlexpansionfactor = 1.1;

	// Size and origin of the simulation domain
	m_size = make_double3(world_l, world_w ,world_h);

	// Physical parameters
	float g = length(m_physparams.gravity);

	m_physparams.dcoeff = 5.0f*g*expected_final_water_level;

	m_physparams.r0 = m_deltap;

	m_physparams.artvisccoeff = 0.3f;
	m_physparams.epsartvisc = 0.01*m_simparams.slength*m_simparams.slength;
	m_physparams.epsxsph = 0.5f;

	// Drawing and saving times

	// will use only 1 ODE body: the floating/moving cube (container is fixed)
	allocate_ODE_bodies(1);
	dInitODE();
	// world setup
	m_ODEWorld = dWorldCreate(); // ODE world for dynamics
	m_ODESpace = dHashSpaceCreate(0); // ODE world for collisions
	m_ODEJointGroup = dJointGroupCreate(0);  // Joint group for collision detection
	// Set gravity（x, y, z)
	dWorldSetGravity(m_ODEWorld,
		m_physparams.gravity.x, m_physparams.gravity.y, m_physparams.gravity.z); */

	//m_simparams.dt = 0.00004f;
	//m_simparams.xsph = false;
	//m_simparams.dtadapt = true;
	//m_simparams.dtadaptfactor = 0.3;
	//m_simparams.buildneibsfreq = 1;
	//m_simparams.shepardfreq = 0;
	//m_simparams.mlsfreq = 0;
	//m_simparams.ferrari = 0.1;
	//m_simparams.mbcallback = false;
	//m_simparams.boundarytype = SA_BOUNDARY;
	//m_simparams.nlexpansionfactor = 1.1;
	//m_physparams.artvisccoeff = 0.3f;
	//m_simparams.sfactor=1.3f;

	//m_physparams.set_density(0, 1000.0, 7.0f, 20.f);
	//set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5


	set_deltap(0.05f);
	m_physparams.r0 = m_deltap;
	m_physparams.r0 = m_deltap;
	m_physparams.gravity = make_float3(0.0, 0.0, -9.81);
	float g = length(m_physparams.gravity);
	double H = 1;
	m_physparams.dcoeff = 5.0f*g*H;
	m_physparams.set_density(0, 1000.0, 7.0f, 20.0f);
	m_simparams.dtadapt = true;
	//m_physparams.kinematicvisc = 1.0e-2f;
	m_simparams.visctype = ARTVISC;
	add_writer(VTKWRITER, 1e-2f);
	m_origin = make_double3(0,0,0);
	m_size = make_double3(2,2,2);
	m_simparams.maxneibsnum = 128;

	// Name of problem used for directory creation
	m_name = "XProblem";

	double side = 1;
	double oside = 0.2;
	double wside = 1 - 2 * m_deltap;
	double wlevel = 0.3 - m_deltap;
	double orig = 0;

	// container
	addCube(GT_FIXED_BOUNDARY, FT_BORDER, Point(orig, orig, orig), side);

	// water
	orig = m_deltap;
	GeometryID water = addBox(GT_FLUID, FT_SOLID, Point(orig, orig, orig), wside, wside, wlevel);

	// cube
	orig = side/2 - oside/2;
	GeometryID cube = addCube(GT_FLOATING_BODY, FT_SOLID, Point(orig, orig, orig), oside);
	rotateGeometry(cube, EulerParameters(0, M_PI/4, 0));
	//rotateGeometry(cube, M_PI/4, 0, 0);

	/*
	double minor_radius = side / 8;
	double major_radius = radius - minor_radius;
	addTorus(GT_FLUID, FT_SOLID, Point(orig, orig, orig), major_radius, minor_radius);
	*/
}

void XProblem::release_memory()
{
	m_fluidParts.clear();
	m_boundaryParts.clear();
}

XProblem::~XProblem()
{
	release_memory();
	if (m_numRigidBodies)
		cleanupODE();
}

void XProblem::initialize()
{
	// only init ODE if m_numRigidBodies
	if (m_numRigidBodies > 0)
		initializeODE();
}

void XProblem::initializeODE()
{
	allocate_ODE_bodies(m_numRigidBodies);
	dInitODE();
	// world setup
	m_ODEWorld = dWorldCreate(); // ODE world for dynamics
	m_ODESpace = dHashSpaceCreate(0); // ODE world for collisions
	m_ODEJointGroup = dJointGroupCreate(0);  // Joint group for collision detection
	// Set gravity（x, y, z)
	dWorldSetGravity(m_ODEWorld,
		m_physparams.gravity.x, m_physparams.gravity.y, m_physparams.gravity.z);
}

void XProblem::cleanupODE()
{
	dJointGroupDestroy(m_ODEJointGroup);
	dSpaceDestroy(m_ODESpace);
	dWorldDestroy(m_ODEWorld);
	dCloseODE();
}

GeometryID XProblem::addGeometry(const GeometryType otype, const FillType ftype, Object* obj_ptr)
{
	GeometryInfo* geomInfo = new GeometryInfo();
	geomInfo->type = otype;
	geomInfo->fill_type = ftype;
	geomInfo->ptr = obj_ptr;
	m_numGeometries++;
	if (geomInfo->type == GT_FLOATING_BODY)
		m_numRigidBodies++;
	m_geometries.push_back(geomInfo);
	return (m_geometries.size() - 1);
}

GeometryID XProblem::addRect(const GeometryType otype, const FillType ftype, const Point &origin,
	const double side1, const double side2)
{
	return addGeometry(otype, ftype,
		new Rect( origin, Vector(side1, 0, 0), Vector(0, side2, 0) )
	);
}

GeometryID XProblem::addDisk(const GeometryType otype, const FillType ftype, const Point &origin,
	const double radius)
{
	return addGeometry(otype, ftype,
		new Disk( origin, radius, Vector(0, 0, 1) )
	);
}

GeometryID XProblem::addCube(const GeometryType otype, const FillType ftype, const Point &origin, const double side)
{
	return addGeometry(otype, ftype,
		new Cube( origin, Vector(side, 0, 0), Vector(0, side, 0), Vector(0, 0, side) )
	);
}

GeometryID XProblem::addBox(const GeometryType otype, const FillType ftype, const Point &origin,
			const double side1, const double side2, const double side3)
{
	return addGeometry(otype, ftype,
		new Cube( origin, Vector(side1, 0, 0), Vector(0, side2, 0), Vector(0, 0, side3) )
	);
}

GeometryID XProblem::addCylinder(const GeometryType otype, const FillType ftype, const Point &origin,
			const double radius, const double height)
{
	return addGeometry(otype, ftype,
		new Cylinder( origin, Vector(radius, 0, 0), Vector(0, 0, height) )
	);
}

GeometryID XProblem::addCone(const GeometryType otype, const FillType ftype, const Point &origin,
	const double bottom_radius, const double top_radius, const double height)
{
	return addGeometry(otype, ftype,
		new Cone( origin, bottom_radius, top_radius, Vector(0, 0, height) )
	);
}

GeometryID XProblem::addSphere(const GeometryType otype, const FillType ftype, const Point &origin,
	const double radius)
{
	return addGeometry(otype, ftype,
		new Sphere( origin, radius )
	);
}

GeometryID XProblem::addTorus(const GeometryType otype, const FillType ftype, const Point &origin,
	const double major_radius, const double minor_radius)
{
	return addGeometry(otype, ftype,
		new Torus( origin, Vector(0, 0, 1), major_radius, minor_radius )
	);
}

void XProblem::deleteGeometry(const GeometryID gid)
{
	m_geometries[gid]->enabled = false;

	// and this is the reason why m_numGeometries not be used to iterate on m_geometries:
	m_numGeometries--;

	if (m_geometries[gid]->ptr->m_ODEBody || m_geometries[gid]->ptr->m_ODEGeom)
		m_numRigidBodies--;

	// TODO: remove from other arrays/counters? (e.g. ODE objs)
	// TODO: print a warning if deletion is requested after fill_parts
}

void XProblem::rotateGeometry(const GeometryID gid, const EulerParameters &ep)
{
	m_geometries[gid]->ptr->setEulerParameters(ep);
}

void XProblem::rotateGeometry(const GeometryID gid, const dQuaternion quat)
{
	m_geometries[gid]->ptr->setEulerParameters( EulerParameters(quat) );
}

void XProblem::rotateGeometry(const GeometryID gid, const double Xrot, const double Yrot, const double Zrot)
{
	double psi, theta, phi;
	// see http://goo.gl/4LQU9w
	theta = Yrot;
	double R11 = cos(Yrot)*cos(Zrot);
	double R21 = cos(Yrot)* sin(Zrot);
	double R32 = sin(Xrot)* cos(Yrot);
	double R33 = cos(Xrot)* cos(Yrot);
	if (cos(theta) == 0.0F) {
		// gimbal lock
		double R12 = sin(Xrot)*sin(Yrot)*cos(Zrot) - cos(Xrot)*sin(Zrot);
		double R13 = cos(Xrot)*sin(Yrot)*cos(Zrot) + sin(Xrot)*sin(Zrot);
		phi = 0;
		psi = atan2(R12, R13);
	} else {
		psi = atan2( R32/cos(theta), R33/cos(theta) );
		phi = atan2( R21/cos(theta), R11/cos(theta) );
		printf(" T %g, PSI %g, FI %g\n", theta, psi, phi);
	}
	rotateGeometry( gid, EulerParameters(psi, theta, phi) );
}

void XProblem::setMass(const GeometryID gid, const double mass)
{
	m_geometries[gid]->ptr->SetMass(mass);
}

void XProblem::setMassByDensity(const GeometryID gid, const double density)
{
	m_geometries[gid]->ptr->SetMass(m_physparams.r0, density);
}

const GeometryInfo* XProblem::getGeometryInfo(GeometryID gid)
{
	return m_geometries[gid];
}

int XProblem::fill_parts()
{
	/* If we needed an accurate collision detection between the container and the cube (i.e.
	 * using the actual container mesh instead of modelling it with 5 infinite planes), we'd
	 * have to create a geometry for the container as well. However, in this case the mesh
	 * should be "thick" (2 layers, with volume inside) and not flat. With a flat mesh, ODE
	 * detects the contained cube as penetrated into the container and thus colliding.
	 * Long story short: don't uncomment the following.
	 */ /*
	//container->ODEGeomCreate(m_ODESpace, m_deltap);

	// cube density half water density
	const double water_density_fraction = 0.5F;
	const double cube_density = 1000.0 * water_density_fraction; // 1000 = water
	//cube->ODEBodyCreate(m_ODEWorld, m_deltap, cube_density); // only dynamics
	//cube->ODEGeomCreate(m_ODESpace, m_deltap); // only collisions
	cube->ODEBodyCreate(m_ODEWorld, m_deltap, cube_density, m_ODESpace); // dynamics + collisions
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

	//return h5File.getNParts();

	//uint particleCounter = 0;
	uint bodies_parts_counter = 0;

	for (vsize_t i = 0; i < m_geometries.size(); i++) {
		PointVect* parts_vector = NULL;
		double dx = 0.0;

		// ignore deleted geometries
		if (!m_geometries[i]->enabled) continue;

		// set dx and recipient vector according to geometry type
		if (m_geometries[i]->type == GT_FLUID) {
			parts_vector = &m_fluidParts;
			dx = m_deltap;
		} else
		if (m_geometries[i]->type == GT_FLOATING_BODY) {
			parts_vector = &(m_geometries[i]->ptr->GetParts());
			dx = m_physparams.r0;
		} else {
			parts_vector = &m_boundaryParts;
			dx = m_physparams.r0;
		}

		m_geometries[i]->ptr->SetPartMass(dx, m_physparams.rho0[0]);

		if (m_geometries[i]->fill_type == FT_BORDER)
			m_geometries[i]->ptr->FillBorder(*parts_vector, m_deltap);
		else {
			//const bool fillBorder = (m_geometries[i]->fill_type != FT_SOLID_BORDERLESS);
			if (m_geometries[i]->fill_type == FT_SOLID_BORDERLESS)
				printf("WARNING: borderless not yet implemented, filling with border\n");
			m_geometries[i]->ptr->Fill(*parts_vector, m_deltap);
		}

		// ODE stuff, anyone?
		if (m_geometries[i]->type == GT_FLOATING_BODY) {
			m_geometries[i]->ptr->ODEBodyCreate(m_ODEWorld, m_deltap);
			m_geometries[i]->ptr->ODEGeomCreate(m_ODESpace, m_deltap);
			add_ODE_body(m_geometries[i]->ptr);
			bodies_parts_counter += m_geometries[i]->ptr->GetParts().size();
		}
	}

	return m_fluidParts.size() + m_boundaryParts.size() + bodies_parts_counter;
}

void XProblem::copy_to_array(BufferList &buffers)
{
	float4 *pos = buffers.getData<BUFFER_POS>();
	hashKey *hash = buffers.getData<BUFFER_HASH>();
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();
	vertexinfo *vertices = buffers.getData<BUFFER_VERTICES>();
	float4 *boundelm = buffers.getData<BUFFER_BOUNDELEMENTS>();
	//float4 *eulerVel = buffers.getData<BUFFER_EULERVEL>();

	/// put this in initialization instead!
	//h5File.read();

	uint n_fparts = 0;
	uint n_vparts = 0;
	uint n_bparts = 0;
	uint elaborated_parts = 0;

	n_fparts = m_fluidParts.size();
	n_bparts = m_boundaryParts.size();

	// count types
	/*
	for (uint i = 0; i<h5File.getNParts(); i++) {
		switch(h5File.buf[i].ParticleType) {
			case 1:
				n_fparts++;
				break;
			case 2:
				n_vparts++;
				break;
			case 3:
				n_bparts++;
				break;
		}
	}
	*/

	std::cout << "Fluid parts: " << n_fparts << "\n";
	for (uint i = 0; i < n_fparts; i++) {
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(FLUIDPART,0,i);
		calc_localpos_and_hash(m_fluidParts[i], info[i], pos[i], hash[i]);
	}
	elaborated_parts += n_fparts;
	std::cout << "Fluid part mass: " << pos[elaborated_parts - 1].w << "\n";
	std::flush(std::cout);

	std::cout << "Boundary parts: " << n_bparts << "\n";
	for (uint i = elaborated_parts; i < elaborated_parts + n_bparts; i++) {
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		//if (eulerVel)
		//	eulerVel[i] = make_float4(0);
		//int specialBoundType = h5File.buf[i].KENT;
		//info[i] = make_particleinfo(BOUNDPART, specialBoundType, i);
		info[i] = make_particleinfo(BOUNDPART, 0, i);
		calc_localpos_and_hash(m_boundaryParts[i - elaborated_parts], info[i], pos[i], hash[i]);
		// Save the id of the first boundary particle that belongs to an ODE object
		/*if (m_firstODEobjectPartId == 0 && specialBoundType != 0 &&  m_ODEobjectId[specialBoundType-1] != UINT_MAX)
			m_firstODEobjectPartId = i;
		// Define the type of boundaries
		if (specialBoundType == 1) {
			// this vertex is part of an open boundary
			SET_FLAG(info[i], IO_PARTICLE_FLAG);
			// if you need to impose the velocity uncomment the following
			//// open boundary imposes velocity
			//SET_FLAG(info[i], VEL_IO_PARTICLE_FLAG);
			// open boundary is an inflow
			SET_FLAG(info[i], INFLOW_PARTICLE_FLAG);
		} else if (specialBoundType == 2) {
			// this vertex is part of a moving object
			SET_FLAG(info[i], MOVING_PARTICLE_FLAG);
			// this moving object is also floating
			//SET_FLAG(info[i], FLOATING_PARTICLE_FLAG);
			//numOdeObjParts++;
		}
		calc_localpos_and_hash(Point(h5File.buf[i].Coords_0, h5File.buf[i].Coords_1, h5File.buf[i].Coords_2, 0.0), info[i], pos[i], hash[i]);
		vertices[i].x = h5File.buf[i].VertexParticle1;
		vertices[i].y = h5File.buf[i].VertexParticle2;
		vertices[i].z = h5File.buf[i].VertexParticle3;
		boundelm[i].x = h5File.buf[i].Normal_0;
		boundelm[i].y = h5File.buf[i].Normal_1;
		boundelm[i].z = h5File.buf[i].Normal_2;
		boundelm[i].w = h5File.buf[i].Surface;*/
	}
	elaborated_parts += n_bparts;
	std::cout << "Boundary part mass: " << pos[elaborated_parts - 1].w << "\n";


	for (uint k = 0; k < m_simparams.numODEbodies; k++) {
		PointVect & rbparts = get_ODE_body(k)->GetParts();
		std::cout << "Rigid body " << k << ": " << rbparts.size() << " particles ";
		for (uint i = elaborated_parts; i < elaborated_parts + rbparts.size(); i++) {
			vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
			info[i] = make_particleinfo(OBJECTPART, k, i - elaborated_parts);
			calc_localpos_and_hash(rbparts[i - elaborated_parts], info[i], pos[i], hash[i]);
		}
		elaborated_parts += rbparts.size();
		std::cout << ", part mass: " << pos[elaborated_parts-1].w << "\n";
	}

	/*std::cout << "Fluid parts: " << n_fparts << "\n";
	for (uint i = 0; i < n_fparts; i++) {
		float rho = density(initial_water_level - h5File.buf[i].Coords_2, 0);
		//float rho = m_physparams.rho0[0];
		vel[i] = make_float4(0, 0, 0, rho);
		if (eulerVel)
			eulerVel[i] = make_float4(0);
		info[i] = make_particleinfo(FLUIDPART, 0, i);
		calc_localpos_and_hash(Point(h5File.buf[i].Coords_0, h5File.buf[i].Coords_1, h5File.buf[i].Coords_2,
			m_physparams.rho0[0]*h5File.buf[i].Volume), info[i], pos[i], hash[i]);
	}
	uint j = n_fparts;
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
				// open boundary is an inflow
				SET_FLAG(info[i], INFLOW_PARTICLE_FLAG);
			} else if (specialBoundType == 2) {
				// this vertex is part of a moving object
				SET_FLAG(info[i], MOVING_PARTICLE_FLAG);
				// this moving object is also floating
				//SET_FLAG(info[i], FLOATING_PARTICLE_FLAG);
			}
			calc_localpos_and_hash(Point(h5File.buf[i].Coords_0, h5File.buf[i].Coords_1, h5File.buf[i].Coords_2,
				m_physparams.rho0[0]*h5File.buf[i].Volume), info[i], pos[i], hash[i]);
		}
		j += n_vparts;
		std::cout << "Vertex part mass: " << pos[j-1].w << "\n";
	}

	uint numOdeObjParts = 0;

	if(n_bparts) {
		std::cout << "Boundary parts: " << n_bparts << "\n";
		for (uint i = j; i < j + n_bparts; i++) {
			vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
			if (eulerVel)
				eulerVel[i] = make_float4(0);
			int specialBoundType = h5File.buf[i].KENT;
			info[i] = make_particleinfo(BOUNDPART, specialBoundType, i);
			// Save the id of the first boundary particle that belongs to an ODE object
			if (m_firstODEobjectPartId == 0 && specialBoundType != 0 &&  m_ODEobjectId[specialBoundType-1] != UINT_MAX)
				m_firstODEobjectPartId = i;
			// Define the type of boundaries
			if (specialBoundType == 1) {
				// this vertex is part of an open boundary
				SET_FLAG(info[i], IO_PARTICLE_FLAG);
				// if you need to impose the velocity uncomment the following
				//// open boundary imposes velocity
				//SET_FLAG(info[i], VEL_IO_PARTICLE_FLAG);
				// open boundary is an inflow
				SET_FLAG(info[i], INFLOW_PARTICLE_FLAG);
			} else if (specialBoundType == 2) {
				// this vertex is part of a moving object
				SET_FLAG(info[i], MOVING_PARTICLE_FLAG);
				// this moving object is also floating
				//SET_FLAG(info[i], FLOATING_PARTICLE_FLAG);
				//numOdeObjParts++;
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

	std::flush(std::cout); */

	// h5File.empty();
}

/*
// for an exaple ODE_nearCallback see
// http://ode-wiki.org/wiki/index.php?title=Manual:_Collision_Detection#Collision_detection
void XProblem::ODE_near_callback(void * data, dGeomID o1, dGeomID o2)
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
		contact[i].surface.mu   = dInfinity;
		contact[i].surface.bounce     = 0.0; // (0.0~1.0) restitution parameter
		contact[i].surface.bounce_vel = 0.0; // minimum incoming velocity for bounce
		dJointID c = dJointCreateContact(m_ODEWorld, m_ODEJointGroup, &contact[i]);
		dJointAttach (c, dGeomGetBody(contact[i].geom.g1), dGeomGetBody(contact[i].geom.g2));
	}
} */

/*
void XProblem::copy_to_array(BufferList &buffers)
{
	float4 *pos = buffers.getData<BUFFER_POS>();
	hashKey *hash = buffers.getData<BUFFER_HASH>();
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();
	vertexinfo *vertices = buffers.getData<BUFFER_VERTICES>();
	float4 *boundelm = buffers.getData<BUFFER_BOUNDELEMENTS>();
	float4 *eulerVel = buffers.getData<BUFFER_EULERVEL>();

	h5File.read();

	uint n_fparts = 0;
	uint n_vparts = 0;
	uint n_bparts = 0;

	for (uint i = 0; i<h5File.getNParts(); i++) {
		switch(h5File.buf[i].ParticleType) {
			case 1:
				n_fparts++;
				break;
			case 2:
				n_vparts++;
				break;
			case 3:
				n_bparts++;
				break;
		}
	}

	std::cout << "Fluid parts: " << n_fparts << "\n";
	for (uint i = 0; i < n_fparts; i++) {
		float rho = density(initial_water_level - h5File.buf[i].Coords_2, 0);
		//float rho = m_physparams.rho0[0];
		vel[i] = make_float4(0, 0, 0, rho);
		if (eulerVel)
			eulerVel[i] = make_float4(0);
		info[i] = make_particleinfo(FLUIDPART, 0, i);
		calc_localpos_and_hash(Point(h5File.buf[i].Coords_0, h5File.buf[i].Coords_1, h5File.buf[i].Coords_2,
			m_physparams.rho0[0]*h5File.buf[i].Volume), info[i], pos[i], hash[i]);
	}
	uint j = n_fparts;
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
				// open boundary is an inflow
				SET_FLAG(info[i], INFLOW_PARTICLE_FLAG);
			} else if (specialBoundType == 2) {
				// this vertex is part of a moving object
				SET_FLAG(info[i], MOVING_PARTICLE_FLAG);
				// this moving object is also floating
				//SET_FLAG(info[i], FLOATING_PARTICLE_FLAG);
			}
			calc_localpos_and_hash(Point(h5File.buf[i].Coords_0, h5File.buf[i].Coords_1, h5File.buf[i].Coords_2,
				m_physparams.rho0[0]*h5File.buf[i].Volume), info[i], pos[i], hash[i]);
		}
		j += n_vparts;
		std::cout << "Vertex part mass: " << pos[j-1].w << "\n";
	}

	uint numOdeObjParts = 0;

	if(n_bparts) {
		std::cout << "Boundary parts: " << n_bparts << "\n";
		for (uint i = j; i < j + n_bparts; i++) {
			vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
			if (eulerVel)
				eulerVel[i] = make_float4(0);
			int specialBoundType = h5File.buf[i].KENT;
			info[i] = make_particleinfo(BOUNDPART, specialBoundType, i);
			// Save the id of the first boundary particle that belongs to an ODE object
			if (m_firstODEobjectPartId == 0 && specialBoundType != 0 &&  m_ODEobjectId[specialBoundType-1] != UINT_MAX)
				m_firstODEobjectPartId = i;
			// Define the type of boundaries
			if (specialBoundType == 1) {
				// this vertex is part of an open boundary
				SET_FLAG(info[i], IO_PARTICLE_FLAG);
				// if you need to impose the velocity uncomment the following
				//// open boundary imposes velocity
				//SET_FLAG(info[i], VEL_IO_PARTICLE_FLAG);
				// open boundary is an inflow
				SET_FLAG(info[i], INFLOW_PARTICLE_FLAG);
			} else if (specialBoundType == 2) {
				// this vertex is part of a moving object
				SET_FLAG(info[i], MOVING_PARTICLE_FLAG);
				// this moving object is also floating
				//SET_FLAG(info[i], FLOATING_PARTICLE_FLAG);
				//numOdeObjParts++;
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
XProblem::init_keps(float* k, float* e, uint numpart, particleinfo* info, float4* pos, hashKey* hash)
{
	const float k0 = 1.0f/sqrtf(0.09f);

	for (uint i = 0; i < numpart; i++) {
		k[i] = k0;
		e[i] = 2.874944542f*k0*0.01f;
	}
}

uint
XProblem::max_parts(uint numpart)
{
	return (uint)((float)numpart*2.0f);
}

void XProblem::fillDeviceMap()
{
	fillDeviceMapByAxis(Y_AXIS);
}

void XProblem::imposeForcedMovingObjects(
			float3*	gravityCenters,
			float3*	translations,
			float*	rotationMatrices,
	const	uint*	ODEobjectId,
	const	uint	numObjects,
	const	double	t,
	const	float	dt)
{
	// for object(info)==n we need to access array index n-1
	uint id = 2-1;
	// if ODEobjectId[id] is not equal to UINT_MAX we have a floating object
	if (ODEobjectId[id] == UINT_MAX) {
		gravityCenters[id] = make_float3(0.0f, 0.0f, 0.0f);
		translations[id] = make_float3(0.2f*dt, 0.0f, 0.0f);
		for (uint i=0; i<9; i++)
			rotationMatrices[id*9+i] = (i==0 || i==4 || i==8) ? 1.0f : 0.0f;
	}
}
*/
