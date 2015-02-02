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
#include "Plane.h"
#include "STLMesh.h"

#include "XProblem.h"
#include "GlobalData.h"

//#define USE_PLANES 0

XProblem::XProblem(const GlobalData *_gdata) : Problem(_gdata)
{
	// *** XProblem initialization
	m_numActiveGeometries = 0;
	m_numRigidBodies = 0;
	m_numPlanes = 0;

	m_extra_world_margin = 0.0;

	m_positioning = PP_CENTER;

	// *** Optional simulation parameters (defaults?)

	//m_simparams.tend = 10.0;
	//m_simparams.testpoints = false;
	//m_simparams.surfaceparticle = false;
	//m_simparams.savenormals = false;
	//m_simparams.calcPrivate = false;
	// TODO: check: do we need the following 4 as well?
	//m_simparams.inoutBoundaries = true;
	//m_simparams.movingBoundaries = true;
	//m_simparams.floatingObjects = true;
	//m_simparams.numObjects = 2;

	// NOTE: no need to set m_simparams.numODEbodies, since it will be set by allocate_ODE_bodies()

	// *** Optional SPH parameters

	//m_simparams.dt = 0.00004f;
	//m_simparams.xsph = false;
	//m_simparams.dtadapt = true; // default?
	//m_simparams.dtadaptfactor = 0.3;
	//m_simparams.buildneibsfreq = 1;
	//m_simparams.shepardfreq = 0;
	//m_simparams.mlsfreq = 0;
	//m_simparams.ferrari = 0.1;
	//m_simparams.mbcallback = false;
	//m_simparams.nlexpansionfactor = 1.1;
	//m_simparams.sfactor=1.3f;

	// *** Optional physical parameters
	//m_physparams.artvisccoeff = 0.3f;
	//m_physparams.epsartvisc = 0.01*m_simparams.slength*m_simparams.slength;
	//m_physparams.epsxsph = 0.5f;
	// TODO: following comment: still makes sense?
	//set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5

	// *** Initialization of minimal physical parameters
	set_deltap(0.02f);
	m_physparams.r0 = m_deltap;
	m_physparams.gravity = make_float3(0.0, 0.0, -9.81);
	float g = length(m_physparams.gravity);
	double H = 3;
	m_physparams.dcoeff = 5.0f*g*H;
	m_physparams.set_density(0, 1000.0, 7.0f, 20.0f);
	//m_physparams.kinematicvisc = 1.0e-2f;

	// *** Initialization of minimal simulation parameters
	m_simparams.maxneibsnum = 256 + 64;
	m_simparams.dtadapt = true;
	// viscositys: ARTVISC, KINEMATICVISC, DYNAMICVISC, SPSVISC, KEPSVISC
	m_simparams.visctype = ARTVISC;
	// boundary types: LJ_BOUNDARY, MK_BOUNDARY, SA_BOUNDARY, DYN_BOUNDARY
	m_simparams.boundarytype = SA_BOUNDARY;
	// also init formulation (SPH_F1, SPH_F2) & kernel type (WENDLAND, CUBICSPLINE)?

	// *** Other parameters and settings
	add_writer(VTKWRITER, 1e-2f);
	m_origin = make_double3(NAN, NAN, NAN);
	m_size = make_double3(NAN, NAN, NAN);
	m_name = "XProblem";

	// example usage

	/*

	addHDF5File(GT_FLUID, Point(0,0,0), "./sa/0.complete_sa_example.fluid.h5sph", NULL);
	// main container
	GeometryID container =
		addHDF5File(GT_FIXED_BOUNDARY, Point(0,0,0), "./sa/0.complete_sa_example.boundary.kent0.h5sph", NULL);
	disableCollisions(container);

	// inflow square
	GeometryID inlet =
		addHDF5File(GT_FIXED_BOUNDARY, Point(0,0,0), "./sa/0.complete_sa_example.boundary.kent1.h5sph", NULL);
	disableCollisions(inlet);

	// floating box
	GeometryID cube =
		addHDF5File(GT_FLOATING_BODY, Point(0,0,0), "./sa/0.complete_sa_example.boundary.kent2.h5sph", "./sa/sa_box_sbgrid_2.stl");
	setMassByDensity(cube, m_physparams.rho0[0] / 2);

	// manually set world size and origin while HDF5 file loading does not support bbox detection yet
	m_origin = make_double3(-1, -1, -1);
	m_size = make_double3(3, 3, 3);

	// add "universe box" of planes. TODO: check: what happens with LJ planes in SA simulation?
	makeUniverseBox(m_origin, m_origin + m_size );
	*/
}

void XProblem::release_memory()
{
	m_fluidParts.clear();
	m_boundaryParts.clear();
	// also cleanup object parts
	for (size_t i = 0, num_geoms = m_geometries.size(); i < num_geoms; i++)
		if (m_geometries[i]->enabled)
			m_geometries[i]->ptr->GetParts().clear();
}

XProblem::~XProblem()
{
	release_memory();
	if (m_numRigidBodies > 0)
		cleanupODE();
}

void XProblem::initialize()
{
	// aux vars to compute bounding box
	Point globalMin = Point (NAN, NAN, NAN);
	Point globalMax = Point (NAN, NAN, NAN);

	// counters of floating objects and generic objects (floating + moving + open bounds)
	uint rigid_body_counter = 0;
	uint object_counter = 0;

	for (size_t g = 0, num_geoms = m_geometries.size(); g < num_geoms; g++) {
		// ignore planes for bbox
		if (m_geometries[g]->type == GT_PLANE)
			continue;

		// ignore deleted geometries
		if (!m_geometries[g]->enabled)
			continue;

		Point currMin, currMax;

		// get bbox of curr geometry
		m_geometries[g]->ptr->getBoundingBox(currMin, currMax);

		// global min and max
		setMinPerElement(globalMin, currMin);
		setMaxPerElement(globalMax, currMax);

		// update m_ODEobjectId map
		// (recall: from object index in particleinfo, incl. I/O, to floating object index, excl. I/O)
		if (m_geometries[g]->type == GT_FLOATING_BODY)
			m_ODEobjectId[ object_counter ] = rigid_body_counter;

		// update object counters
		if (m_geometries[g]->type == GT_FLOATING_BODY)
			rigid_body_counter++;
		if (m_geometries[g]->type == GT_FLOATING_BODY ||
			m_geometries[g]->type == GT_MOVING_BODY   ||
			m_geometries[g]->type == GT_OPENBOUNDARY )
			object_counter++;
	}

	// store number of floating objects (aka ODE bodies)
	// NOTE: allocate_ODE_bodies() sets it again, but we will get rid of that
	m_simparams.numODEbodies = rigid_body_counter;
	// store number of objects (floating + moving + I/O)
	m_simparams.numObjects = object_counter;

	// set computed world origin and size without overriding possible user choices
	if (!isfinite(m_origin.x)) m_origin.x = globalMin(0);
	if (!isfinite(m_origin.y)) m_origin.y = globalMin(1);
	if (!isfinite(m_origin.z)) m_origin.z = globalMin(2);
	if (!isfinite(m_size.x)) m_size.x = globalMax(0) - globalMin(0);
	if (!isfinite(m_size.y)) m_size.y = globalMax(1) - globalMin(1);
	if (!isfinite(m_size.z)) m_size.z = globalMax(2) - globalMin(2);

	// add user-defined world margin, if any
	if (m_extra_world_margin > 0.0) {
		m_origin -= m_extra_world_margin;
		m_size += 2 * m_extra_world_margin;
	}

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

	// consider collisions where at least one of the two bodies is a floating body...
	bool isOneFloating = false;
	for (uint gid = 0, num_geoms = m_geometries.size(); gid < num_geoms && !isOneFloating; gid++) {
		// ignore deleted geometries
		if (!m_geometries[gid]->enabled)
			continue;
		// is the current geometry a floating body?
		if (m_geometries[gid]->type == GT_FLOATING_BODY) {
			// read ODE geom ID
			const dGeomID curr_odegeom_id = m_geometries[gid]->ptr->m_ODEGeom;
			// check if this is one of the two colliding
			if (curr_odegeom_id == o1 || curr_odegeom_id == o2)
				isOneFloating = true;
		} // if current geom is floating
	} // iterating on all geometries

	// ...ignore otherwise
	if (!isOneFloating) return;

	// collide the candidate pair o1, o2
	int num_contacts = dCollide(o1, o2, MAX_CONTACTS, &contact[0].geom, skip_offset);

	// resulting collision points are treated by ODE as joints. We use them all
	for (int i = 0; i < num_contacts; i++) {
		contact[i].surface.mode = dContactBounce;
		contact[i].surface.mu = 1.0; // min 1, max dInfinity (min-max friction)
		contact[i].surface.bounce = 0.5; // (0.0~1.0) restitution parameter
		contact[i].surface.bounce_vel = 0.0; // minimum incoming velocity for bounce
		dJointID c = dJointCreateContact(m_ODEWorld, m_ODEJointGroup, &contact[i]);
		dJointAttach (c, dGeomGetBody(contact[i].geom.g1), dGeomGetBody(contact[i].geom.g2));
	}
}

GeometryID XProblem::addGeometry(const GeometryType otype, const FillType ftype, Object* obj_ptr,
	const char *hdf5_fname)
{
	GeometryInfo* geomInfo = new GeometryInfo();
	geomInfo->type = otype;
	geomInfo->fill_type = ftype;
	geomInfo->ptr = obj_ptr;
	if (hdf5_fname) {
		geomInfo->hdf5_filename = std::string(hdf5_fname);
		geomInfo->has_hdf5_file = true;
	}
	m_numActiveGeometries++;

	// --- Default collision and dynamics
	switch (geomInfo->type) {
		case GT_FLUID:
			geomInfo->handle_collisions = false;
			geomInfo->handle_dynamics = false;
			break;
		case GT_FIXED_BOUNDARY:
			geomInfo->handle_collisions = true; // optional
			geomInfo->handle_dynamics = false;
			break;
		case GT_FLOATING_BODY:
			geomInfo->handle_collisions = true; // optional
			geomInfo->handle_dynamics = true;
			break;
		case GT_MOVING_BODY:
			geomInfo->handle_collisions = true; // optional
			geomInfo->handle_dynamics = false; // optional
			break;
		case GT_PLANE:
			geomInfo->handle_collisions = true; // optional
			geomInfo->handle_dynamics = false;
			break;
	}

	// --- Default intersection type
	// It is IT_SUBTRACT by default, except for planes: they are usually used
	// to delimit the boundaries of the domain, so we likely want to intersect
	if (geomInfo->type == GT_PLANE)
		geomInfo->intersection_type = IT_INTERSECT;
	else
		geomInfo->intersection_type = IT_SUBTRACT;

	// --- Default erase operation
	// Upon intersection or subtraction we can choose to interact with fluid
	// or boundaries. By default, water erases only other water, while boundaries
	// erase water and other boundaries.
	if (geomInfo->type == GT_FLUID)
		geomInfo->erase_operation = ET_ERASE_FLUID;
	else
		geomInfo->erase_operation = ET_ERASE_ALL;

	// NOTE: we don't need to check handle_collisions at all, since if there are no bodies
	// we don't need collisions nor ODE at all
	if (geomInfo->handle_dynamics)
		m_numRigidBodies++;

	if (geomInfo->type == GT_PLANE)
		m_numPlanes++;

	m_geometries.push_back(geomInfo);
	return (m_geometries.size() - 1);
}

bool XProblem::validGeometry(GeometryID gid)
{
	// ensure gid refers to a valid position
	if (gid >= m_geometries.size()) {
		printf("WARNING: invalid GeometryID %u\n");
		return false;
	}

	// ensure geometry was not deleted
	if (!m_geometries[gid]->enabled) {
		printf("WARNING: GeometryID %u refers to a deleted geometry!\n");
		return false;
	}

	return true;
}

GeometryID XProblem::addRect(const GeometryType otype, const FillType ftype, const Point &origin,
	const double side1, const double side2)
{
	double offsetX = 0, offsetY = 0;
	if (m_positioning == PP_CENTER || m_positioning == PP_BOTTOM_CENTER) {
		offsetX = - side1 / 2.0;
		offsetY = - side2 / 2.0;
	}

	return addGeometry(otype, ftype,
		new Rect( Point( origin(0) + offsetX, origin(1) + offsetY, origin(2) ),
			side1, side2, EulerParameters() )
	);
}

GeometryID XProblem::addDisk(const GeometryType otype, const FillType ftype, const Point &origin,
	const double radius)
{
	double offsetX = 0, offsetY = 0;
	if (m_positioning == PP_CORNER)
		offsetX = offsetY = radius;

	return addGeometry(otype, ftype,
		new Disk( Point( origin(0) + offsetX, origin(1) + offsetY, origin(2) ),
			radius, EulerParameters() )
	);
}

GeometryID XProblem::addCube(const GeometryType otype, const FillType ftype, const Point &origin, const double side)
{
	double offsetXY = 0, offsetZ = 0;
	if (m_positioning == PP_CENTER || m_positioning == PP_BOTTOM_CENTER)
		offsetXY = - side / 2.0;
	if (m_positioning == PP_CENTER)
		offsetZ = - side / 2.0;

	return addGeometry(otype, ftype,
		new Cube( Point( origin(0) + offsetXY, origin(1) + offsetXY, origin(2) + offsetZ ),
			side, side, side, EulerParameters() )
	);
}

GeometryID XProblem::addBox(const GeometryType otype, const FillType ftype, const Point &origin,
			const double side1, const double side2, const double side3)
{
	double offsetX = 0, offsetY = 0, offsetZ = 0;
	if (m_positioning == PP_CENTER || m_positioning == PP_BOTTOM_CENTER) {
		offsetX = - side1 / 2.0;
		offsetY = - side2 / 2.0;
	}
	if (m_positioning == PP_CENTER)
		offsetZ = - side3 / 2.0;

	return addGeometry(otype, ftype,
		new Cube( Point( origin(0) + offsetX, origin(1) + offsetY, origin(2) + offsetZ ),
			side1, side2, side3, EulerParameters() )
	);
}

GeometryID XProblem::addCylinder(const GeometryType otype, const FillType ftype, const Point &origin,
			const double radius, const double height)
{
	double offsetXY = 0, offsetZ = 0;
	if (m_positioning == PP_CORNER)
		offsetXY = radius;
	else
	if (m_positioning == PP_CENTER)
		offsetZ = - height / 2.0;

	return addGeometry(otype, ftype,
		new Cylinder( Point( origin(0) + offsetXY, origin(1) + offsetXY, origin(2) + offsetZ ),
			radius, height, EulerParameters() )
	);
}

GeometryID XProblem::addCone(const GeometryType otype, const FillType ftype, const Point &origin,
	const double bottom_radius, const double top_radius, const double height)
{
	double offsetXY = 0, offsetZ = 0;
	if (m_positioning == PP_CORNER)
		offsetXY = bottom_radius;
	else
	if (m_positioning == PP_CENTER) {
		// height of the geometrical center of the truncated cone (round frustum)
		offsetZ = - height *
			(bottom_radius * bottom_radius + 2.0 * bottom_radius * top_radius + 3.0 * top_radius * top_radius)/
			(4.0 * (bottom_radius * bottom_radius + bottom_radius * top_radius + top_radius * top_radius));
		// center of the bounding box
		// offsetZ = height / 2.0;
	}

	return addGeometry(otype, ftype,
		new Cone( Point( origin(0) + offsetXY, origin(1) + offsetXY, origin(2) + offsetZ ),
			bottom_radius, top_radius, height, EulerParameters() )
	);
}

GeometryID XProblem::addSphere(const GeometryType otype, const FillType ftype, const Point &origin,
	const double radius)
{
	double offsetXY = 0, offsetZ = 0;
	if (m_positioning == PP_CORNER || m_positioning == PP_BOTTOM_CENTER)
		offsetZ = radius;
	if (m_positioning == PP_CORNER)
		offsetXY = radius;

	return addGeometry(otype, ftype,
		new Sphere( Point( origin(0) + offsetXY, origin(1) + offsetXY, origin(2) + offsetZ ),
			radius )
	);
}

GeometryID XProblem::addTorus(const GeometryType otype, const FillType ftype, const Point &origin,
	const double major_radius, const double minor_radius)
{
	if (otype == GT_FLOATING_BODY) {
		printf("WARNING: torus not yet supported as floating body, use mesh instead. Ignoring\n");
		return GEOMETRY_ERROR;
	}

	double offsetXY = 0, offsetZ = 0;
	if (m_positioning == PP_CORNER || m_positioning == PP_BOTTOM_CENTER)
		offsetZ = minor_radius;
	if (m_positioning == PP_CORNER)
		offsetXY = (major_radius + minor_radius);

	return addGeometry(otype, ftype,
		new Torus( origin, major_radius, minor_radius, EulerParameters() )
	);
}

GeometryID XProblem::addPlane(
	const double a_coeff, const double b_coeff, const double c_coeff, const double d_coeff)
{
	return addGeometry(GT_PLANE, FT_NOFILL,
		new Plane( a_coeff, b_coeff, c_coeff, d_coeff )
	);
}

// NOTE: "origin" has a slightly different meaning than for the other primitives: here it is actually
// an offset to shift the STL coordinates. Use 0 to import STL coords as they are.
// If positioning is PP_NONE and origin is (0,0,0), mesh coordinates are imported unaltered.
GeometryID XProblem::addSTLMesh(const GeometryType otype, const FillType ftype, const Point &origin,
	const char *filename)
{
	STLMesh *stlmesh = STLMesh::load_stl(filename);

	double offsetX = 0, offsetY = 0, offsetZ = 0;

	// handle positioning
	if (m_positioning != PP_NONE) {

		// Make the origin coincide with the lower corner of the mesh bbox.
		// Now the positioning is PP_CORNER
		stlmesh->shift( - stlmesh->get_minbounds() );

		// NOTE: STLMesh::get_meshsize() returns #triangles instead
		const double3 mesh_size = stlmesh->get_maxbounds() - stlmesh->get_minbounds();

		if (m_positioning == PP_CENTER || m_positioning == PP_BOTTOM_CENTER) {
			offsetX = - mesh_size.x / 2.0;
			offsetY = - mesh_size.y / 2.0;
		}

		if (m_positioning == PP_CENTER) {
			offsetZ = - mesh_size.z / 2.0;
		}

	} // if positioning is PP_NONE

	// shift STL origin to given point
	stlmesh->shift( make_double3(origin(0) + offsetX, origin(1) + offsetY, origin(2) + offsetZ) );

	return addGeometry(otype, ftype,
		stlmesh
	);
}

// NOTE: particles loaded from HDF5 files will not be erased!
// To enable erase-like interaction we need to copy them to the particle vectors, which
// requires unnecessary memory allocation
GeometryID XProblem::addHDF5File(const GeometryType otype, const Point &origin,
	const char *fname_hdf5, const char *fname_stl)
{
	// NOTES about HDF5 files
	// - fill type is FT_NOFILL since particles are read from file
	// - may add a null STLMesh if the hdf5 file is given but not the mesh

	// create an empty STLMesh if the STL filename is not given
	STLMesh *stlmesh = ( fname_stl == NULL ? new STLMesh(0) : STLMesh::load_stl(fname_stl) );

	// TODO: handle positioning like in addSTLMesh()? Better, simply trust the hdf5 file

	return addGeometry(otype, FT_NOFILL,
		stlmesh,
		fname_hdf5
	);
}

void XProblem::deleteGeometry(const GeometryID gid)
{
	m_geometries[gid]->enabled = false;

	// and this is the reason why m_numActiveGeometries not be used to iterate on m_geometries:
	m_numActiveGeometries--;

	if (m_geometries[gid]->ptr->m_ODEBody || m_geometries[gid]->ptr->m_ODEGeom)
		m_numRigidBodies--;

	if (m_geometries[gid]->type == GT_PLANE)
		m_numPlanes--;

	// TODO: print a warning if deletion is requested after fill_parts
}

void XProblem::enableDynamics(const GeometryID gid)
{
	// ensure geometry was not deleted
	if (!m_geometries[gid]->enabled) {
		printf("WARNING: trying to enable dynamics on a deleted geometry! Ignoring\n");
		return;
	}
	// ensure dynamics are consistent with geometry type
	if (m_geometries[gid]->type != GT_FLOATING_BODY &&
		m_geometries[gid]->type != GT_MOVING_BODY) {
		printf("WARNING: dynamics only available for rigid bodies! Ignoring\n");
		return;
	}
	m_geometries[gid]->handle_dynamics = true;
}

void XProblem::enableCollisions(const GeometryID gid)
{
	if (!validGeometry(gid)) return;

	// ensure collisions are consistent with geometry type
	if (m_geometries[gid]->type != GT_FLOATING_BODY &&
		m_geometries[gid]->type != GT_MOVING_BODY &&
		m_geometries[gid]->type != GT_PLANE) {
		printf("WARNING: collisions only available for rigid bodies and planes! Ignoring\n");
		return;
	}
	m_geometries[gid]->handle_collisions = true;
}

void XProblem::disableDynamics(const GeometryID gid)
{
	if (!validGeometry(gid)) return;

	// ensure no-dynamics is consistent with geometry type
	if (m_geometries[gid]->type == GT_FLOATING_BODY) {
		printf("WARNING: dynamics are mandatory for floating bodies! Ignoring\n");
		return;
	}
	m_geometries[gid]->handle_dynamics = false;
}

void XProblem::disableCollisions(const GeometryID gid)
{
	if (!validGeometry(gid)) return;

	// it is possible to disable collisions for any geometry type, so no need to check it
	m_geometries[gid]->handle_collisions = false;
}

// NOTE: GPUSPH uses ZXZ angles counterclockwise, while ODE XYZ clockwise (http://goo.gl/bV4Zeb - http://goo.gl/oPnMCv)
void XProblem::setOrientation(const GeometryID gid, const EulerParameters &ep)
{
	if (!validGeometry(gid)) return;

	m_geometries[gid]->ptr->setEulerParameters(ep);
}

void XProblem::setOrientation(const GeometryID gid, const dQuaternion quat)
{
	if (!validGeometry(gid)) return;

	m_geometries[gid]->ptr->setEulerParameters( EulerParameters(quat) );
}

void XProblem::rotate(const GeometryID gid, const dQuaternion quat)
{
	if (!validGeometry(gid)) return;

	// will compute qNewOrientation as qCurrentOrientation + requested rotation (quat)
	dQuaternion qCurrentOrientation, qNewOrientation;

	// read current orientation
	m_geometries[gid]->ptr->getEulerParameters()->ToODEQuaternion(qCurrentOrientation);

	// add requested rotation
	dQMultiply0(qNewOrientation, quat, qCurrentOrientation);

	// set the new orientation
	setOrientation( gid, qNewOrientation );
}


// NOTE: rotates X first, then Y, then Z
void XProblem::rotate(const GeometryID gid, const double Xrot, const double Yrot, const double Zrot)
{
	if (!validGeometry(gid)) return;

	// multiple temporary variables to keep code readable
	dQuaternion qX, qY, qZ, qXY, qXYZ;

	// compute single-axes rotations
	// NOTE: ODE uses clockwise angles for Euler, thus we invert them
	dQFromAxisAndAngle(qX, 1.0, 0.0, 0.0, -Xrot);
	dQFromAxisAndAngle(qY, 0.0, 1.0, 0.0, -Yrot);
	dQFromAxisAndAngle(qZ, 0.0, 0.0, 1.0, -Zrot);

	// concatenate rotations in order (X, Y, Z)
	dQMultiply0(qXY, qY, qX);
	dQMultiply0(qXYZ, qZ, qXY);

	// NOTE: we could use the unified ODE method dRFromEulerAngles(); however, it
	// is poorly documented and it is not clear what is the order of the rotations.
	// It would have been:
	// dMatrix3 Rotation;
	// dRFromEulerAngles(Rotation, Xrot, Yrot, Zrot);
	// dRtoQ(Rotation, qXYZ);
	// rotate(gid, qXYZ);

	// rotate with computed quaternion
	rotate( gid, qXYZ );
}

void XProblem::setIntersectionType(const GeometryID gid, IntersectionType i_type)
{
	if (!validGeometry(gid)) return;

	m_geometries[gid]->intersection_type = i_type;
}

void XProblem::setEraseOperation(const GeometryID gid, EraseOperation e_operation)
{
	if (!validGeometry(gid)) return;

	m_geometries[gid]->erase_operation = e_operation;
}

void XProblem::setMass(const GeometryID gid, const double mass)
{
	if (!validGeometry(gid)) return;

	if (m_geometries[gid]->type != GT_FLOATING_BODY)
		printf("WARNING: setting mass of a non-floating body\n");
	m_geometries[gid]->ptr->SetMass(mass);
}

double XProblem::setMassByDensity(const GeometryID gid, const double density)
{
	if (!validGeometry(gid)) return NAN;

	if (m_geometries[gid]->type != GT_FLOATING_BODY)
		printf("WARNING: setting mass of a non-floating body\n");
	return m_geometries[gid]->ptr->SetMass(m_physparams.r0, density);
}

const GeometryInfo* XProblem::getGeometryInfo(GeometryID gid)
{
	// ensure gid refers to a valid position
	if (gid >= m_geometries.size()) {
		printf("WARNING: invalid GeometryID %u\n");
		return NULL;
	}

	// return geometry even if deleted
	return m_geometries[gid];
}

// set the positioning policy for geometries added after the call
void XProblem::setPositioning(PositioningPolicy positioning)
{
	m_positioning = positioning;
}

// Create 6 planes delimiting the box defined by the two points and update (overwrite) the world origin and size.
// Write their GeometryIDs in planesIds, if given, so that it is possible to delete one or more of them afterwards.
void XProblem::makeUniverseBox(const double3 corner1, const double3 corner2, GeometryID *planesIds)
{
	// compute min and max
	double3 min, max;
	min.x = std::min(corner1.x, corner2.x);
	min.y = std::min(corner1.y, corner2.y);
	min.z = std::min(corner1.z, corner2.z);
	max.x = std::max(corner1.x, corner2.x);
	max.y = std::max(corner1.y, corner2.y);
	max.z = std::max(corner1.z, corner2.z);

	// create planes
	GeometryID plane_min_x = addPlane(  1,  0,  0, -min.x);
	GeometryID plane_max_x = addPlane( -1,  0,  0,  max.x);
	GeometryID plane_min_y = addPlane(  0,  1,  0, -min.y);
	GeometryID plane_max_y = addPlane(  0, -1,  0,  max.y);
	GeometryID plane_min_z = addPlane(  0,  0,  1, -min.z);
	GeometryID plane_max_z = addPlane(  0,  0, -1,  max.z);

	// set world origin and size
	m_origin = min;
	m_size = max - min;

	// write in output
	if (planesIds) {
		planesIds[0] = plane_min_x;
		planesIds[1] = plane_max_x;
		planesIds[2] = plane_min_y;
		planesIds[3] = plane_max_y;
		planesIds[4] = plane_min_z;
		planesIds[5] = plane_max_z;
	}
}

void XProblem::addExtraWorldMargin(const double margin)
{
	if (margin >= 0.0)
		m_extra_world_margin = margin;
	else
		printf("WARNING: tried to add negative world margin! Ignoring\n");
}

int XProblem::fill_parts()
{
	// if for debug reason we need to test the position and verse of a plane, we can ask ODE to
	// compute the distance of a probe point from a plane (positive if penetrated, negative out)
	/*
	double3 probe_point = make_double3 (0.5, 0.5, 0.5);
	printf("Test: probe point is distant %g from the bottom plane.\n,
		dGeomPlanePointDepth(m_box_planes[0], probe_point.x, probe_point.y, probe_point.z)
	*/

	//uint particleCounter = 0;
	uint bodies_parts_counter = 0;
	uint hdf5file_parts_counter = 0;

	for (size_t i = 0, num_geoms = m_geometries.size(); i < num_geoms; i++) {
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

		// set part mass
		if (m_geometries[i]->type == GT_FLUID)
			m_geometries[i]->ptr->SetPartMass(m_deltap, m_physparams.rho0[0]);
		else
		if (m_geometries[i]->type != GT_PLANE)
			m_geometries[i]->ptr->SetPartMass(dx, m_physparams.rho0[0]);

		// prepare for erase operations
		bool del_fluid = (m_geometries[i]->erase_operation == ET_ERASE_FLUID);
		bool del_bound = (m_geometries[i]->erase_operation == ET_ERASE_BOUNDARY);
		if (m_geometries[i]->erase_operation == ET_ERASE_ALL) del_fluid = del_bound = true;

		// erase operations with existent geometries
		if (del_fluid) {
			if (m_geometries[i]->intersection_type == IT_SUBTRACT)
				m_geometries[i]->ptr->Unfill(m_fluidParts, dx);
			else
				m_geometries[i]->ptr->Intersect(m_fluidParts, dx);
		}
		if (del_bound) {
			if (m_geometries[i]->intersection_type == IT_SUBTRACT)
				m_geometries[i]->ptr->Unfill(m_boundaryParts, dx);
			else
				m_geometries[i]->ptr->Intersect(m_boundaryParts, dx);
		}

		// after making some space, fill
		switch (m_geometries[i]->fill_type) {
			case FT_BORDER:
				m_geometries[i]->ptr->FillBorder(*parts_vector, m_deltap);
				break;
			case FT_SOLID:
				m_geometries[i]->ptr->Fill(*parts_vector, m_deltap);
				break;
			case FT_SOLID_BORDERLESS:
				printf("WARNING: borderless not yet implemented, filling with border\n");
				break;
			// case FT_NOFILL: ;
			// yes, it is legal to have no "default:": ISO/IEC 9899:1999, section 6.8.4.2
		}

		// geometries loaded from HDF5file do not undergo filling, but should be counted as well
		if (m_geometries[i]->has_hdf5_file) {
			m_hdf5_reader.setFilename(m_geometries[i]->hdf5_filename);
			hdf5file_parts_counter += m_hdf5_reader.getNParts();
			// do not reset() file reader: if we load only one, we'll avoid re-reading file header
		}

#if 0
		// dbg: fill horizontal XY planes with particles, only within the world domain
		if (m_geometries[i]->type == GT_PLANE) {
			Plane *plane = (Plane*)(m_geometries[i]->ptr);
			// only XY planes planes
			if (! (plane->getA() == 0 && plane->getB() == 0) )
				continue;
			// fill will print a warning
			// NOTE: since parts are added to m_boundaryParts, setting part mass is probably pointless
			plane->SetPartMass(dx, m_physparams.rho0[0]);
			// will round r0 to fit each dimension
			const uint xpn = (uint) trunc(m_size.x / m_physparams.r0 + 0.5);
			const uint ypn = (uint) trunc(m_size.y / m_physparams.r0 + 0.5);
			// compute Z
			const double z_coord = - plane->getD() / plane->getC();
			// aux vectors
			const Point start = Point(m_origin.x, m_origin.y, z_coord);
			const Vector v1 = Vector(m_size.x / xpn, 0, 0);
			const Vector v2 = Vector(0, m_size.x / ypn, 0);
			// fill
			parts_vector = &m_boundaryParts;
			for (uint xp = 0; xp <= xpn; xp++)
				for (uint yp = 0; yp <= ypn; yp++)
					parts_vector->push_back( Point( start + xp * v1 + yp*v2 ) );
		}
#endif

		// ODE-related operations - only if we have at least one floating body
		if (m_numRigidBodies > 0) {

			// We should not call both ODEBodyCreate() and ODEGeomCreate(), since the former
			// calls the latter in a dummy way if no ODE space is passed and this messes
			// up the position for the actual later ODEGeomCreate() call.
			// Thus, special call if both are active, individual calls otherwise.
			if (m_geometries[i]->handle_dynamics && m_geometries[i]->handle_collisions)
				// both body and geom
				m_geometries[i]->ptr->ODEBodyCreate(m_ODEWorld, m_deltap, m_ODESpace);
			else {
				// only one is active
				if (m_geometries[i]->handle_dynamics)
					m_geometries[i]->ptr->ODEBodyCreate(m_ODEWorld, m_deltap);
				if (m_geometries[i]->handle_collisions)
					m_geometries[i]->ptr->ODEGeomCreate(m_ODESpace, m_deltap);
			}

			// dynamics-only stuff
			if (m_geometries[i]->handle_dynamics) {
				add_ODE_body(m_geometries[i]->ptr);
				bodies_parts_counter += m_geometries[i]->ptr->GetParts().size();
			}

			// update ODE rotation matrix according to possible rotation - excl. planes!
			if ((m_geometries[i]->handle_collisions || m_geometries[i]->handle_dynamics) &&
				m_geometries[i]->type != GT_PLANE)
				m_geometries[i]->ptr->updateODERotMatrix();
		} // if m_numRigidBodies > 0
	}

	return m_fluidParts.size() + m_boundaryParts.size() + bodies_parts_counter + hdf5file_parts_counter;
}

uint XProblem::fill_planes()
{
	return m_numPlanes;
}

void XProblem::copy_planes(float4 *planes, float *planediv)
{
	if (m_numPlanes == 0) return;
	// look for planes
	uint currPlaneIdx = 0;
	// NOTE: could iterate on planes only with a map plane_index -> gid
	for (uint gid = 0, num_geoms = m_geometries.size(); gid < num_geoms; gid++) {

		// skip deleted
		if (! m_geometries[gid]->enabled) continue;

		// not a plane?
		if (m_geometries[gid]->type != GT_PLANE) continue;

		Plane *plane = (Plane*)(m_geometries[gid]->ptr);

		planes[currPlaneIdx] = make_float4( plane->getA(), plane->getB(), plane->getC(), plane->getD() );
		planediv[currPlaneIdx] = plane->getNorm();

		currPlaneIdx++;
	}
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

	// NOTEs and TODO
	// - Automatic hydrostatic filling. Or, callback?
	// - SA currently supported only from file. Support runtime generation?
	// - I/O support, inlcuding: setting IO_PARTICLE_FLAG, VEL_IO_PARTICLE_FLAG,
	//   INFLOW_PARTICLE_FLAG, MOVING_PARTICLE_FLAG, FLOATING_PARTICLE_FLAG.
	//   E.g. SET_FLAG(info[i], IO_PARTICLE_FLAG);
	// - Warn if loaded particle has different type than filled, but only once
	// - Save the id of the first boundary particle that belongs to an ODE object?
	//   Was in previous code but we probably don't need it

	// particles counters, by type
	uint fluid_parts = 0;
	uint boundary_parts = 0;
	uint vertex_parts = 0;
	// count #particles loaded from HDF5 files. Needed also to adjust connectivity afterward
	uint loaded_parts = 0;
	// Total number of filled parts, i.e. in GPUSPH array and ready to be uploaded. The following hold:
	//   total = fluid_parts + boundary_parts + vertex_parts
	//   total >= loaded_parts
	//   total >= object_parts
	uint tot_parts = 0;

	// store mass for each particle type
	double fluid_part_mass = NAN;
	double boundary_part_mass = NAN;
	double vertex_part_mass = NAN;

	// we use a simple map for the HDF5_id->id map translation (connectivity fix)
	std::map<uint, uint> hdf5idx_to_idx_map;

	// count how many particles will be loaded from file
	for (size_t g = 0, num_geoms = m_geometries.size(); g < num_geoms; g++)
		if (m_geometries[g]->has_hdf5_file) {
			m_hdf5_reader.setFilename(m_geometries[g]->hdf5_filename);
			loaded_parts += m_hdf5_reader.getNParts();
		}

	// copy filled fluid parts
	for (uint i = tot_parts; i < tot_parts + m_fluidParts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(FLUIDPART,0,i);
		calc_localpos_and_hash(m_fluidParts[i], info[i], pos[i], hash[i]);
		if (i == tot_parts)
			fluid_part_mass = pos[i].w;
	}
	tot_parts += m_fluidParts.size();
	fluid_parts += m_fluidParts.size();

	// copy filled boundary parts
	for (uint i = tot_parts; i < tot_parts + m_boundaryParts.size(); i++) {
		// TODO: eulerVel
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i] = make_particleinfo(BOUNDPART, 0, i);
		calc_localpos_and_hash(m_boundaryParts[i - tot_parts], info[i], pos[i], hash[i]);
		if (i == tot_parts)
			boundary_part_mass = pos[i].w;
	}
	tot_parts += m_boundaryParts.size();
	boundary_parts += m_boundaryParts.size();

	// We've already counted the objects in initialize(), but now we need incremental counters
	// to set the association objId->floatingObjId (m_ODEobjectId) and possibly print information
	uint object_counter = 0;
	uint rigid_body_counter = 0;
	// store particle mass of last added rigid body
	double rigid_body_part_mass = NAN;

	// Until now we copied fluid and boundary particles not belonging to floating objects and/or not to be loaded
	// from HDF5 files. Now we iterate on the geometries with the aim to
	// - load and copy HDF5 files (they are not copied to the global particle vectors, and that's the reason why
	//   currently no erase operations are supported with HDF5-loaded geometries);
	// - copy particles of floating objects, since they fill their own point vector;
	// - setup stuff related to floating objects (e.g. object particle count).
	for (size_t g = 0, num_geoms = m_geometries.size(); g < num_geoms; g++) {

		// planes do not fill particles nor they load from files
		if (m_geometries[g]->type == GT_PLANE)
			continue;

		// number of particles loaded or filled by the current geometry
		uint current_geometry_particles = 0;
		// special attention to number of boundary particles, used for rb buffers of floating objs
		uint current_geometry_boundary_particles = 0;

		const uint object_id = ( m_geometries[g]->type == GT_FLOATING_BODY ? object_counter : 0 );

		// load from HDF5 file, whether fluid, boundary, floating or else
		if (m_geometries[g]->has_hdf5_file) {
			// set and check filename
			m_hdf5_reader.setFilename(m_geometries[g]->hdf5_filename);
			// read number of particles
			current_geometry_particles = m_hdf5_reader.getNParts();
			// read particles
			m_hdf5_reader.read();
			// add every particle
			for (uint i = tot_parts; i < tot_parts + current_geometry_particles; i++) {

				// "i" is the particle index in GPUSPH host arrays, "bi" the one in current HDF5 file)
				const uint bi = i - tot_parts;

				// TODO: warning as follows? But should be printed only once
				// if (m_hdf5_reader.buf[bi].ParticleType != 0) ... warning, filling with different particle type
				//float rho = density(initial_water_level - m_hdf5_reader.buf[i].Coords_2, 0); // how to?
				float rho = m_physparams.rho0[0];
				vel[i] = make_float4(0, 0, 0, rho);
				//if (eulerVel)
				//	eulerVel[i] = make_float4(0);

				// TODO: define an invalid/unknown particle type?
				// NOTE: update particle counters here, since current_geometry_particles does not distinguish vertex/bound;
				// tot_parts instead is updated in the outer loop
				ushort ptype = FLUIDPART;
				switch (m_hdf5_reader.buf[bi].ParticleType) {
					case 1: // 2 aka CRIXUS_FLUID
						// TODO: warn user if (m_geometries[g]->type != GT_FLUID)
						ptype = FLUIDPART;
						fluid_parts++;
						break;
					case 2: // 2 aka CRIXUS_VERTEX
						// TODO: warn user if (m_geometries[g]->type == GT_FLUID)
						ptype = VERTEXPART;
						vertex_parts++;
						break;
					case 3: // 3 aka CRIXUS_BOUNDARY
						// TODO: warn user if (m_geometries[g]->type == GT_FLUID)
						ptype = BOUNDPART;
						boundary_parts++;
						current_geometry_boundary_particles++;
						break;
					default:
						// TODO: print warning or throw fatal
						break;
				}

				// compute particle info, local pos, cellhash
				// Remember: +1 since object 0 means no object (ugh)
				info[i] = make_particleinfo(ptype, object_id + 1, i);
				// not yet enabled?
				if (m_geometries[g]->type == GT_FLOATING_BODY) {
					SET_FLAG(info[i], FLOATING_PARTICLE_FLAG);
					SET_FLAG(info[i], MOVING_PARTICLE_FLAG);
				}
				// if (m_geometries[g]->type == GT_FLOATING_BODY)
				//	SET_FLAG(info[i], FLOATING_PARTICLE_FLAG);

				calc_localpos_and_hash(
					Point(m_hdf5_reader.buf[bi].Coords_0, m_hdf5_reader.buf[bi].Coords_1, m_hdf5_reader.buf[bi].Coords_2,
						m_physparams.rho0[0]*m_hdf5_reader.buf[bi].Volume),
					info[i], pos[i], hash[i]);

				// store particle mass for current type, if it was not store already
				if (ptype == FLUIDPART && !isfinite(fluid_part_mass))
					fluid_part_mass = pos[i].w;
				else
				if (ptype == BOUNDPART && !isfinite(boundary_part_mass))
					boundary_part_mass = pos[i].w;
				else
				if (ptype == VERTEXPART && !isfinite(vertex_part_mass))
					vertex_part_mass = pos[i].w;
				// also set rigid_body_part_mass, which is orthogonal the the previous values
				// TODO: with SA bounds, this value has little meaning or should be split
				if (m_geometries[g]->type == GT_FLOATING_BODY && !isfinite(rigid_body_part_mass))
					rigid_body_part_mass = pos[i].w;

				// load boundary-specific data (SA bounds only)
				if (ptype == BOUNDPART) {
					vertices[i].x = m_hdf5_reader.buf[bi].VertexParticle1;
					vertices[i].y = m_hdf5_reader.buf[bi].VertexParticle2;
					vertices[i].z = m_hdf5_reader.buf[bi].VertexParticle3;
					boundelm[i].x = m_hdf5_reader.buf[bi].Normal_0;
					boundelm[i].y = m_hdf5_reader.buf[bi].Normal_1;
					boundelm[i].z = m_hdf5_reader.buf[bi].Normal_2;
					boundelm[i].w = m_hdf5_reader.buf[bi].Surface;
				}

				// update hash map
				if (ptype == VERTEXPART)
					hdf5idx_to_idx_map[ m_hdf5_reader.buf[bi].AbsoluteIndex ] = i;
			}
			// free memory and prepare for next file
			m_hdf5_reader.reset();

			// set numParts, which will be read while allocating device buffers for obj parts
			m_geometries[g]->ptr->SetNumParts(current_geometry_boundary_particles);

		} // if (m_geometries[g]->has_hdf5_file)

		// copy particles from the point vector of objects which have not been loaded from file
		// FIXME: should include MOVING, maybe I/O; also, check: here assuming non-SA?
		if (m_geometries[g]->type == GT_FLOATING_BODY && !(m_geometries[g]->has_hdf5_file)) {
			// not loading from file: take object vector
			PointVect & rbparts = m_geometries[g]->ptr->GetParts();
			current_geometry_particles = rbparts.size();
			// copy particles
			for (uint i = tot_parts; i < tot_parts + current_geometry_particles; i++) {
				vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
				info[i] = make_particleinfo(OBJECTPART, object_id + 1, i);
				calc_localpos_and_hash(rbparts[i - tot_parts], info[i], pos[i], hash[i]);
				// NOTE: setting/showing rigid_body_part_mass only makes sense with non-SA bounds
				if (m_geometries[g]->type == GT_FLOATING_BODY && !isfinite(rigid_body_part_mass))
					rigid_body_part_mass = pos[i].w;
			}
		}

		// floating-objects-related settings, regardless they were loaded from file or not
		if (m_geometries[g]->type == GT_FLOATING_BODY) {

			// store index (currently identical to id) of first object particle in m_firstODEobjectPartId
			// NOTE: relies on tot_parts not being updated yet
			if (current_geometry_particles > 0 && m_firstODEobjectPartId == 0){
				if (m_simparams.boundarytype == SA_BOUNDARY)
					// SA bounds: we want the first boundary element and we know vertices are filled first.
					m_firstODEobjectPartId = tot_parts + current_geometry_particles - current_geometry_boundary_particles;
				else
					// Other: the index (thus the id) of the first part of the object will do
					m_firstODEobjectPartId = tot_parts;
			}

			// recap on stdout
			std::cout << " - Rigid body " << rigid_body_counter << ": " << current_geometry_particles << " particles";
			std::cout << ", part mass: " << rigid_body_part_mass << "\n";

			// reset value to spot possible anomalies in next bodies
			rigid_body_part_mass = NAN;
			// update counter
			rigid_body_counter++;
		}

		// update counter of objects
		if (m_geometries[g]->type == GT_FLOATING_BODY ||
			m_geometries[g]->type == GT_MOVING_BODY   ||
			m_geometries[g]->type == GT_OPENBOUNDARY )
			object_counter++;

		// update global particle counter
		tot_parts += current_geometry_particles;

	} // for each geometry

	// fix connectivity by replacing Crixus' AbsoluteIndex with local index
	// TODO: instead of iterating on all the particles, we could create a list of boundary particles while
	// loading them from file, and here iterate only on that vector
	if (loaded_parts > 0) {
		std::cout << "Fixing connectivity..." << std::flush;
		for (uint i=0; i< tot_parts; i++)
			if (BOUNDARY(info[i])) {
				if (hdf5idx_to_idx_map.count(vertices[i].x) == 0 ||
					hdf5idx_to_idx_map.count(vertices[i].y) == 0 ||
					hdf5idx_to_idx_map.count(vertices[i].z) == 0 ) {
					printf("FATAL: connectivity: particle id %u index %u loaded from HDF5 points to non-existing vertices (%u,%u,%u)!\n",
						id(info[i]), i, vertices[i].x, vertices[i].y, vertices[i].z );
					exit(1);
				} else {
					vertices[i].x = id(info[ hdf5idx_to_idx_map.find(vertices[i].x)->second ]);
					vertices[i].y = id(info[ hdf5idx_to_idx_map.find(vertices[i].y)->second ]);
					vertices[i].z = id(info[ hdf5idx_to_idx_map.find(vertices[i].z)->second ]);
				}
			}
		std::cout << "DONE" << "\n";
		hdf5idx_to_idx_map.clear();
	}

	std::cout << "Fluid parts: " << fluid_parts << "\n";
	std::cout << "Fluid part mass: " << fluid_part_mass << "\n";
	std::cout << "Boundary parts: " << boundary_parts << "\n";
	std::cout << "Boundary part mass: " << boundary_part_mass << "\n";
	if (m_simparams.boundarytype == SA_BOUNDARY) {
		std::cout << "Vertex parts: " << vertex_parts << "\n";
		std::cout << "Vertex part mass: " << vertex_part_mass << "\n";
	}
	std::cout << "Tot parts: " << tot_parts << "\n";
	std::flush(std::cout);
}

/*void
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
