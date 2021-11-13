/*  Copyright (c) 2011-2019 INGV, EDF, UniCT, JHU

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

/*! \file
 * Implementation of the high-level interface for Problem defintions
 */

#include <string>
#include <iostream>

// limits
#include <cfloat>
#include <limits>

#include "ProblemAPI_1.h"

#if USE_CHRONO == 1
#include "chrono/physics/ChSystem.h"
#include "chrono/fea/ChLinkPointTriface.h"
#endif

#include "Segment.h"
#include "Rect.h"
#include "Disk.h"
#include "Cube.h"
#include "Cylinder.h"
#include "Cone.h"
#include "Sphere.h"
#include "Torus.h"
#include "Plane.h"
#include "STLMesh.h"
#include "TopoCube.h"
#include "GlobalData.h"

//#define USE_PLANES 0

using namespace std;

ProblemAPI<1>::ProblemAPI(GlobalData *_gdata) : ProblemCore(_gdata)
{
	// *** XProblem initialization
	m_numActiveGeometries = 0;
	m_numForcesBodies = 0;
	m_numFloatingBodies = 0;
	m_numFEAObjects = 0;
	m_numPlanes = 0;
	m_numOpenBoundaries = 0;

	m_numBoundLayers = 0;

	m_extra_world_margin = 0.0;

	m_positioning = PP_CENTER;

	m_dem_geometry = INVALID_GEOMETRY;
	m_dem_zmin_scale = 5.0;
	m_dem_dx_scale = 5.0;
	m_dem_dy_scale = 5.0;

	// NAN water level and max fall: will autocompute if user doesn't define them
	m_maxFall = NAN;
	m_maxParticleSpeed = NAN;

	// We don't initialize simparams because it will be done by the SETUP_FRAMEWORK
	// in the subclass

}

void ProblemAPI<1>::release_memory()
{
	m_fluidParts.clear();
	m_boundaryParts.clear();
	m_testpointParts.clear();
	// also cleanup object parts
	for (size_t g = 0, num_geoms = m_geometries.size(); g < num_geoms; g++) {
		if (m_geometries[g]->enabled)
			m_geometries[g]->ptr->GetParts().clear();
		if (m_geometries[g]->hdf5_reader)
			delete m_geometries[g]->hdf5_reader;
		if (m_geometries[g]->xyz_reader)
			delete m_geometries[g]->xyz_reader;
	}
}

uint ProblemAPI<1>::suggestedNumBoundaryLayers()
{
	return simparams()->boundary_is_multilayer() ?
		(uint)simparams()->get_influence_layers() :
		1;
}

double ProblemAPI<1>::preferredDeltaP(GeometryType type)
{
	if (std::isnan(m_deltap))
		throw runtime_error("cannot determine optimal inter-particle spacing, deltap not set");
	if (std::isnan(physparams()->r0)) {
		physparams()->r0 = m_deltap;
		printf("Setting wall inter-particle spacing from deltap: %g\n", physparams()->r0);
	}
	return type == GT_FLUID ? m_deltap : physparams()->r0;
}

ProblemAPI<1>::~ProblemAPI<1>()
{
	release_memory();
	if (m_numFloatingBodies || m_numFEAObjects)
		cleanupChrono();
}

bool ProblemAPI<1>::initialize()
{
	// setup the framework if the subclass did not do it; will have all defaults
	if (!simframework()) {
		// TODO automatic framework setup
		// This must be done in a CU file
		//SETUP_FRAMEWORK();
		throw runtime_error("no simulation framework defined");
	}

	// If we have a DEM, and it's set to not fill, the simulation framework must have
	// the ENABLE_DEM flag, otherwise there will be no interaction with the topography
	if (validGeometry(m_dem_geometry))
	{
		if ((m_geometries[m_dem_geometry]->fill_type == FT_NOFILL) && !HAS_DEM(simparams()->simflags))
			throw invalid_argument("DEM with FT_NOFILL requires ENABLE_DEM flag");
	}

	// *** Add a writer, if none was specified
	if (get_writers().size() == 0)
		add_writer(VTKWRITER, 1e-2f);

	// *** Add DisplayWriter if visualization is enabled
	if (gdata->clOptions->visualization) {
		add_writer(DISPLAYWRITER, gdata->clOptions->visu_freq);
	}

	// *** Initialization of minimal physical parameters
	if (std::isnan(m_deltap))
		set_deltap(0.02f);
	if (std::isnan(physparams()->r0))
		physparams()->r0 = m_deltap;

	const uint dims = space_dimensions_for(simparams()->dimensions);

	// aux vars to compute bounding box
	Point globalMin = Point(DBL_MAX, DBL_MAX, DBL_MAX);
	Point globalMax = Point(-DBL_MAX, -DBL_MAX, -DBL_MAX);

	// counters of floating objects and generic objects (floating + moving + open bounds)
	uint bodies_counter = 0;
	uint open_boundaries_counter = 0;

	// aux var for automatic water level computation
	double highest_water_part = NAN;

	// Enable free surface boundaries if we are repacking
	bool enableFreeSurf = (gdata->clOptions->repack == true
		|| gdata->clOptions->repack_only == true);

	for (size_t g = 0, num_geoms = m_geometries.size(); g < num_geoms; g++) {
		// aux vars to store bbox of current geometry
		Point currMin, currMax;

		// ignore planes for bbox
		if (m_geometries[g]->type == GT_PLANE)
			continue;
		// Load the free-surface boundary particles when repacking only
		if (m_geometries[g]->type == GT_FREE_SURFACE)
			m_geometries[g]->enabled = enableFreeSurf;

		// ignore deleted geometries
		if (!m_geometries[g]->enabled)
			continue;

		// load HDF5 files
		if (m_geometries[g]->has_hdf5_file)
			m_geometries[g]->hdf5_reader->read();
		else
		// load XYZ files and store their bounding box, at the same time
		if (m_geometries[g]->has_xyz_file)
			m_geometries[g]->xyz_reader->read(&currMin, &currMax);

		// geometries loaded from HDF5 files but not featuring a STL file do not have a
		// bounding box yet; let's compute it
		if (m_geometries[g]->has_hdf5_file && !m_geometries[g]->has_mesh_file) {

			// initialize temp variables
			currMin(0) = currMin(1) = currMin(2) = DBL_MAX;
			currMax(0) = currMax(1) = currMax(2) = -DBL_MAX;

			// iterate on particles - could printf something if long...
			for (size_t p = 0; p < m_geometries[g]->hdf5_reader->getNParts(); p++) {
				// utility pointer & var
				ReadParticles *part = &(m_geometries[g]->hdf5_reader->buf[p]);
				Point currPoint(part->Coords_0, part->Coords_1, part->Coords_2);
				// set current per-coordinate minimum and maximum
				setMinPerElement(currMin, currPoint);
				setMaxPerElement(currMax, currPoint);
			}
			// TODO: store the so-computed bbox somewhere?
			// Not using it yet, but we have it for free here
		} else
		// points loaded from XYZ files already stored their bounding box; all other
		// geometries should have one ready
		if (!m_geometries[g]->has_xyz_file)
			m_geometries[g]->ptr->getBoundingBox(currMin, currMax);

		// global min and max
		setMinPerElement(globalMin, currMin);
		setMaxPerElement(globalMax, currMax);

		// store highest fluid part Z coordinate
		if (m_geometries[g]->type == GT_FLUID) {
			if (!isfinite(highest_water_part) || currMax(dims-1) > highest_water_part)
				highest_water_part = currMax(dims-1);
		}

		// update object counters
		if (m_geometries[g]->type == GT_FLOATING_BODY ||
			m_geometries[g]->type == GT_MOVING_BODY)
			bodies_counter++;
		if (m_geometries[g]->type == GT_OPENBOUNDARY)
			open_boundaries_counter++;
	}

	// here should be bodies_counter == m_numFloatingBodies
	if ( bodies_counter > MAX_BODIES ) {
		printf("Fatal: number of bodies > MAX_BODIES (%u > %u)\n",
			bodies_counter, MAX_BODIES);
		return false;
	}

	// do not store the number of floating objects in simparams: add_moving_body()
	// will increment it and use it for the insertion in the vector

	// store number of objects (floating + moving + I/O)
	simparams()->numOpenBoundaries = open_boundaries_counter;

	// compute the number of layers needed by the boundary model, if not set
	if (m_numBoundLayers == 0) {
		// force autocomputation
		m_numBoundLayers = getNumBoundaryLayers();
	}

	// Adjust the world dimensions to account for periodicity and outer boundary filling.
	// The logic is that in periodic directions we add half a dp on either side,
	// so that particles on either side of the periodic wall are at a distance of half a dp,
	// and in non-periodic directions we increase the size by number-of-layers dp
	// (this is half a dp too much when using BORDER_CENTERED filling method, but that's OK.

	const bool is_periodic_x = !!(simparams()->periodicbound & PERIODIC_X);
	const bool is_periodic_y = !!(simparams()->periodicbound & PERIODIC_Y);
	const bool is_periodic_z = !!(simparams()->periodicbound & PERIODIC_Z);

	const double periodic_margin = Object::get_default_filling_method() == Object::BORDER_CENTERED ? m_deltap/2 : 0;
	const double non_periodic_margin = m_numBoundLayers*m_deltap;

	globalMin(0) -= is_periodic_x ? periodic_margin : non_periodic_margin;
	globalMax(0) += is_periodic_x ? periodic_margin : non_periodic_margin;

	if (dims > 1) {
		globalMin(1) -= is_periodic_y ? periodic_margin : non_periodic_margin;
		globalMax(1) += is_periodic_y ? periodic_margin : non_periodic_margin;
	}
	if (dims > 2) {
		globalMin(2) -= is_periodic_z ? periodic_margin : non_periodic_margin;
		globalMax(2) += is_periodic_z ? periodic_margin : non_periodic_margin;
	}

	// set computed world origin and size without overriding possible user choices
	if (!isfinite(m_origin.x)) m_origin.x = globalMin(0);
	if (!isfinite(m_origin.y)) m_origin.y = globalMin(1);
	if (!isfinite(m_origin.z)) m_origin.z = globalMin(2);
	if (!isfinite(m_size.x)) m_size.x = globalMax(0) - globalMin(0);
	if (!isfinite(m_size.y)) m_size.y = globalMax(1) - globalMin(1);
	if (!isfinite(m_size.z)) m_size.z = globalMax(2) - globalMin(2);

	// add user-defined world margin, if any
	if (m_extra_world_margin > 0.0) {
		m_origin.x -= m_extra_world_margin;
		m_size.x += 2 * m_extra_world_margin;
		if (dims > 1) {
			m_origin.y -= m_extra_world_margin;
			m_size.y += 2 * m_extra_world_margin;
		}
		if (dims > 2) {
			m_origin.z -= m_extra_world_margin;
			m_size.z += 2 * m_extra_world_margin;
		}
	}

	// TODO warn user if m_origin/m_size result in a smaller domain that we computed,
	// as this is usually a setup issue

	/* Compute the DEM position fixup, if needed */
	if (validGeometry(m_dem_geometry)) {
		const auto dem = static_pointer_cast<TopoCube>(m_geometries[m_dem_geometry]->ptr);
		const float ewres = dem->get_ewres();
		const float nsres = dem->get_nsres();
		const Point& dem_origin = dem->get_origin();
		auto& fixup = physparams()->dem_pos_fixup;
		fixup =	make_float2(
			(m_origin.x - dem_origin(0))/ewres,
			(m_origin.y - dem_origin(1))/nsres);
		printf("DEM position fixup: (%g, %g)\n", fixup.x, fixup.y);
	}


	// compute water level automatically, if not set
	if (!isfinite(m_waterLevel)) {
		// water level: highest fluid coordinate or (absolute) domain height
		m_waterLevel = ( isfinite(highest_water_part) ? highest_water_part : m_size.z - m_origin.z );
		printf("Water level not set, autocomputed: %g\n", m_waterLevel);
	}

	// ditto for max fall; approximated as (waterLevel - lowest_domain_point)
	// NOTE: if there is no fluid geometry and both water level and maxFall are autocomputed, then
	// water level will be equal to highest domain point and max fall to domain height
	if (!isfinite(m_maxFall)) {
		m_maxFall = m_waterLevel - globalMin(2);
		printf("Max fall height not set, autocomputed: %g\n", m_maxFall);
	}

	// set physical parameters depending on m_maxFall or m_waterLevel: LJ dcoeff, sspeed (through set_density())
	const float g = length(physparams()->gravity);

	// The LJ d coefficient is used by LJ_BOUNDARY, but also by the repulsive force of planes and DEM
	const bool needs_dcoeff = (simparams()->boundarytype == LJ_BOUNDARY) || HAS_DEM_OR_PLANES(simparams()->simflags);
	if (needs_dcoeff && !isfinite(physparams()->dcoeff)) {
		physparams()->dcoeff = 5.0f * g * m_maxFall;
		printf("Lennard–Jones D coefficient not set, autocomputed: %g\n", physparams()->dcoeff);
	}

	if (simparams()->boundarytype == MK_BOUNDARY && !isfinite(physparams()->MK_K)) {
		physparams()->MK_K = g * m_maxFall;
		printf("Monaghan–Kajtar K coefficient not set, autocomputed: %g\n", physparams()->MK_K);
	}

	// hydrostatic filling works only if gravity has only vertical component and
	// there isn't a periodic boundary in the gravity direction
	// TODO When hydrostatic filling will be implemented to work with all directions,
	// the disabling condition below have to be changed
	if (g == 0)
		m_hydrostaticFilling = false;
	if (dims == 3 && (physparams()->gravity.x != 0 || physparams()->gravity.y != 0 ||
			(simparams()->periodicbound & PERIODIC_Z)))
	{
		m_hydrostaticFilling = false;
	}
	if (dims == 2 && (physparams()->gravity.x != 0 || (simparams()->periodicbound & PERIODIC_Y)))
	{
		m_hydrostaticFilling = false;
	}
	if (dims == 1 && (simparams()->periodicbound & PERIODIC_Y))
	{
		m_hydrostaticFilling = false;
	}


	// if multiple fluids, hydrostatic filling should be done by hand
	// in the problem-specific initilizeParticles
	if (physparams()->numFluids() > 1)
		m_hydrostaticFilling = false;

	// maximum fall velocity, computed from the maximum fall height
	double max_fall_velocity = sqrt(2.0 * g * m_maxFall);

	if (!isfinite(m_maxParticleSpeed)) {
		m_maxParticleSpeed = max_fall_velocity;
		printf("Max particle speed not set, autocomputed from max fall: %g\n", m_maxParticleSpeed);
	}

	// Note that in the general case, the maximum particle speed might be less than the maximum fall velocity
	// (think e.g. of a highly viscous Poiseuille test case). The default speed of sound will 
	// take into account the largest of these values
	double max_vel_for_c0 = fmax(max_fall_velocity, m_maxParticleSpeed);


	const float default_rho = 1000.0;
	const float default_kinematic_visc = 1.0e-6f;
	const float default_gamma = 7;
	// numerical speed of sound TODO multifluid
	const float default_c0 = 10.0 * max_vel_for_c0;

	if (physparams()->numFluids() == 0) {
		physparams()->add_fluid(default_rho);
		printf("No fluids specified, assuming water (rho: %g\n",
			default_rho);
	}

	for (size_t fluid = 0 ; fluid < physparams()->numFluids(); ++fluid) {
		const bool must_set_gamma = std::isnan(physparams()->gammacoeff[fluid]);
		const bool must_set_c0 = std::isnan(physparams()->sscoeff[fluid]);

		// tell the user what we're going to do
		if (must_set_gamma && must_set_c0) {
			printf("EOS for fluid %zu not specified, assuming water (gamma: %g, c0: %g)\n",
				fluid, default_gamma, default_c0);
		} else if (must_set_c0) {
			printf("Speed of sound for fluid %zu auto-computed as c0 = %g\n",
				fluid, default_c0);
		} else if (must_set_gamma) {
			// we consider this an anomalous situation, since gamma should always
			// be specified if c0 was, hence stderr
			fprintf(stderr, "Incomplete EOS for fluid %zu, assuming water (gamma: %g)\n",
				fluid, default_gamma);
		}

		// set the EOS if needed
		if (must_set_gamma || must_set_c0)
			physparams()->set_equation_of_state(
				fluid,
				must_set_gamma ? default_gamma : physparams()->gammacoeff[fluid],
				must_set_c0 ? default_c0 : physparams()->sscoeff[fluid]);

		// set the viscosity if needed
		if (std::isnan(physparams()->kinematicvisc[fluid])) {
			printf("Viscosity for fluid %zu not specified, assuming water (nu = %g)\n",
				fluid, default_kinematic_visc);
			physparams()->set_kinematic_visc(fluid, default_kinematic_visc);
		}
	}

	// only init Chrono if there are floating bodies
	if (m_numFloatingBodies || m_numFEAObjects)
		initializeChrono();

	// check open boundaries consistency
	// TODO ideally we should enable/disable them depending on whether
	// they are present, but this isn't trivial to do with the static framework
	// options
	if (m_numOpenBoundaries > 0 && !HAS_INLET_OUTLET(simparams()->simflags))
		throw invalid_argument("open boundaries present, but ENABLE_INLET_OUTLET not specified in framework flag");
	if (m_numOpenBoundaries == 0 && HAS_INLET_OUTLET(simparams()->simflags))
		throw invalid_argument("no open boundaries present, but ENABLE_INLET_OUTLET specified in framework flag");

	// TODO FIXME m_numMovingObjects does not exist yet
	//if (m_numMovingObjects > 0)
	//	simparams()->movingBoundaries = true;

	// Call Problem's initialization that takes care of the common
	// initialization functions (checking dt, preparing the grid,
	// creating the problem dir, etc)
	return ProblemCore::initialize();
}

void ProblemAPI<1>::initializeChrono()
{
#if USE_CHRONO == 1
	InitializeChrono();
#endif
}

void ProblemAPI<1>::cleanupChrono()
{
#if USE_CHRONO == 1
	FinalizeChrono();
#endif
}

GeometryID ProblemAPI<1>::addGeometry(const GeometryType otype, const FillType ftype, ObjectPtr obj_ptr,
	const char *hdf5_fname, const char *xyz_fname, const char *stl_fname)
{
	// TODO: before even creating the new GeometryInfo we should check the compatibility of
	// the combination of paramenters (e.g. no moving planes; no HDF5 testpoints)
	GeometryInfo* geomInfo = new GeometryInfo();
	geomInfo->type = otype;
	geomInfo->fill_type = ftype;
IGNORE_WARNINGS(deprecated-declarations)
	if (ftype == FT_BORDER) {
		// convert FT_BORDER to INNER for 3D geometries, OUTER for Rect and Disk
		FillType alt_ftype =
			(dynamic_pointer_cast<Rect>(obj_ptr) || dynamic_pointer_cast<Disk>(obj_ptr)) ?
			FT_OUTER_BORDER :
			FT_INNER_BORDER ;
		geomInfo->fill_type = alt_ftype;
		cerr << "Deprecated FT_BORDER converted to FT_OUTER_BORDER" << (alt_ftype == FT_OUTER_BORDER ?
			"FT_OUTER_BORDER" : "FT_INNER_BORDER") << endl;
	}
RESTORE_WARNINGS
	geomInfo->ptr = obj_ptr;
	if (hdf5_fname) {
		geomInfo->hdf5_filename = string(hdf5_fname);
		geomInfo->has_hdf5_file = true;
		// initialize the reader
		// TODO: error checking
		geomInfo->hdf5_reader = new HDF5SphReader();
		geomInfo->hdf5_reader->setFilename(hdf5_fname);
	} else
	if (xyz_fname) {
		geomInfo->xyz_filename = string(xyz_fname);
		geomInfo->has_xyz_file = true;
		// initialize the reader
		// TODO: error checking
		geomInfo->xyz_reader = new XYZReader();
		geomInfo->xyz_reader->setFilename(xyz_fname);
	}
	if (stl_fname) {
		geomInfo->stl_filename = string(stl_fname);
		geomInfo->has_mesh_file = true;
	}
	m_numActiveGeometries++;

	// --- Default collision and dynamics
	switch (geomInfo->type) {
		case GT_FLUID:
			geomInfo->handle_collisions = false;
			geomInfo->handle_dynamics = false;
			geomInfo->measure_forces = false;
			geomInfo->fea = false;
			break;
		case GT_FIXED_BOUNDARY:
			geomInfo->handle_collisions = true; // optional
			geomInfo->handle_dynamics = false;
			geomInfo->measure_forces = false;
			geomInfo->fea = false;
			break;
		case GT_OPENBOUNDARY:
			geomInfo->handle_collisions = false; // TODO: make optional?
			geomInfo->handle_dynamics = false;
			geomInfo->measure_forces = false; // TODO: make optional?
			geomInfo->fea = false;
			break;
		case GT_FLOATING_BODY:
			geomInfo->handle_collisions = true; // optional
			geomInfo->handle_dynamics = true;
			geomInfo->measure_forces = true;
			geomInfo->fea = false;
			break;
		case GT_MOVING_BODY:
			geomInfo->handle_collisions = true; // optional
			geomInfo->handle_dynamics = false; // optional
			geomInfo->measure_forces = false; // optional
			geomInfo->fea = false;
			break;
		case GT_PLANE:
			geomInfo->handle_collisions = true; // optional
			geomInfo->handle_dynamics = false;
			geomInfo->measure_forces = false;
			geomInfo->fea = false;
			break;
		case GT_TESTPOINTS:
			geomInfo->handle_collisions = false;
			geomInfo->handle_dynamics = false;
			geomInfo->measure_forces = false;
			geomInfo->fea = false;
			break;
		case GT_FREE_SURFACE:
			// free-surface particles for repacking behave like a fixed boundary
			// they are used only if repack mode is on
			geomInfo->handle_collisions = false;
			geomInfo->handle_dynamics = false;
			geomInfo->measure_forces = false;
			geomInfo->enabled = false;
			geomInfo->fea = false;
			break;
		case GT_DEFORMABLE_BODY:
			geomInfo->handle_collisions = false;
			geomInfo->handle_dynamics = false;
			geomInfo->measure_forces = false;
			geomInfo->fea = true;
			break;
		case GT_FEA_RIGID_JOINT:
		case GT_FEA_FLEXIBLE_JOINT:
		case GT_FEA_FORCE:
			geomInfo->handle_collisions = false;
			geomInfo->handle_dynamics = false;
			geomInfo->measure_forces = false;
			geomInfo->fea = false;
			break;
	}

	// --- Default intersection type
	// It is IT_SUBTRACT by default, except for planes: they are usually used
	// to delimit the boundaries of the domain, so we likely want to intersect
	switch (geomInfo->type) {
		case GT_PLANE:
			geomInfo->intersection_type = IT_INTERSECT;
			break;
		case GT_TESTPOINTS:
			geomInfo->intersection_type = IT_NONE;
			break;
		default:
			geomInfo->intersection_type = IT_SUBTRACT;
	}

	// --- Default erase operation
	// Upon intersection or substraction we can choose to interact with fluid
	// or boundaries. By default, water erases only other water, while boundaries
	// erase water and other boundaries. Testpoints eras nothing.
	switch (geomInfo->type) {
		case GT_FLUID:
			geomInfo->erase_operation = ET_ERASE_FLUID;
			break;
		case GT_TESTPOINTS:
			geomInfo->erase_operation = ET_ERASE_NOTHING;
			break;
		default:
			geomInfo->erase_operation = ET_ERASE_ALL;
	}

	if (geomInfo->fea)
		m_numFEAObjects++;

	// NOTE: we don't need to check handle_collisions at all if there are no bodies
	if (geomInfo->handle_dynamics)
		m_numFloatingBodies++;

	if (geomInfo->measure_forces)
		m_numForcesBodies++;

	if (geomInfo->type == GT_PLANE)
		m_numPlanes++;

	if (geomInfo->type == GT_OPENBOUNDARY)
		m_numOpenBoundaries++;

	// For DEMs, check that we don't have an existing DEM already
	if (geomInfo->type == GT_DEM) {
		// FillType UNFILL indicates that this topography will only be used to “carve out”
		// other geometries, and we can have more than one of these
		if (geomInfo->fill_type != FT_UNFILL)
		{
			if (validGeometry(m_dem_geometry))
				throw std::invalid_argument("cannot add a second DEM");
			m_dem_geometry = m_geometries.size();
		}
	}


	m_geometries.push_back(geomInfo);
	return (m_geometries.size() - 1);
}

bool ProblemAPI<1>::validGeometry(GeometryID gid)
{
	// no warning if the gid explicitly refers to the invalid geometry
	if (gid == INVALID_GEOMETRY)
		return false;

	// ensure gid refers to a valid position
	if (gid >= m_geometries.size()) {
		printf("WARNING: invalid GeometryID %zu\n", gid);
		return false;
	}

	// ensure geometry was not deleted
	if (!m_geometries[gid]->enabled) {
		printf("WARNING: GeometryID %zu refers to a deleted geometry!\n", gid);
		return false;
	}

	return true;
}

GeometryID ProblemAPI<1>::addSegment(const GeometryType otype, const FillType ftype, const Point &origin,
	const double length)
{
	double offsetX = 0;
	if (m_positioning == PP_CENTER ||
		(m_positioning == PP_BOTTOM_CENTER && space_dimensions_for(simparams()->dimensions) > 2))
	{
		offsetX = - length / 2.0;
	}

	return addGeometry(otype, ftype,
		make_shared<Segment>( Point( origin(0) + offsetX, origin(1), origin(2) ),
			length, EulerParameters() )
	);
}

GeometryID ProblemAPI<1>::addRect(const GeometryType otype, const FillType ftype, const Point &origin,
	const double side1, const double side2)
{
	double offsetX = 0, offsetY = 0;
	if (m_positioning == PP_CENTER || m_positioning == PP_BOTTOM_CENTER) {
		offsetX = - side1 / 2.0;
		if (m_positioning == PP_CENTER || space_dimensions_for(simparams()->dimensions) > 2)
			offsetY = - side2 / 2.0;
	}

	return addGeometry(otype, ftype,
		make_shared<Rect>( Point( origin(0) + offsetX, origin(1) + offsetY, origin(2) ),
			side1, side2, EulerParameters() )
	);
}

GeometryID ProblemAPI<1>::addDisk(const GeometryType otype, const FillType ftype, const Point &origin,
	const double radius)
{
	double offsetX = 0, offsetY = 0;
	if (m_positioning == PP_CORNER)
		offsetX = offsetY = -radius;
	if ((m_positioning == PP_BOTTOM_CENTER) && (space_dimensions_for(simparams()->dimensions) == 2))
		offsetY = radius;

	return addGeometry(otype, ftype,
		make_shared<Disk>( Point( origin(0) + offsetX, origin(1) + offsetY, origin(2) ),
			radius, EulerParameters() )
	);
}

GeometryID ProblemAPI<1>::addCube(const GeometryType otype, const FillType ftype, const Point &origin, const double side)
{
	double offsetXY = 0, offsetZ = 0;
	if (m_positioning == PP_CENTER || m_positioning == PP_BOTTOM_CENTER)
		offsetXY = - side / 2.0;
	if (m_positioning == PP_CENTER)
		offsetZ = - side / 2.0;

	return addGeometry(otype, ftype,
		make_shared<Cube>( Point( origin(0) + offsetXY, origin(1) + offsetXY, origin(2) + offsetZ ),
			side, side, side, 1, 1, 1, EulerParameters() )
	);
}

GeometryID ProblemAPI<1>::addBox(const GeometryType otype, const FillType ftype, const Point &origin,
			const double side1, const double side2, const double side3)
{
	return addBox(otype, ftype, origin, side1, side2, side3, 1, 1);
}

GeometryID ProblemAPI<1>::addBox(const GeometryType otype, const FillType ftype, const Point &origin,
			//const double side1, const double side2, const double side3, int nelsx, int nelsy, int nelsz) // we are using ANCF shells, then by default we have one element over the thickness
			const double side1, const double side2, const double side3, int nelsx, int nelsy)
{
	int nelsz = 1;
	double offsetX = 0, offsetY = 0, offsetZ = 0;
	if (m_positioning == PP_CENTER || m_positioning == PP_BOTTOM_CENTER) {
		offsetX = - side1 / 2.0;
		offsetY = - side2 / 2.0;
	}
	if (m_positioning == PP_CENTER)
		offsetZ = - side3 / 2.0;

	return addGeometry(otype, ftype,
		make_shared<Cube>( Point( origin(0) + offsetX, origin(1) + offsetY, origin(2) + offsetZ ),
			side1, side2, side3, nelsx, nelsy, nelsz, EulerParameters())
	);
}

GeometryID ProblemAPI<1>::addCylinder(const GeometryType otype, const FillType ftype, const Point &origin,
			const double outer_radius, const double height)
{
	return addCylinder(otype, ftype, origin, outer_radius, outer_radius, height, 1);
}

GeometryID ProblemAPI<1>::addCylinder(const GeometryType otype, const FillType ftype, const Point &origin,
			const double outer_radius, const double inner_radius, const double height,
			//const uint nelst, const uint nelsc, const uint nelsh) we are using ANCF cable, then no elements over the circumference and the thickness
			const uint nelsh) // just elements over height
{
	uint nelsc = 1;
	uint nelst = 1;
	double offsetXY = 0, offsetZ = 0;
	if (m_positioning == PP_CORNER)
		offsetXY = outer_radius;
	else
	if (m_positioning == PP_CENTER)
		offsetZ = - height / 2.0;

	return addGeometry(otype, ftype,
		make_shared<Cylinder>( Point( origin(0) + offsetXY, origin(1) + offsetXY, origin(2) + offsetZ ),
			outer_radius, inner_radius, height, nelst, nelsc, nelsh, EulerParameters() )
	);
}

GeometryID ProblemAPI<1>::addCone(const GeometryType otype, const FillType ftype, const Point &origin,
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
		make_shared<Cone>( Point( origin(0) + offsetXY, origin(1) + offsetXY, origin(2) + offsetZ ),
			bottom_radius, top_radius, height, EulerParameters() )
	);
}

GeometryID ProblemAPI<1>::addSphere(const GeometryType otype, const FillType ftype, const Point &origin,
	const double radius)
{
	double offsetXY = 0, offsetZ = 0;
	if (m_positioning == PP_CORNER || m_positioning == PP_BOTTOM_CENTER)
		offsetZ = radius;
	if (m_positioning == PP_CORNER)
		offsetXY = radius;

	return addGeometry(otype, ftype,
		make_shared<Sphere>( Point( origin(0) + offsetXY, origin(1) + offsetXY, origin(2) + offsetZ ),
			radius )
	);
}

GeometryID ProblemAPI<1>::addTorus(const GeometryType otype, const FillType ftype, const Point &origin,
	const double major_radius, const double minor_radius)
{
	double offsetXY = 0, offsetZ = 0;
	if (m_positioning == PP_CORNER || m_positioning == PP_BOTTOM_CENTER)
		offsetZ = minor_radius;
	if (m_positioning == PP_CORNER)
		offsetXY = (major_radius + minor_radius);

	return addGeometry(otype, ftype,
		make_shared<Torus>( origin, major_radius, minor_radius, EulerParameters() )
	);
}

GeometryID ProblemAPI<1>::addPlane(
	const double a_coeff, const double b_coeff, const double c_coeff, const double d_coeff, const FillType ftype)
{
	return addGeometry(GT_PLANE, ftype,
		make_shared<Plane>( a_coeff, b_coeff, c_coeff, d_coeff )
	);
}

// NOTE: "origin" has a slightly different meaning than for the other primitives: here it is actually
// an offset to shift the STL coordinates. Use 0 to import STL coords as they are.
// If positioning is PP_NONE and origin is (0,0,0), mesh coordinates are imported unaltered.
GeometryID ProblemAPI<1>::addSTLMesh(const GeometryType otype, const FillType ftype, const Point &origin,
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

	return addGeometry(
		otype,
		ftype,
		shared_ptr<STLMesh>(stlmesh),
		NULL,			// HDF5 filename
		NULL,			// XYZ filename
		filename		// STL filename
	);
}

// NOTE: "origin" has a slightly different meaning than for the other primitives: here it is actually
// an offset to shift the STL coordinates. Use 0 to import STL coords as they are.
// If positioning is PP_NONE and origin is (0,0,0), mesh coordinates are imported unaltered.
GeometryID ProblemAPI<1>::addOBJMesh(const GeometryType otype, const FillType ftype, const Point &origin,
	const char *filename)
{
	STLMesh *stlmesh = new STLMesh();
	stlmesh->setObjectFile(filename);
	stlmesh->loadObjBounds(); // update bbox

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

	return addGeometry(
		otype,
		ftype,
		shared_ptr<STLMesh>(stlmesh),
		NULL,			// HDF5 filename
		NULL,			// XYZ filename
		filename		// STL filename
	);
}


// NOTE: particles loaded from HDF5 files will not be erased!
// To enable erase-like interaction we need to copy them to the particle vectors, which
// requires unnecessary memory allocation
GeometryID ProblemAPI<1>::addHDF5File(const GeometryType otype, const Point &origin,
	const char *fname_hdf5, const char *fname_obj)
{
	// NOTES about HDF5 files
	// - fill type is FT_NOFILL since particles are read from file
	// - may add a null STLMesh if the hdf5 file is given but not the mesh
	// - should adding an HDF5 file of type GT_TESTPOINTS be forbidden?

	// will create anyway an empty STLMesh if the STL filename is not given
	STLMesh *stlmesh = new STLMesh();
	if (fname_obj) {
		stlmesh->setObjectFile(fname_obj);
		stlmesh->loadObjBounds(); // update bbox

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
	}

	// NOTE: an empty STL mesh does not return a meaningful bounding box. Will read parts in initialize() for that

	return addGeometry(
		otype,
		FT_NOFILL,
		shared_ptr<STLMesh>(stlmesh),
		fname_hdf5,			// HDF5 filename
		NULL,			// XYZ filename
		fname_obj		// STL filename
	);
}

// NOTE: "origin" has a slightly different meaning than for the other primitives: here it is actually
// an offset to shift the STL coordinates. Use 0 to import STL coords as they are.
// If positioning is PP_NONE and origin is (0,0,0), mesh coordinates are imported unaltered.
GeometryID ProblemAPI<1>::addTetFile(const GeometryType otype, const FillType ftype, const Point &origin,
	const char *nodes_file, const char *elems_file, const double z_frame) //FIXME instead of z_frame define a plane in 3D space
{

	const double corr_z_frame = z_frame - origin(2); // apply correction due to mesh translation
	STLMesh *stlmesh = STLMesh::load_TetFile(nodes_file, elems_file, corr_z_frame);


	double offsetX = 0, offsetY = 0, offsetZ = 0;
/* // FIXME try to understand what this does and in case it is useful for fea mesh fix it, because makes particles disappear
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
*/
	// shift STL origin to given point
	stlmesh->shift( make_double3(origin(0) + offsetX, origin(1) + offsetY, origin(2) + offsetZ) );

	cout << "handled STL mesh position" << endl;
	return addGeometry(
		otype,
		ftype,
		shared_ptr<STLMesh>(stlmesh),
		NULL,			// HDF5 filename
		NULL,			// XYZ filename
		nodes_file		// STL filename
	);
}



// NOTE: particles loaded from XYZ files will not be erased!
// To enable erase-like interaction we need to copy them to the global particle vectors, by passing an
// existing vector to loadPointCloudFromXYZFile(). We can implement this if needed.
GeometryID ProblemAPI<1>::addXYZFile(const GeometryType otype, const Point &origin,
			const char *fname_xyz, const char *fname_obj)
{
	// NOTE: fill type is FT_NOFILL since particles are read from file

	// create an empty STLMesh if a mesh filename is not given
	STLMesh *stlmesh = new STLMesh(0);
	if (fname_obj)
		stlmesh->setObjectFile(fname_obj);

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

	// NOTE: an empty STL mesh does not return a meaningful bounding box. Will read parts for that

	return addGeometry(
		otype,
		FT_NOFILL,
		shared_ptr<STLMesh>(stlmesh),
		NULL,			// HDF5 filename
		fname_xyz,		// XYZ filename
		fname_obj		// STL filename
	);
}

GeometryID
ProblemAPI<1>::addDEM(const char * fname_dem, const TopographyFormat dem_fmt, const FillType fill_type)
{
	// Our TopographyFormat match the TopoCube::Format values, except for DEM_FMT_ASCII_STRICT
	// which corresponds to DEM_FMT_ASCII plus the STRICT option
	TopoCube::Format tc_fmt =
		dem_fmt == DEM_FMT_ASCII_STRICT ?
		TopoCube::Format::DEM_FMT_ASCII :
		TopoCube::Format(dem_fmt);
	TopoCube::FormatOptions tc_fmt_opt =
		dem_fmt == DEM_FMT_ASCII_STRICT ?
		TopoCube::FormatOptions::STRICT :
		TopoCube::FormatOptions::RELAXED;

	TopoCube * dem = TopoCube::load_file(fname_dem, tc_fmt, tc_fmt_opt);
	GeometryID ret = addGeometry(
		GT_DEM,
		fill_type,
		shared_ptr<TopoCube>(dem),
		NULL,
		NULL,
		fname_dem);

	m_geometries[ret]->dem_fmt = dem_fmt;

	return ret;
}

GeometryID
ProblemAPI<1>::addDEMFluidBox(double height, GeometryID dem_gid)
{
	if (!validGeometry(dem_gid)) dem_gid = m_dem_geometry;
	if (!validGeometry(dem_gid))
		throw invalid_argument("invalid DEM geometry ID");

	GeometryInfo *gdem = m_geometries.at(dem_gid);

	if (!gdem || (gdem->type != GT_DEM))
		throw invalid_argument("invalid DEM geometry ID");

	auto dem = static_pointer_cast<TopoCube>(gdem->ptr);

	dem->SetCubeHeight(height);
	dem->SetFillingOffset(physparams()->r0);

	GeometryID ret = addGeometry(
		GT_FLUID,
		FT_SOLID,
		dem,
		NULL, NULL, NULL);

	return ret;
}

// Add a single testpoint; returns the position of the testpoint in the vector of
// testpoints, which will correspond to its particle id.
// NOTE: testpoints should be assigned with consecutive particle ids starting from 0
size_t ProblemAPI<1>::addTestPoint(const Point &coordinates)
{
	m_testpointParts.push_back(coordinates);
	return (m_testpointParts.size() - 1);
}

// Simple overload
size_t ProblemAPI<1>::addTestPoint(const double posx, const double posy, const double posz)
{
	return addTestPoint(Point(posx, posy, posz));
}

// request to invert normals while loading - only for HDF5 files
void ProblemAPI<1>::flipNormals(const GeometryID gid, bool flip)
{
	if (!validGeometry(gid)) return;

	// this makes sense only for geometries loading a HDF5 file
	// TODO: also enable for planes?
	if (!m_geometries[gid]->has_hdf5_file) {
		printf("WARNING: trying to invert normals on a geometry without HDF5-files associated! Ignoring\n");
		return;
	}

	m_geometries[gid]->flip_normals = flip;
}

void ProblemAPI<1>::deleteGeometry(const GeometryID gid)
{
	if (!validGeometry(gid)) return;

	m_geometries[gid]->enabled = false;

	// and this is the reason why m_numActiveGeometries not be used to iterate on m_geometries:
	m_numActiveGeometries--;

	if (m_geometries[gid]->measure_forces)
		m_numForcesBodies--;

	if (m_geometries[gid]->handle_dynamics)
		m_numFloatingBodies--;

	if (m_geometries[gid]->type == GT_PLANE)
		m_numPlanes--;

	if (m_geometries[gid]->type == GT_OPENBOUNDARY)
		m_numOpenBoundaries--;

	if (gid == m_dem_geometry)
		m_dem_geometry = INVALID_GEOMETRY;

	// TODO: print a warning if deletion is requested after fill_parts
}

void ProblemAPI<1>::enableDynamics(const GeometryID gid)
{
	if (!validGeometry(gid)) return;

	// ensure dynamics are consistent with geometry type
	if (m_geometries[gid]->type != GT_FLOATING_BODY &&
		m_geometries[gid]->type != GT_MOVING_BODY) {
		printf("WARNING: dynamics only available for rigid bodies! Ignoring\n");
		return;
	}
	// update counter of rigid bodies if needed
	if (!m_geometries[gid]->handle_dynamics)
		m_numFloatingBodies++;
	m_geometries[gid]->handle_dynamics = true;
}

void ProblemAPI<1>::enableCollisions(const GeometryID gid)
{
	if (!validGeometry(gid)) return;

	// TODO: allow collisions for open boundaries? Why not for fixed bounds?
	// ensure collisions are consistent with geometry type
	if (m_geometries[gid]->type != GT_FLOATING_BODY &&
		m_geometries[gid]->type != GT_MOVING_BODY &&
		m_geometries[gid]->type != GT_PLANE) {
		printf("WARNING: collisions only available for rigid bodies and planes! Ignoring\n");
		return;
	}
	m_geometries[gid]->handle_collisions = true;
}

void ProblemAPI<1>::disableDynamics(const GeometryID gid)
{
	if (!validGeometry(gid)) return;

	// ensure no-dynamics is consistent with geometry type
	if (m_geometries[gid]->type == GT_FLOATING_BODY) {
		printf("WARNING: dynamics are mandatory for floating bodies! Ignoring\n");
		return;
	}
	// update counter of rigid bodies if needed
	if (m_geometries[gid]->handle_dynamics)
		m_numFloatingBodies--;
	m_geometries[gid]->handle_dynamics = false;
}

void ProblemAPI<1>::disableCollisions(const GeometryID gid)
{
	if (!validGeometry(gid)) return;

	// it is possible to disable collisions for any geometry type, so no need to check it
	m_geometries[gid]->handle_collisions = false;
}

void ProblemAPI<1>::enableFeedback(const GeometryID gid)
{
	if (!validGeometry(gid)) return;

	// TODO: allow collisions for open boundaries? Why not for fixed bounds?
	// ensure collisions are consistent with geometry type
	if (m_geometries[gid]->type != GT_FLOATING_BODY &&
		m_geometries[gid]->type != GT_MOVING_BODY) {
		printf("WARNING: feedback only available for floating or moving bodies! Ignoring\n");
		return;
	}

	if (!m_geometries[gid]->measure_forces)
		m_numForcesBodies++;
	m_geometries[gid]->measure_forces = true;
}

void ProblemAPI<1>::disableFeedback(const GeometryID gid)
{
	if (!validGeometry(gid)) return;

	// ensure no-dynamics is consistent with geometry type
	if (m_geometries[gid]->type == GT_FLOATING_BODY) {
		printf("WARNING: feedback is mandatory for floating bodies! Ignoring\n");
		return;
	}

	if (m_geometries[gid]->measure_forces)
		m_numForcesBodies--;
	m_geometries[gid]->measure_forces = false;
}

// Set a custom inertia matrix (main diagonal only). Will overwrite the precomputed one
void ProblemAPI<1>::setInertia(const GeometryID gid, const double i11, const double i22, const double i33)
{
	if (!validGeometry(gid)) return;

	// implicitly checking that geometry is a GT_FLOATING_BODY
	if (!m_geometries[gid]->handle_dynamics) {
		printf("WARNING: trying to set inertia of a geometry with no dynamics! Ignoring\n");
		return;
	}
	m_geometries[gid]->custom_inertia[0] = i11;
	m_geometries[gid]->custom_inertia[1] = i22;
	m_geometries[gid]->custom_inertia[2] = i33;
}

// overload
void ProblemAPI<1>::setInertia(const GeometryID gid, const double* mainDiagonal)
{
	setInertia(gid, mainDiagonal[0], mainDiagonal[1], mainDiagonal[2]);
}

// Set a custom center of gravity. Will overwrite the precomputed one
void ProblemAPI<1>::setCenterOfGravity(const GeometryID gid, const double3 cg)
{
	if (!validGeometry(gid)) return;

	// implicitly checking that geometry is a GT_FLOATING_BODY
	if (!m_geometries[gid]->handle_dynamics) {
		printf("WARNING: trying to set inertia of a geometry with no dynamics! Ignoring\n");
		return;
	}
	m_geometries[gid]->custom_cg.x = cg.x;
	m_geometries[gid]->custom_cg.y = cg.y;
	m_geometries[gid]->custom_cg.z = cg.z;
}

// NOTE: GPUSPH uses ZXZ angles counterclockwise, ODE used XYZ clockwise (http://goo.gl/bV4Zeb - http://goo.gl/oPnMCv)
// We should check what's used by Chrono
void ProblemAPI<1>::setOrientation(const GeometryID gid, const EulerParameters &ep)
{
	if (!validGeometry(gid)) return;

	m_geometries[gid]->ptr->setEulerParameters(ep);
}

// DEPRECATED until we'll have a GPUSPH Quaternion class
void ProblemAPI<1>::rotate(const GeometryID gid, const EulerParameters ep)
{
	if (!validGeometry(gid)) return;

	//m_geometries[gid]->ptr->setEulerParameters(ep * (*(m_geometries[gid]->ptr->getEulerParameters())) );
	m_geometries[gid]->ptr->setEulerParameters(ep * m_geometries[gid]->ptr->getEulerParameters());
}

// NOTE: rotates X first, then Y, then Z
void ProblemAPI<1>::rotate(const GeometryID gid, const double Xrot, const double Yrot, const double Zrot)
{
	if (!validGeometry(gid)) return;

	// compute single-axes rotations
	double4 rotX, rotY, rotZ;
	// TODO: check if Chrono uses clockwise (like ODE did) or counterclockwise angles for Euler
	// NOTE: with ODE we computed the quaternions ourselves for accuracy reasons:
	// for each rotation, the real part of the quaternion is cos(angle/2),
	// and the imaginary part (which for rotations around principal axis
	// is only 1, 2 or 3) is sin(angle/2); the rest of the components are 0.
	// TODO: check if we need it with Chrono as well
	rotX.x = cos(-Xrot/2); rotX.y = sin(-Xrot/2); rotX.z = rotX.w = 0;
	rotY.x = cos(-Yrot/2); rotY.z = sin(-Yrot/2); rotY.y = rotY.w = 0;
	rotZ.x = cos(-Zrot/2); rotZ.w = sin(-Zrot/2); rotZ.y = rotZ.z = 0;
	// Problem: even with a “nice” angle such as M_PI we might end up
	// with not-exactly-zero components, so we kill anything which is less
	// than half the double-precision machine epsilon. If you REALLY care
	// about angles that differ from quadrant angles by less than 2^-53,
	// sorry, we don't have enough accuracy for you.
	if (fabs(rotX.x) < DBL_EPSILON/2)
		rotX.x = 0;
	if (fabs(rotX.y) < DBL_EPSILON/2)
		rotX.y = 0;
	if (fabs(rotY.x) < DBL_EPSILON/2)
		rotY.x = 0;
	if (fabs(rotY.z) < DBL_EPSILON/2)
		rotY.z = 0;
	if (fabs(rotZ.x) < DBL_EPSILON/2)
		rotZ.x = 0;
	if (fabs(rotZ.w) < DBL_EPSILON/2)
		rotZ.w = 0;

	// concatenate rotations in order (X, Y, Z)
	EulerParameters rotXY = EulerParameters(rotY.x, rotY.y, rotY.z, rotY.w) * EulerParameters(rotX.x, rotX.y, rotX.z, rotX.w);
	EulerParameters rotXYZ = EulerParameters(rotZ.x, rotZ.y, rotZ.z, rotZ.w) * rotXY;

	// rotate with computed quaternion
	rotate( gid, rotXYZ );
}

void ProblemAPI<1>::shift(const GeometryID gid, const double Xoffset, const double Yoffset, const double Zoffset)
{
	if (!validGeometry(gid)) return;

	if (m_geometries[gid]->type == GT_PLANE) {
		printf("WARNING: shift is not available for planes! Ignoring\n");
		return;
	}

	m_geometries[gid]->ptr->shift(make_double3(Xoffset, Yoffset, Zoffset));
}

void ProblemAPI<1>::setIntersectionType(const GeometryID gid, IntersectionType i_type)
{
	if (!validGeometry(gid)) return;

	m_geometries[gid]->intersection_type = i_type;
}

void ProblemAPI<1>::setEraseOperation(const GeometryID gid, EraseOperation e_operation)
{
	if (!validGeometry(gid)) return;

	m_geometries[gid]->erase_operation = e_operation;
}

void ProblemAPI<1>::setMass(const GeometryID gid, const double mass)
{
	if (!validGeometry(gid)) return;

	if (m_geometries[gid]->type != GT_FLOATING_BODY)
		printf("WARNING: setting mass of a non-floating body\n");
	m_geometries[gid]->ptr->SetMass(mass);
	m_geometries[gid]->mass_was_set = true;
}

void ProblemAPI<1>::setYoungModulus(const GeometryID gid, const double youngModulus)
{
	if (!validGeometry(gid)) return;

	if (m_geometries[gid]->type != GT_DEFORMABLE_BODY)
		printf("WARNING: setting Young's modulus of a non-deformable body\n");
	m_geometries[gid]->ptr->SetYoungModulus(youngModulus);
}

void ProblemAPI<1>::setDynamometer(const GeometryID gid, const bool isDynamometer)
{
	if (!validGeometry(gid)) return;
	if (m_geometries[gid]->type != GT_FEA_RIGID_JOINT)
		printf("WARNING: Only GT_FEA_RIGID_JOINT bodies can be set Dynamometer\n");
	m_geometries[gid]->is_dynamometer = isDynamometer;
}

void ProblemAPI<1>::setAlphaDamping(const GeometryID gid, const double alphaDamping)
{
	if (!validGeometry(gid)) return;

	if (m_geometries[gid]->type != GT_DEFORMABLE_BODY)
		printf("WARNING: setting Alpha damping of a non-deformable body\n");
	m_geometries[gid]->ptr->SetAlphaDamping(alphaDamping);
}

void ProblemAPI<1>::setPoissonRatio(const GeometryID gid, const double poissonRatio)
{
	if (!validGeometry(gid)) return;

	if (m_geometries[gid]->type != GT_DEFORMABLE_BODY)
		printf("WARNING: setting Poisson ratio of a non-deformable body\n");
	m_geometries[gid]->ptr->SetPoissonRatio(poissonRatio);
}

void ProblemAPI<1>::setDensity(const GeometryID gid, const double density)
{
	if (!validGeometry(gid)) return;

	if (m_geometries[gid]->type != GT_DEFORMABLE_BODY)
		printf("WARNING: setting density of a non-deformable body\n");
	m_geometries[gid]->ptr->SetDensity(density);
}

double ProblemAPI<1>::setMassByDensity(const GeometryID gid, const double density)
{
	if (!validGeometry(gid)) return NAN;

	auto geom = m_geometries[gid];
	if (geom->type != GT_FLOATING_BODY)
		printf("WARNING: setting mass of a non-floating body\n");

	const double volume_add_dx = Object::get_default_filling_method() == Object::BORDER_CENTERED ?
		preferredDeltaP(geom->type) : 0;
	const double mass = geom->ptr->SetMass(volume_add_dx, density);
	geom->mass_was_set = true;

	return mass;
}

void ProblemAPI<1>::setParticleMass(const GeometryID gid, const double mass)
{
	if (!validGeometry(gid)) return;

	// TODO: if SA bounds && geom is not fluid, throw exception or print warning

	m_geometries[gid]->ptr->SetPartMass(mass);
	m_geometries[gid]->particle_mass_was_set = true;
}

double ProblemAPI<1>::setParticleMassByDensity(const GeometryID gid, const double density)
{
	if (!validGeometry(gid)) return NAN;

	if ( (m_geometries[gid]->has_xyz_file || m_geometries[gid]->has_hdf5_file) &&
		 !m_geometries[gid]->has_mesh_file)
		printf("WARNING: setting the mass by density can't work with a point-based geometry without a mesh!\n");

	const double dx = preferredDeltaP(m_geometries[gid]->type);
	const double particle_mass = m_geometries[gid]->ptr->SetPartMass(dx, density);
	m_geometries[gid]->particle_mass_was_set = true;

	return particle_mass;
}

// Flag an open boundary as velocity driven, so its particles will be flagged with
// as VEL_IO during fill. Use with false to revert to pressure driven.
// Only makes sense with GT_OPENBOUNDARY geometries.
void ProblemAPI<1>::setVelocityDriven(const GeometryID gid, bool isVelocityDriven)
{
	if (!validGeometry(gid)) return;

	if (m_geometries[gid]->type != GT_OPENBOUNDARY) {
		printf("WARNING: trying to set as velocity driven a non-GT_OPENBOUNDARY geometry! Ignoring\n");
		return;
	}

	m_geometries[gid]->velocity_driven = isVelocityDriven;
}

// Set custom radius for unfill operations. NAN means: use dp
void ProblemAPI<1>::setUnfillRadius(const GeometryID gid, double unfillRadius)
{
	if (!validGeometry(gid)) return;

	m_geometries[gid]->unfill_radius = unfillRadius;
}

const GeometryInfo* ProblemAPI<1>::getGeometryInfo(GeometryID gid) const
{
	// NOTE: not checking validGeometry() to allow for deleted geometries

	// ensure gid refers to a valid position
	if (gid >= m_geometries.size()) {
		printf("WARNING: invalid GeometryID %zu\n", gid);
		return NULL;
	}

	// return geometry even if deleted
	return m_geometries[gid];
}

const ObjectPtr ProblemAPI<1>::getGeometryObject(GeometryID gid) const
{
	const GeometryInfo *gi = getGeometryInfo(gid);
	if (gi) return gi->ptr;
	throw runtime_error("Requested objected for null geometry");
}

ObjectPtr ProblemAPI<1>::getGeometryObject(GeometryID gid)
{
	if (gid >=m_geometries.size())
		throw runtime_error("Requested objected for null geometry");
	return m_geometries[gid]->ptr;
}

// set the positioning policy for geometries added after the call
void ProblemAPI<1>::setPositioning(PositioningPolicy positioning)
{
	m_positioning = positioning;
}

// Create 6 planes delimiting the box defined by the two points and update (overwrite) the world origin and size.
// Write their GeometryIDs in planesIds, if given, so that it is possible to delete one or more of them afterwards.
vector<GeometryID> ProblemAPI<1>::makeUniverseBox(double3 const& corner1, double3 const& corner2)
{
	vector<GeometryID> planes;

	// compute min and max
	double3 min, max;
	min.x = fmin(corner1.x, corner2.x);
	min.y = fmin(corner1.y, corner2.y);
	min.z = fmin(corner1.z, corner2.z);
	max.x = fmax(corner1.x, corner2.x);
	max.y = fmax(corner1.y, corner2.y);
	max.z = fmax(corner1.z, corner2.z);

	// we need the periodicity to see which planes are needed. If simparams() is NULL,
	// it means SETUP_FRAMEWORK was not invoked, in which case we assume no periodicity.
	const Periodicity periodicbound = simparams() ? simparams()->periodicbound : PERIODIC_NONE;

	// create planes
	if (!(periodicbound & PERIODIC_X)) {
		planes.push_back(addPlane(  1,  0,  0, -min.x));
		planes.push_back(addPlane( -1,  0,  0,  max.x));
	}
	if (!(periodicbound & PERIODIC_Y)) {
		planes.push_back(addPlane(  0,  1,  0, -min.y));
		planes.push_back(addPlane(  0, -1,  0,  max.y));
	}
	if (!(periodicbound & PERIODIC_Z)) {
		planes.push_back(addPlane(  0,  0,  1, -min.z));
		planes.push_back(addPlane(  0,  0, -1,  max.z));
	}

	// set world origin and size
	m_origin = min;
	m_size = max - min;

	return planes;
}

std::vector<GeometryID>
ProblemAPI<1>::makeUniverseBox(Point const& corner1, Point const& corner2)
{
	return makeUniverseBox( make_double3(corner1), make_double3(corner2) );
}

vector<GeometryID> ProblemAPI<1>::addDEMPlanes(GeometryID gid)
{
	if (!validGeometry(gid)) gid = m_dem_geometry;
	FillType ftype = gid < m_geometries.size() ? m_geometries[gid]->fill_type : FT_NOFILL;
	return addDEMPlanes(gid, ftype);
}

vector<GeometryID> ProblemAPI<1>::addDEMPlanes(GeometryID gid, FillType ftype)
{
	vector<GeometryID> planes;
	planes.reserve(4);
	if (!validGeometry(gid)) gid = m_dem_geometry;
	const GeometryInfo *gi = getGeometryInfo(gid);
	if (!gi)
		throw std::invalid_argument("no DEM to add planes from");
	if (gi->type != GT_DEM)
		throw std::invalid_argument("adding planes from a geometry that is not a DEM");
	vector<double4> coeffs = static_pointer_cast<const TopoCube>(gi->ptr)->get_planes();
	for (auto c : coeffs) {
		planes.push_back( addPlane(c.x, c.y, c.z, c.w, ftype) );
	}
	return planes;
}

void ProblemAPI<1>::addExtraWorldMargin(const double margin)
{
	if (margin >= 0.0)
		m_extra_world_margin = margin;
	else
		printf("WARNING: tried to add negative world margin! Ignoring\n");
}

void ProblemAPI<1>::computeDEMphysparams()
{
	if (!validGeometry(m_dem_geometry))
		throw std::runtime_error("cannot compute DEM physical parameters without a valid DEM");

	auto dem = static_pointer_cast<TopoCube>(m_geometries[m_dem_geometry]->ptr);

	set_dem(dem->get_dem(), dem->get_ncols(), dem->get_nrows());

	PhysParams *pp = physparams();

	float ewres = dem->get_ewres();
	float nsres = dem->get_nsres();
	pp->ewres = ewres;
	pp->nsres = nsres;
	if (isfinite(m_dem_dx_scale)) pp->demdx = ewres/m_dem_dx_scale;
	if (isfinite(m_dem_dy_scale)) pp->demdy = nsres/m_dem_dy_scale;
	pp->demdxdy = pp->demdx*pp->demdy;
	if (isfinite(m_dem_zmin_scale)) pp->demzmin = m_dem_zmin_scale*m_deltap;
}

void ProblemAPI<1>::setDEMZminScale(double scale)
{
	m_dem_zmin_scale = scale;
	computeDEMphysparams();
}

void ProblemAPI<1>::setDEMZmin(double demzmin)
{
	m_dem_zmin_scale = NAN; // disable automatic computation
	physparams()->demzmin = demzmin;
}

void ProblemAPI<1>::setDEMNormalDisplacementScale(double scalex, double scaley)
{
	if (!isfinite(scaley)) scaley = scalex;
	m_dem_dx_scale = scalex;
	m_dem_dy_scale = scaley;
	computeDEMphysparams();
}

void ProblemAPI<1>::setDEMNormalDisplacement(double demdx, double demdy)
{
	if (!isfinite(demdy)) demdy = demdx;
	m_dem_dx_scale = NAN;
	m_dem_dy_scale = NAN;
	physparams()->demdx = demdx;
	physparams()->demdy = demdy;
}


// set number of layers for dynamic boundaries. Default is 0, which means: autocompute
void ProblemAPI<1>::setNumBoundaryLayers(const uint numLayers)
{
	if (m_numBoundLayers != 0 && numLayers != m_numBoundLayers)
		printf("WARNING: resetting number of layers");

	const uint suggestedNumLayers = suggestedNumBoundaryLayers();
	if (numLayers > 0 && numLayers < suggestedNumLayers)
		printf("WARNING: number of layers for boundaries is low (%u), suggested number is %u\n",
			numLayers, suggestedNumLayers);

	m_numBoundLayers = numLayers;
}


#if USE_CHRONO == 1

/*********** Utility functions for GT_FEA_DEFORMABLE_JOINT ******************/
// function to sort nodes according to the distance from the center of the joint
bool compareDistance(const feaNodeInfo d1, const feaNodeInfo d2)
{
	return (d1.dist > d2.dist);
}

// function to determine which triangle a node should be associated to
bool is_above(::chrono::ChVector<> p1, ::chrono::ChVector<> p2, ::chrono::ChVector<> point)
{
	/*difference between the point y coord and the y value obtained from the point x coordinate, using the 
	 * equation of the line passing by the two reference points*/

	double y_diff = point.y() - (p1.y() + (p1.y() - p2.y())/(p1.x() - p2.x())*(point.x() - p1.x()));

	return y_diff > 0;
}
/******************************************************************************/

#endif


uint ProblemAPI<1>::getNumBoundaryLayers()
{
	if (m_numBoundLayers == 0) {
		m_numBoundLayers = suggestedNumBoundaryLayers();
		printf("Number of dynamic boundary layers not set, autocomputed: %u\n", m_numBoundLayers);
	}
	return m_numBoundLayers;
}

int ProblemAPI<1>::fill_parts(bool fill)
{
	// if for debug reason we need to test the position and verse of a plane, we can ask ODE to
	// compute the distance of a probe point from a plane (positive if penetrated, negative out)
	// TODO FIXME chronomerge
	/*
	double3 probe_point = make_double3 (0.5, 0.5, 0.5);
	printf("Test: probe point is distant %g from the bottom plane.\n,
		dGeomPlanePointDepth(m_box_planes[0], probe_point.x, probe_point.y, probe_point.z)
	*/

	cout << "running fill_parts " << endl;
	//uint particleCounter = 0;
	uint bodies_parts_counter = 0;
	uint hdf5file_parts_counter = 0;
	uint xyzfile_parts_counter = 0;
	uint tetfile_parts_counter = 0;
	cout << "Update " << tetfile_parts_counter << endl;

	for (size_t g = 0, num_geoms = m_geometries.size(); g < num_geoms; g++) {
		PointVect* parts_vector = NULL;
		double dx = 0.0;

		// ignore deleted geometries
		if (!m_geometries[g]->enabled) continue;

		// set dx and recipient vector according to geometry type
		dx = preferredDeltaP(m_geometries[g]->type);
		switch (m_geometries[g]->type) {
			case GT_FLUID:
				parts_vector = &m_fluidParts;
				break;
			case GT_TESTPOINTS:
			case GT_FEA_RIGID_JOINT: // TODO FIXME for joints use moving particles or remove particles at all
			case GT_FEA_FLEXIBLE_JOINT:
			case GT_FEA_FORCE:
				parts_vector = &m_testpointParts;
				break;
			case GT_DEFORMABLE_BODY:
			case GT_FLOATING_BODY:
			case GT_MOVING_BODY:
				parts_vector = &(m_geometries[g]->ptr->GetParts());
				break;
			case GT_DEM:
				if (g == m_dem_geometry)
					computeDEMphysparams();
				/* fallthrough */
			default:
				parts_vector = &m_boundaryParts;
		}

		// Now will set the particle and object mass if still unset
		const double DEFAULT_DENSITY = atrest_density(0);
		const double DEFAULT_PHYSICAL_DENSITY = physparams()->numFluids() > 1 ?
			1 : physical_density(DEFAULT_DENSITY, 0);

		// Setting particle mass by means of dx and default density only. This leads to same mass
		// everywhere but possibly slightly different densities. (set the define to 1 to test)
#define SET_PARTICLE_MASS_DIRECTLY 0
#if SET_PARTICLE_MASS_DIRECTLY
		const double DEFAULT_PARTICLE_MASS = (dx * dx * dx) * DEFAULT_PHYSICAL_DENSITY;
#endif


		// Set part mass, if not set already.
		if (m_geometries[g]->type != GT_PLANE && !m_geometries[g]->particle_mass_was_set) {
			// TODO: should the following be an option?
			// (most likely not, especially with the multi-fluid support planned for APIv2
#if SET_PARTICLE_MASS_DIRECTLY
			setParticleMass(g, DEFAULT_PARTICLE_MASS);
#else
			setParticleMassByDensity(g, DEFAULT_PHYSICAL_DENSITY);
#endif
		}

		// Set object mass for floating objects, if not set already
		if (m_geometries[g]->type == GT_FLOATING_BODY && !m_geometries[g]->mass_was_set)
			setMassByDensity(g, DEFAULT_PHYSICAL_DENSITY);

		// prepare for erase operations
		bool del_fluid = (m_geometries[g]->erase_operation == ET_ERASE_FLUID);
		bool del_bound = (m_geometries[g]->erase_operation == ET_ERASE_BOUNDARY);
		if (m_geometries[g]->erase_operation == ET_ERASE_ALL) del_fluid = del_bound = true;

		double unfill_dx = Object::get_default_filling_method() == Object::BORDER_CENTERED
			? dx // or, dp also if (r0!=dp)?
			: 0;
		if (!std::isnan(m_geometries[g]->unfill_radius))
			unfill_dx = m_geometries[g]->unfill_radius;
		// erase operations with existent geometries
		if (del_fluid) {
			if (m_geometries[g]->intersection_type == IT_SUBTRACT)
				m_geometries[g]->ptr->Unfill(m_fluidParts, unfill_dx);
			else
				m_geometries[g]->ptr->Intersect(m_fluidParts, unfill_dx);
		}
		if (del_bound) {
			if (m_geometries[g]->intersection_type == IT_SUBTRACT)
				m_geometries[g]->ptr->Unfill(m_boundaryParts, unfill_dx);
			else
				m_geometries[g]->ptr->Intersect(m_boundaryParts, unfill_dx);
		}

		if (m_geometries[g]->fill_type == FT_UNFILL)
			continue;

		// after making some space, fill
		if (fill) {
			// the only different between inner and outer fills is the direction of the FillIn
			// in the multi-layer boundary case
			uint fill_in_sign = 1;
			switch (m_geometries[g]->fill_type) {
				case FT_OUTER_BORDER:
					fill_in_sign =-1 ;
					/* fallthrough */
				case FT_INNER_BORDER:
					m_geometries[g]->ptr->FillIn(*parts_vector, dx, fill_in_sign*m_numBoundLayers);
					break;
				case FT_SOLID:
					m_geometries[g]->ptr->Fill(*parts_vector, dx);
					break;
				// case FT_NOFILL: ;
				// yes, it is legal to have no "default:": ISO/IEC 9899:1999, section 6.8.4.2
			}
		}

		// floating and moving bodies fill in their local point vector; let's increase
		// the dedicated bodies_parts_counter
		if (m_geometries[g]->type == GT_FLOATING_BODY ||
			m_geometries[g]->type == GT_MOVING_BODY) {
			bodies_parts_counter += m_geometries[g]->ptr->GetParts().size();
		}

		// geometries loaded from HDF5file do not undergo filling, but should be counted as well
		if (m_geometries[g]->has_hdf5_file)
			hdf5file_parts_counter += m_geometries[g]->hdf5_reader->getNParts();
		// ditto for XYZ files
		if (m_geometries[g]->has_xyz_file)
			xyzfile_parts_counter += m_geometries[g]->xyz_reader->getNParts();

		if (m_geometries[g]->type == GT_DEFORMABLE_BODY) {// FIXME make a has_tet file? 
			cout << "adding " << m_geometries[g]->ptr->GetNumParts() << endl;
			tetfile_parts_counter += m_geometries[g]->ptr->GetNumParts();
			cout << "update " << tetfile_parts_counter << endl;
		}



#if 0
		// dbg: fill horizontal XY planes with particles, only within the world domain
		if (m_geometries[g]->type == GT_PLANE) {
			Plane *plane = (Plane*)(m_geometries[g]->ptr);
			// only XY planes planes
			if (! (plane->getA() == 0 && plane->getB() == 0) )
				continue;
			// fill will print a warning
			// NOTE: since parts are added to m_boundaryParts, setting part mass is probably pointless
			plane->SetPartMass(dx, physparams()->rho0[0]);
			// will round r0 to fit each dimension
			const uint xpn = (uint) trunc(m_size.x / physparams()->r0 + 0.5);
			const uint ypn = (uint) trunc(m_size.y / physparams()->r0 + 0.5);
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

#if USE_CHRONO == 1
		/* We need to create a Chrono body if
		 * - There is at least one floating body in the simulation, AND
		 * - Body is not a plane (they work on fluids only, we need big boxes in place of planes) AND
		 * - One of the following holds:
		 *   * Body is FLOATING; then handle_collisions is passed to Chrono (handle_dynamics must be true)
		 *   * Body is MOVING and handle_collisions is true (handle_dynamics must be false);
		 *   * Body is FIXED and handle_collisions is true (handle_dynamics must be false).
		 */
		if ( (m_numFloatingBodies) &&
			 (m_geometries[g]->type != GT_PLANE) &&
			 (m_geometries[g]->handle_dynamics || m_geometries[g]->handle_collisions) ) {

			// Overwrite the computed inertia matrix if user set a custom one
			// NOTE: this must be done before body creation!
			// NOTE: this is needed also if only handle_collision holds

			// Use custom inertia only if entirely finite (no partial overwrite)
			const double i11 = m_geometries[g]->custom_inertia[0];
			const double i22 = m_geometries[g]->custom_inertia[1];
			const double i33 = m_geometries[g]->custom_inertia[2];
			if (isfinite(i11) && isfinite(i22) && isfinite(i33)) {
				m_geometries[g]->ptr->SetInertia(i11, i22, i33);
			} else {
				// if no custom inertia has been set, call default Object::SetInertia()
				// NOTE: this overwrites default inertia set by Chrono for "easy" bodies
				// (Box, Sphere, Cylinder), but it would be only necessary for non-primitive
				// geometries (e.g. STL meshes)
				const double inertia_add_dx = Object::get_default_filling_method() == Object::BORDER_CENTERED ?
					physparams()->r0 : 0;
				m_geometries[g]->ptr->SetInertia(inertia_add_dx);
			}

			// Use custom center of gravity only if entirely finite (no partial overwrite)
			double cg[3];
			cg[0] = m_geometries[g]->custom_cg.x;
			cg[1] = m_geometries[g]->custom_cg.y;
			cg[2] = m_geometries[g]->custom_cg.z;
			if (isfinite(cg[0]) && isfinite(cg[1]) && isfinite(cg[2]))
				m_geometries[g]->ptr->SetCenterOfGravity(cg);


			// Fix the geometry *before the creation of the Chrono body* if not moving nor floating.
			// TODO: check if it holds for moving
			if (m_geometries[g]->type == GT_FIXED_BOUNDARY)
				m_geometries[g]->ptr->SetFixed();
			// NOTE: could use SetNoSpeedNoAcceleration() for MOVING chrono bodies?
			const double body_add_dx = Object::get_default_filling_method() == Object::BORDER_CENTERED ?
				m_deltap : 0;
			m_geometries[g]->ptr->BodyCreate(m_chrono_system, m_deltap, m_geometries[g]->handle_collisions);

			// recap object info such as bounding box, mass, inertia matrix, etc.
			// NOTE: BodyPrintInformation() would be meaningless on planes (excluded above) but harmless anyway
			m_geometries[g]->ptr->BodyPrintInformation();
		} // if m_numFloatingBodies > 0

		if ( (m_numFEAObjects) && m_geometries[g]->fea) {
			m_geometries[g]->ptr->CreateFemMesh(m_chrono_system);
			groundFeaNodes(m_geometries[g]->ptr->GetFeaMesh());
			m_chrono_system->Add(m_geometries[g]->ptr->GetFeaMesh()); //TODO FIXME this should be done inside the CreateFemMesh and inside the load_tet for STLMesh
		}

		// geometry to confine a region where to apply force to FEA nodes
		if ( m_geometries[g]->type == GT_FEA_FORCE) {

			/* Here we fill a vector of booleans relative to all the FEA nodes in the system.
			 * The boolean will be true if we want to apply the force to the corresponding node.*/
			uint nodes_with_force = 0; // number of nodes the force will be applied to

			// The indexing in the vector is global. Since geometries with associated FEA object are not 
			// necessarily defined in sequence we need to take trace of the current number of visited nodes
			uint num_prev_nodes = 0; // current number of visited nodes

			for (size_t k = 0, num_geoms = m_geometries.size(); k < num_geoms; k++) {
				if (m_geometries[k]->ptr->HasFeaMesh()){

					std::shared_ptr<::chrono::fea::ChMesh> fea_mesh = m_geometries[k]->ptr->GetFeaMesh();

					nodes_with_force += m_geometries[g]->ptr->findForceNodes(
						fea_mesh,
						m_deltap,
						num_prev_nodes,
						gdata->s_hFeaExtForce);

					num_prev_nodes += fea_mesh->GetNnodes();
				}
			}

			// The force is distributed over all the nodes contained in the geometry.
			// We need to save the number of nodes we are applying the force to in order
			// to compute the force per node.
			simparams()->numForceNodes = nodes_with_force;
		}


		//TODO we could do a GT_FEA_SELECTION that selects a bunch of nodes, and then we decide what to
		// do on these nodes: e.g. apply force, print, ground, etc.. so we avoid having too many types

		// geometry to confine a region where nodes are written to file
		if ( m_geometries[g]->type == GT_FEA_WRITE) {

			/* Here we fill a vector of booleans relative to all the FEA nodes in the system.
			 * The boolean will be true if we want to write the position of the corresponding node.*/
			uint nodes_to_write = 0; // number of nodes to write

			// The indexing in the vector is global. Since geometries with associated FEA object are not 
			// necessarily defined in sequence we need to take trace of the current number of visited nodes
			uint num_prev_nodes = 0; // current number of visited nodes

			for (size_t k = 0, num_geoms = m_geometries.size(); k < num_geoms; k++) {
				if (m_geometries[k]->ptr->HasFeaMesh()){

					std::shared_ptr<::chrono::fea::ChMesh> fea_mesh = m_geometries[k]->ptr->GetFeaMesh();

					nodes_to_write += m_geometries[g]->ptr->findNodesToWrite(
						fea_mesh,
						m_deltap,
						num_prev_nodes,
						m_WriteFeaNodesIndices,
						m_WriteFeaNodesPointers);

					num_prev_nodes += fea_mesh->GetNnodes();
				}
			}

			// The force is distributed over all the nodes contained in the geometry.
			// We need to save the number of nodes we are applying the force to in order
			// to compute the force per node.
			simparams()->numNodesToWrite = nodes_to_write;
		}

		/*Add a rigid body and attach all the nodes to the rigid body.
		As a rigid body, this kind of joint can have pjysical properties associated,
		for example mass and inertia tensor*/
		if ( m_geometries[g]->type == GT_FEA_RIGID_JOINT) {
			const double body_add_dx = Object::get_default_filling_method() == Object::BORDER_CENTERED ?
				m_deltap : 0;
			m_geometries[g]->ptr->BodyCreate(m_chrono_system, body_add_dx, false);

			uint nodes_in_truss = 0;

			for (size_t k = 0, num_geoms = m_geometries.size(); k < num_geoms; k++) {
				if (m_geometries[k]->ptr->HasFeaMesh()){

					nodes_in_truss += m_geometries[g]->ptr->JoinFeaNodes( m_chrono_system, m_geometries[k]->ptr->GetFeaMesh(), m_deltap);
				}
			}

			if (!nodes_in_truss) throw std::runtime_error("Error: Adding a FEA Joint with no intersecting nodes");

			if (m_geometries[g]->is_dynamometer == true) {
				m_geometries[g]->ptr->makeDynamometer(m_chrono_system,
					m_WriteFeaPointConstrPointers,
					m_WriteFeaDirConstrPointers);
				simparams()->numConstraintsToWrite += 1;
			}
		}

		/* Join elements by means of ChLink
		 * this kind of joint is obtained as a constrain among nodes, then misses the
		 * physical properties that can be associated to a GT_FEA_RIGID_JOINT
		 * but is designed to have minor impact on the simulation performances*/
		if ( m_geometries[g]->type == GT_FEA_FLEXIBLE_JOINT) {


			std::vector<feaNodeInfo> included_nodes;

			uint nodes_in_truss = 0;

			for (size_t k = 0, num_geoms = m_geometries.size(); k < num_geoms; k++) {
				if (m_geometries[k]->ptr->HasFeaMesh()){

					nodes_in_truss += m_geometries[g]->ptr->findNodesToJoin(m_geometries[k]->ptr->GetFeaMesh(), m_deltap, included_nodes);
				}
			}

			if (!nodes_in_truss) throw std::runtime_error("Error: Adding a FEA Joint with no intersecting nodes");

			/* Now we create links among nodes. The four furthest nodes from the center of the geometry
			 * will be used as vertices of the triangular surfaces. This should help guaranteeing that
			 * all the other nodes will be inside the covered area but FIXME it is not guaranteed.
			 */

			std::sort(included_nodes.begin(), included_nodes.end(), compareDistance);


			included_nodes.resize(nodes_in_truss);


			/*For the time being the geometry of the Joint will be limited to a box.
			 * the box will be covered by two triangular areas, associated to the four furthest nodes.
			 * from the latter nodes we fix a spatial limit (a plane) that separates the nodes that
			 * will be assigned respectively to the two triangles*/


			/* We need to nodes to track the diagonal of the Joint, that will be the separation of the two triangles
			 * let us consider the first of the nodes in the sorted vector. The second node used to track the diagonal
			 * separating the two triangles is the furthest form the first.
			 */

			feaNodeInfo first_node = included_nodes[0];
			::chrono::ChVector<> ref_pos = first_node.node->GetPos();
			std::vector<feaNodeInfo> ref_nodes;
			Point dist_from_ref;

			for (uint i = 1; i < 4; i++) {

				feaNodeInfo ref_node;

				dist_from_ref(0) = included_nodes[i].node->GetPos().x() - ref_pos.x();
				dist_from_ref(1) = included_nodes[i].node->GetPos().y() - ref_pos.y();
				dist_from_ref(2) = included_nodes[i].node->GetPos().z() - ref_pos.z();

				//TODO use a function to compute modulus
				//ref_node.dist = Dist(dist_from_ref);
				ref_node.dist = sqrt(dist_from_ref(0)*dist_from_ref(0) +
					dist_from_ref(1)*dist_from_ref(1) +
					dist_from_ref(2)*dist_from_ref(2));


				ref_node.node = included_nodes[i].node;

				ref_nodes.push_back(ref_node);
			}

			std::sort(ref_nodes.begin(), ref_nodes.end(), compareDistance);

			ref_nodes.resize(3);
			std::cout << "reference nodes: " << std::endl;
			for (uint s = 0; s < 3; s++) {
				std::cout << ref_nodes[s].node->GetIndex() << std::endl;
			}

			// let us see which of the remaining nodes is above the line
			uint above = is_above(first_node.node->GetPos(), ref_nodes[0].node->GetPos(), ref_nodes[1].node->GetPos()) ? 1 : 2;
			uint below = is_above(first_node.node->GetPos(), ref_nodes[0].node->GetPos(), ref_nodes[1].node->GetPos()) ? 2 : 1;
			std::cout << "above is : " << ref_nodes[above].node->GetIndex()<< std::endl;


			/*TODO FIXME now we consider the case of triangles lying in horizontal planes, then we consider a
			 * line between the two reference nodes. To estend the concept use a plane containing the first two
			 * reference nodes and orthogonal to the rect passing from the other two reference nodes*/

			for (uint i = 4; i < nodes_in_truss; i++) {

				uint j = is_above(first_node.node->GetPos(), ref_nodes[0].node->GetPos(), included_nodes[i].node->GetPos()) ? above : below;
				auto constraint = std::make_shared<::chrono::fea::ChLinkPointTriface>();
				constraint->Initialize(included_nodes[i].node,
					first_node.node, //first node
					ref_nodes[0].node, // second node
					ref_nodes[j].node);
				m_chrono_system->Add(constraint);
				std::cout << "linked " << included_nodes[i].node->GetIndex() << " to " << first_node.node->GetIndex()<< " " << ref_nodes[0].node->GetIndex() << " " << ref_nodes[j].node->GetIndex() << std::endl ;
			}


		}


#endif

		// tell Problem to add the proper type of body
		// FIXME ProblemCore should be using shared_ptrs too
		// TODO when that's done we won't need the .get()s when
		// passing the Object ptr to add_moving_body
		if (m_geometries[g]->type == GT_FLOATING_BODY)
			add_moving_body(m_geometries[g]->ptr.get(), MB_FLOATING);
		else
		if (m_geometries[g]->type == GT_MOVING_BODY) {
			if (m_geometries[g]->measure_forces)
				add_moving_body(m_geometries[g]->ptr.get(), MB_FORCES_MOVING);
			else
				add_moving_body(m_geometries[g]->ptr.get(), MB_MOVING);
		}

		if (m_geometries[g]->type == GT_DEFORMABLE_BODY) {// FIXME make a has_tet file? 
			add_fea_body(m_geometries[g]->ptr.get()); // TODO use shared_ptrs also in FeaBodyData
			tetfile_parts_counter += m_geometries[g]->ptr->GetNumFeaNodes();
			cout << "Adding " << m_geometries[g]->ptr->GetNumFeaNodes() << endl;
			cout << "Update " << tetfile_parts_counter << endl;


			// the following code was creating a link between two nodes sharig the coordinates. In the newest version
			// if several nodes would share the coordinate we create just one of them and reuse it for the other meshes
			// The latter approach has a much higher complexity but makes a lighter mesh
#if 0
			/*temporary to joint nodes: liks two nodes of different geometries, that share the coordinates*/
			std::shared_ptr<::chrono::fea::ChNodeFEAxyz> node;
			std::shared_ptr<::chrono::fea::ChNodeFEAxyz> node2;
			uint numnodes = m_geometries[g]->ptr->GetNumFeaNodes();

			std::shared_ptr<::chrono::fea::ChMesh> fea_mesh_g;
			fea_mesh_g = m_geometries[g]->ptr->GetFeaMesh();

			if (g > 0) { // check on previous geoms, then must be > 0
				for (uint j = 0; j < numnodes; j++) {

					node = std::dynamic_pointer_cast<::chrono::fea::ChNodeFEAxyzD>(fea_mesh_g->GetNode(j));

					if (!node) throw std::runtime_error("Error: impossible to read nodes in JointFeaNode");

					Point ncords;

					ncords(0) = node->GetPos().x();
					ncords(1) = node->GetPos().y();
					ncords(2) = node->GetPos().z();

					for (size_t k = 0; k < g; k++) {

						if (m_geometries[k]->ptr->HasFeaMesh()){

							uint numnodes2 = m_geometries[k]->ptr->GetNumFeaNodes();
							std::shared_ptr<::chrono::fea::ChMesh> fea_mesh_k;
							fea_mesh_k = m_geometries[k]->ptr->GetFeaMesh();

							for (uint i = 0; i < numnodes2; i++) {

								node2 = std::dynamic_pointer_cast<::chrono::fea::ChNodeFEAxyzD>(fea_mesh_k->GetNode(i));
								if (!node2) throw std::runtime_error("Error: impossible to read nodes in JointFeaNode");

								Point ncords2;

								ncords2(0) = node2->GetPos().x();
								ncords2(1) = node2->GetPos().y();
								ncords2(2) = node2->GetPos().z();

								if (abs(ncords2(0) - ncords(0)) < m_deltap &&
									abs(ncords2(1) - ncords(1)) < m_deltap &&
									abs(ncords2(2) - ncords(2)) < m_deltap) {

									auto constraint = std::make_shared<::chrono::fea::ChLinkPointPoint>();
									constraint->Initialize(node, node2);
									m_chrono_system->Add(constraint);
									cout << "linked nodes " << node << " and " << node2 << endl;
								}

							}
						}
					}

				}
			}
			/*-------------------------------------*/

#endif


		}

	} // iterate on geometries

	// call user-set filtering routine, if any
	filterPoints(m_fluidParts, m_boundaryParts);

	static const string sep = string(18, '-') + "+";
	printf("| %72s%58s |\n", "Particle count", "");
	printf("+");
	for (int i = 0; i < 7; ++i) cout << sep;
	cout << endl;
	printf("| %16s | %16s | %16s | %16s | %16s | %16s | %16s |\n",
		"fluid", "boundary", "testpoints", "bodies", "HDF5", "XYZ", "TeT");
	printf("| %16zu | %16zu | %16zu | %16u | %16u | %16u | %16u |\n",
		m_fluidParts.size(),
		m_boundaryParts.size(),
		m_testpointParts.size(),
		bodies_parts_counter,
		hdf5file_parts_counter,
		xyzfile_parts_counter,
		tetfile_parts_counter);

	return m_fluidParts.size() + m_boundaryParts.size() + m_testpointParts.size() +
		bodies_parts_counter + hdf5file_parts_counter + xyzfile_parts_counter +
		tetfile_parts_counter;
}

void ProblemAPI<1>::copy_planes(PlaneList &planes)
{
	if (m_numPlanes == 0) return;
	// look for planes
	uint currPlaneIdx = 0;
	// NOTE: could iterate on planes only with a map plane_index -> gid
	for (uint gid = 0, num_geoms = m_geometries.size(); gid < num_geoms; gid++) {

		// skip deleted
		if (! m_geometries[gid]->enabled) continue;

		if ( m_geometries[gid]->fill_type == FT_UNFILL)
			continue;

		// not a plane?
		if (m_geometries[gid]->type != GT_PLANE) continue;

		auto plane = static_pointer_cast<Plane>(m_geometries[gid]->ptr);

		planes.push_back( implicit_plane( plane->getA(), plane->getB(), plane->getC(), plane->getD() ) );

		currPlaneIdx++;
	}
}

// auxiliary method to return the “vertical” component depending on the number of dimensions
static constexpr double vertical_coord(double4 const& globalPos, int dims)
{
	return
		dims == 1 ? globalPos.x :
		dims == 2 ? globalPos.y :
		globalPos.z;
}

void ProblemAPI<1>::copy_to_array(BufferList &buffers)
{
	const uint dims = space_dimensions_for(simparams()->dimensions);

	float4 *pos = buffers.getData<BUFFER_POS>();
	double4 *globalPos = buffers.getData<BUFFER_POS_GLOBAL>();
	hashKey *hash = buffers.getData<BUFFER_HASH>();
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();
	vertexinfo *vertices = buffers.getData<BUFFER_VERTICES>();
	float4 *boundelm = buffers.getData<BUFFER_BOUNDELEMENTS>();
	float4 *eulerVel = buffers.getData<BUFFER_EULERVEL>();

	// NOTEs and TODO
	// - SA currently supported only from file. Support runtime generation?
	// - Warn if loaded particle has different type than filled, but only once

	// particles counters, by type
	uint fluid_parts = 0;
	uint boundary_parts = 0;
	uint vertex_parts = 0;
	uint testpoint_parts = 0;
	// count #particles loaded from HDF5 files. Needed also to adjust connectivity afterward
	uint hdf5_loaded_parts = 0;
	// count #particles loaded from XYZ files. Only for information
	uint xyz_loaded_parts = 0;
	// Total number of filled parts, i.e. in GPUSPH array and ready to be uploaded.
	uint tot_parts = 0;
	// The following hold:
	//   total = fluid_parts + boundary_parts + vertex_parts + testpoint_parts
	//   total >= hdf5_loaded_parts + xyz_loaded_parts
	//   total >= object_parts
	// NOTE: particles loaded from HDF5 or XYZ files are counted both in the respective
	// *_loaded_parts counter and in the type counter (e.g. fluid_parts).

	// store mass for each particle type
	double fluid_part_mass = NAN;
	double boundary_part_mass = NAN;
	double vertex_part_mass = NAN;

	// we use a simple map for the HDF5_id->id map translation (connectivity fix)
	map<uint, uint> hdf5idx_to_idx_map;

	// count how many particles will be loaded from file
	for (size_t g = 0, num_geoms = m_geometries.size(); g < num_geoms; g++) {
		if (m_geometries[g]->has_hdf5_file)
			hdf5_loaded_parts += m_geometries[g]->hdf5_reader->getNParts();
		else
		if (m_geometries[g]->has_xyz_file)
			xyz_loaded_parts += m_geometries[g]->xyz_reader->getNParts();
	}

	// copy filled testpoint parts
	// NOTE: filling testpoint parts first so that if they are a fixed number they will have
	// the same particle id, independently from the deltap used
	for (uint i = tot_parts; i < tot_parts + m_testpointParts.size(); i++) {
		info[i] = make_particleinfo(PT_TESTPOINT, 0, i);
		calc_localpos_and_hash(m_testpointParts[i - tot_parts], info[i], pos[i], hash[i]);
		globalPos[i] = m_testpointParts[i - tot_parts].toDouble4();
		// Compute density for hydrostatic filling. FIXME for multifluid
		float rho = atrest_density(0);
		if (m_hydrostaticFilling && simparams()->boundarytype == DYN_BOUNDARY)
			rho = hydrostatic_density(m_waterLevel - vertical_coord(globalPos[i], dims), 0);
		vel[i] = make_float4(0, 0, 0, rho);
		if (eulerVel)
			eulerVel[i] = make_float4(0);
		if (i == tot_parts)
			boundary_part_mass = pos[i].w;
	}
	tot_parts += m_testpointParts.size();
	testpoint_parts += m_testpointParts.size();

	// copy filled fluid parts
	for (uint i = tot_parts; i < tot_parts + m_fluidParts.size(); i++) {
		info[i]= make_particleinfo(PT_FLUID,0,i);
		calc_localpos_and_hash(m_fluidParts[i - tot_parts], info[i], pos[i], hash[i]);
		globalPos[i] = m_fluidParts[i - tot_parts].toDouble4();
		// Compute density for hydrostatic filling. FIXME for multifluid
		float rho = atrest_density(0);
		if (m_hydrostaticFilling)
			rho = hydrostatic_density(m_waterLevel - vertical_coord(globalPos[i], dims), 0);
		vel[i] = make_float4(0, 0, 0, rho);
		if (eulerVel)
			eulerVel[i] = make_float4(0);
		if (i == tot_parts)
			fluid_part_mass = pos[i].w;
	}
	tot_parts += m_fluidParts.size();
	fluid_parts += m_fluidParts.size();

	// copy filled boundary parts
	for (uint i = tot_parts; i < tot_parts + m_boundaryParts.size(); i++) {
		info[i] = make_particleinfo(PT_BOUNDARY, 0, i);
		calc_localpos_and_hash(m_boundaryParts[i - tot_parts], info[i], pos[i], hash[i]);
		globalPos[i] = m_boundaryParts[i - tot_parts].toDouble4();
		// Compute density for hydrostatic filling. FIXME for multifluid
		float rho = atrest_density(0);
		if (m_hydrostaticFilling && simparams()->boundarytype == DYN_BOUNDARY)
			rho = hydrostatic_density(m_waterLevel - vertical_coord(globalPos[i], dims), 0);
		vel[i] = make_float4(0, 0, 0, rho);
		if (eulerVel)
			eulerVel[i] = make_float4(0);
		if (i == tot_parts)
			boundary_part_mass = pos[i].w;
	}
	tot_parts += m_boundaryParts.size();
	boundary_parts += m_boundaryParts.size();

	// We've already counted the objects in initialize(), but now we need incremental counters
	// to compute the correct object_id according to the insertion order and body type.
	// Specifically, object_ids are coherent with the insertion order but floating bodies
	// are assigned first; then forces bodies; and finally moving bodies.
	// Please note how they count one kind of body, not including the previous category, as
	// opposite as the global counters (e.g. m_numForcesBodies includes m_numFloatingBodies,
	// but forces_nonFloating_bodies_incremental does not).
	// (i.e. it counts forces non-floating bodies only)
	// Also see Problem::add_moving_body()
	uint floating_bodies_incremental = 0;
	uint forces_nonFloating_bodies_incremental = 0;
	uint moving_nonForces_bodies_incremental = 0;

	// fea bodies are counted as objects independently in their list.
	uint fea_bodies_incremental = 0;

	// finally, a counter of all forces bodies (either floating or not), only for printing information
	uint forces_bodies_incremental = 0;
	// Open boundaries are orthogonal to any kind of bodies, so their object_id will be simply
	// equal to the incremental counter.
	uint open_boundaries_counter = 0;
	// store particle mass of last added rigid body
	double rigid_body_part_mass = NAN;

	// Filling s_hRbLastIndex[i] requires the knowledge of the number of particles filled by any
	// object_id < i. Unfortunately, since object_id does not follow the GeometryID, we do not have
	// this knowledge. Thus, we prepare a small array we will fill the number of particles per
	// object, and we will use it at the end of the filling process to fill s_hRbLastIndex[] (and
	// fix s_hRbFirstIndex - see comments later).
	uint *body_particle_counters = new uint[m_numForcesBodies];


	/*for every object we detect the nodes associated to the elements considering index 0
	 * for the first node in the object. The number of nodes previously added must be considered
	 * to get the global value.*/
	uint nodes_offset = 0;

	// counts fea particles
	uint fea_parts_tracker = 0;
	// countr fea nodes
	uint fea_nodes_tracker = 0;

	// Until now we copied fluid and boundary particles not belonging to floating objects and/or not to be loaded
	// from HDF5 files. Now we iterate on the geometries with the aim to
	// - copy HDF5 particles from HDF5 files (they are not copied to the global particle vectors, and that's the reason
	//   why currently no erase operations are supported with HDF5-loaded geometries);
	// - copy particles of floating objects, since they fill their own point vector;
	// - setup stuff related to floating objects (e.g. object particle count, flags, etc.).
	for (size_t g = 0, num_geoms = m_geometries.size(); g < num_geoms; g++) {

		// planes do not fill particles nor they load from files
		if (m_geometries[g]->type == GT_PLANE)
			continue;

		// skip deleted geometries
		if (!m_geometries[g]->enabled)
			continue;

		// number of particles loaded or filled by the current geometry
		uint current_geometry_particles = 0;
		// id of first boundary particle (both LJ and SA) and number of them in
		// current geometry, used to compute the rb index offset for forces bodies
		uint current_geometry_num_boundary_parts = 0;
		uint current_geometry_first_boundary_id = UINT_MAX;

		// object id (GPUSPH, not Chrono) that will be used in particleinfo
		// TODO: will also be fluid_number for multifluid
		// NOTE: see comments in the declaration of the counters, above
		uint object_id = 0;
		if (m_geometries[g]->type == GT_FLOATING_BODY)
			object_id = floating_bodies_incremental++;
		else
		if (m_geometries[g]->type == GT_MOVING_BODY && m_geometries[g]->measure_forces)
			// forces body; not floating. ID after floating bodies
			object_id = m_numFloatingBodies + forces_nonFloating_bodies_incremental++;
		else
		if (m_geometries[g]->type == GT_MOVING_BODY)
			// moving body; not floating, no feedback. ID after forces (incl. floating) bodies
			object_id = m_numForcesBodies + moving_nonForces_bodies_incremental++;
		if (m_geometries[g]->type == GT_OPENBOUNDARY)
			// open boundary; nothing to do with bodies
			object_id = open_boundaries_counter++;
		// now update the forces bodies counter, which includes floating ones, only for printing info later
		if (m_geometries[g]->measure_forces)
			forces_bodies_incremental++; //FIXME does not assign it to object_id?

		// load fea mesh. // FIXME now we do this way since we only use STL files. after we must consider that a
		// fea mesh can come from any file

		//cout << "Looking for deformable bodies... " << endl;
		if (m_geometries[g]->type == GT_DEFORMABLE_BODY) {

			//FIXME FIXME ACHTUNG!! this is just to compute the number of fea particles and allocate s_hFeaNatCoords
			//very temporary solution.. find something better
			get_fea_objects_numparts();
			get_fea_objects_numnodes();

			cout << "Copying deformable bodies particles... " << endl;

			const std::vector<Point> TetVector = m_geometries[g]->ptr->GetParts();
			current_geometry_particles = m_geometries[g]->ptr->GetNumParts();

			// Copying FG_DEFORMABLE particles, i.e. boundary of deformable bodies
			for (uint i = tot_parts; i < tot_parts + current_geometry_particles; i++) {

				info[i] = make_particleinfo_by_ids(PT_BOUNDARY, 0, fea_bodies_incremental, i);
				calc_localpos_and_hash(TetVector[i - tot_parts], info[i], pos[i], hash[i]);
				globalPos[i] = TetVector[i - tot_parts].toDouble4();


				// Assigning particle properties
				float rho = atrest_density(0);
				if (m_hydrostaticFilling && simparams()->boundarytype == DYN_BOUNDARY)
					rho = hydrostatic_density(m_waterLevel - globalPos[i].z, 0);
				vel[i] = make_float4(0, 0, 0, rho);

				if (eulerVel)
					eulerVel[i] = make_float4(0);

				if (i == tot_parts)
					boundary_part_mass = pos[i].w;

				SET_FLAG(info[i], FG_DEFORMABLE);


				// natural coordinates: coordinates of the particle within the associated element
				gdata->s_hFeaNatCoords[i - tot_parts + fea_parts_tracker] = m_geometries[g]->ptr->getNaturalCoords(globalPos[i]);

				// nodes associated to the element the FG_DEFORMABLE particle belongs to
				int4 nodes = m_geometries[g]->ptr->getOwningNodes(globalPos[i]);

				// the indices of the owning nodes provided by the function are relative to the
				// object mesh (it can also be negative, if the node belongs to a previously defined object)
				// Here we add an offset, being the sum of nodes already defined for the other geometries)
				// in order to retrieve the global indices
				nodes.x += nodes_offset;
				nodes.y += nodes_offset;
				nodes.z += nodes_offset;
				nodes.w += nodes_offset;

				uint4 casting; //TODO FIXME ... a fancier (or better, clever) way?
				casting.x = (uint) nodes.x;
				casting.y = (uint) nodes.y;
				casting.z = (uint) nodes.z;
				casting.w = (uint) nodes.w;

				gdata->s_hFeaOwningNodes[i - tot_parts + fea_parts_tracker] = casting;

				if (current_geometry_first_boundary_id == UINT_MAX)
					current_geometry_first_boundary_id = id(info[i]);
			}

			// FIXME manage body offset as done for floating bodies
			gdata->s_hFeaPartsFirstIndex[fea_bodies_incremental] = make_int2(-(int) current_geometry_first_boundary_id, -fea_parts_tracker);

			boundary_parts += current_geometry_particles;
			fea_parts_tracker += current_geometry_particles;

			current_geometry_first_boundary_id = UINT_MAX;




			// Copying FG_FEA_NODES: particles with no SPH meaning. Follow the motion of FEA nodes and
			// gather forces from deformable boundary particles.
			cout << "Copying deformable bodies nodes... " << endl;
			const std::vector<Point> NodesVector = m_geometries[g]->ptr->GetFeaNodes();
			uint num_fea_nodes = m_geometries[g]->ptr->GetNumFeaNodes();

			uint already_added = tot_parts + current_geometry_particles;
			for (uint i = already_added; i < already_added + num_fea_nodes; i++) {

				// Addigning particle properties
				info[i] = make_particleinfo_by_ids(PT_BOUNDARY, 0, fea_bodies_incremental, i); //FIXME classifying as boundary: see if something else is better
				calc_localpos_and_hash(NodesVector[i - already_added], info[i], pos[i], hash[i]);
				globalPos[i] = NodesVector[i - already_added].toDouble4();

				float rho = atrest_density(0);
				if (m_hydrostaticFilling && simparams()->boundarytype == DYN_BOUNDARY)
					rho = hydrostatic_density(m_waterLevel - globalPos[i].z, 0);
				vel[i] = make_float4(0, 0, 0, rho);

				if (eulerVel)
					eulerVel[i] = make_float4(0);

				if (i == already_added)
					boundary_part_mass = pos[i].w;

				SET_FLAG(info[i], FG_FEA_NODE);

				if (current_geometry_first_boundary_id == UINT_MAX)
					current_geometry_first_boundary_id = id(info[i]);
			}

			boundary_parts += num_fea_nodes;
			fea_nodes_tracker += num_fea_nodes;
			current_geometry_particles += num_fea_nodes;

			gdata->s_hFeaNodesFirstIndex[fea_bodies_incremental] = make_int2(- (int)current_geometry_first_boundary_id, - nodes_offset);

			nodes_offset += num_fea_nodes;
			fea_bodies_incremental ++;

		}


		// load from HDF5 file, whether fluid, boundary, floating or else
		if (m_geometries[g]->has_hdf5_file) {
			// read number of particles
			current_geometry_particles = m_geometries[g]->hdf5_reader->getNParts();
			// utility pointer
			const ReadParticles *hdf5Buffer = m_geometries[g]->hdf5_reader->buf;
			// add every particle
			for (uint i = tot_parts; i < tot_parts + current_geometry_particles; i++) {

				// "i" is the particle index in GPUSPH host arrays, "bi" the one in current HDF5 file)
				const uint bi = i - tot_parts;

				// By default, set the particle type according to the geometry type
				// (boundary unless geometry type is GT_FLUID). This will be overridden
				// by the ParticleType field imported from the HDF5 file, if present/known.
				ushort ptype = m_geometries[g]->type == GT_FLUID ? PT_FLUID : PT_BOUNDARY;

				// NOTE: update particle counters here, since current_geometry_particles does not distinguish vertex/bound;
				// tot_parts instead is updated in the outer loop
				switch (hdf5Buffer[bi].ParticleType) {
					case CRIXUS_FLUID:
						// TODO: warn user if (m_geometries[g]->type != GT_FLUID)
						ptype = PT_FLUID;
						fluid_parts++;
						break;
					case CRIXUS_VERTEX:
						// TODO: warn user if (m_geometries[g]->type == GT_FLUID)
						ptype = PT_VERTEX;
						vertex_parts++;
						break;
					case CRIXUS_BOUNDARY_PARTICLE:
					case CRIXUS_BOUNDARY:
						// TODO: warn user if (m_geometries[g]->type == GT_FLUID)
						ptype = PT_BOUNDARY;
						boundary_parts++;
						break;
					default:
						// TODO: print warning or throw fatal
						break;
				}

				// compute particle info, local pos, cellhash
				// NOTE: using explicit constructor make_particleinfo_by_ids() since some flags may
				// be set afterward (e.g. in initializeParticles() callback)
				info[i] = make_particleinfo_by_ids(ptype, 0, object_id, i);

				// set appropriate particle flags
				switch (m_geometries[g]->type) {
					case GT_MOVING_BODY:
						SET_FLAG(info[i], FG_MOVING_BOUNDARY);
						if (m_geometries[g]->measure_forces)
							SET_FLAG(info[i], FG_COMPUTE_FORCE);
						break;
					case GT_FLOATING_BODY:
						SET_FLAG(info[i], FG_MOVING_BOUNDARY | FG_COMPUTE_FORCE);
						break;
					case GT_FREE_SURFACE:
						SET_FLAG(info[i], FG_SURFACE);
						break;
					case GT_OPENBOUNDARY:
						const ushort VELOCITY_DRIVEN_FLAG =
							(m_geometries[g]->velocity_driven ? FG_VELOCITY_DRIVEN : 0);
						SET_FLAG(info[i], FG_INLET | FG_OUTLET | VELOCITY_DRIVEN_FLAG);
						break;
				}

				// FIXME for multifluid
				Point tmppoint = Point(hdf5Buffer[bi].Coords_0, hdf5Buffer[bi].Coords_1, hdf5Buffer[bi].Coords_2,
					atrest_physical_density(0)*hdf5Buffer[bi].Volume);
				calc_localpos_and_hash(tmppoint, info[i], pos[i], hash[i]);
				globalPos[i] = tmppoint.toDouble4();

				// Compute density for hydrostatic filling. FIXME for multifluid
				float rho = atrest_density(0);
				if (m_hydrostaticFilling && (ptype == PT_FLUID || ptype == PT_VERTEX || simparams()->boundarytype == DYN_BOUNDARY))
					rho = hydrostatic_density(m_waterLevel - vertical_coord(globalPos[i], dims), 0);
				vel[i] = make_float4(0, 0, 0, rho);

				// Update boundary particles counters for rb indices
				// NOTE: the same check will be done for non-HDF5 bodies
				if (ptype == PT_BOUNDARY && COMPUTE_FORCE(info[i])) {
					current_geometry_num_boundary_parts++;
					if (current_geometry_first_boundary_id == UINT_MAX)
						current_geometry_first_boundary_id = id(info[i]); // which should be == i
				}

				if (eulerVel)
					eulerVel[i] = make_float4(0);

				// store particle mass for current type, if it was not store already
				if (ptype == PT_FLUID && !isfinite(fluid_part_mass))
					fluid_part_mass = pos[i].w;
				else
				if (ptype == PT_BOUNDARY && !isfinite(boundary_part_mass))
					boundary_part_mass = pos[i].w;
				else
				if (ptype == PT_VERTEX && !isfinite(vertex_part_mass))
					vertex_part_mass = pos[i].w;
				// also set rigid_body_part_mass, which is orthogonal the the previous values
				// TODO: with SA bounds, this value has little meaning or should be split
				if ((m_geometries[g]->type == GT_FLOATING_BODY ||
					 m_geometries[g]->type == GT_MOVING_BODY) &&
					 !isfinite(rigid_body_part_mass))
					rigid_body_part_mass = pos[i].w;

				// load boundary-specific data (SA bounds only)
				if (ptype == PT_BOUNDARY && simparams()->boundarytype == SA_BOUNDARY) {
					if (m_geometries[g]->flip_normals) {
						// NOTE: simulating with flipped normals has not been numerically validated...
						// invert the order of vertices so that for the mass it is m_ref - m_v
						vertices[i].x = hdf5Buffer[bi].VertexParticle3;
						vertices[i].y = hdf5Buffer[bi].VertexParticle2;
						vertices[i].z = hdf5Buffer[bi].VertexParticle1;
						// load with inverted sign
						boundelm[i].x = - hdf5Buffer[bi].Normal_0;
						boundelm[i].y = - hdf5Buffer[bi].Normal_1;
						boundelm[i].z = - hdf5Buffer[bi].Normal_2;
					} else {
						// regular loading
						vertices[i].x = hdf5Buffer[bi].VertexParticle1;
						vertices[i].y = hdf5Buffer[bi].VertexParticle2;
						vertices[i].z = hdf5Buffer[bi].VertexParticle3;
						boundelm[i].x = hdf5Buffer[bi].Normal_0;
						boundelm[i].y = hdf5Buffer[bi].Normal_1;
						boundelm[i].z = hdf5Buffer[bi].Normal_2;
					}

					boundelm[i].w = hdf5Buffer[bi].Surface;
				}

				// update hash map
				if (ptype == PT_VERTEX)
					hdf5idx_to_idx_map[ hdf5Buffer[bi].AbsoluteIndex ] = i;

			} // for every particle in the HDF5 buffer

		} else // if (m_geometries[g]->has_hdf5_file)
		// load from HDF5 file, whether fluid, boundary, floating or else
		if (m_geometries[g]->has_xyz_file) {

			// read number of particles
			current_geometry_particles = m_geometries[g]->xyz_reader->getNParts();

			// all particles in XYZ file will have the same type and flags
			ushort ptype = PT_FLUID;
			flag_t pflags = 0;

			// update particle counters, set ptype, flags - all same for the whole XYZ geometry
			switch (m_geometries[g]->type) {
				case GT_FLUID:
					ptype = PT_FLUID;
					fluid_parts += current_geometry_particles;
					break;
				case GT_FIXED_BOUNDARY:
					ptype = PT_BOUNDARY;
					boundary_parts += current_geometry_particles;
					break;
				case GT_TESTPOINTS:
					ptype = PT_TESTPOINT;
					testpoint_parts += current_geometry_particles;
					break;
				case GT_MOVING_BODY:
					pflags = FG_MOVING_BOUNDARY;
					if (m_geometries[g]->measure_forces)
						pflags |= FG_COMPUTE_FORCE;
					ptype = PT_BOUNDARY;
					boundary_parts += current_geometry_particles;
					break;
				case GT_FLOATING_BODY:
					pflags = FG_MOVING_BOUNDARY | FG_COMPUTE_FORCE;
					ptype = PT_BOUNDARY;
					boundary_parts += current_geometry_particles;
					break;
				case GT_OPENBOUNDARY:
					const ushort VELOCITY_DRIVEN_FLAG =
						(m_geometries[g]->velocity_driven ? FG_VELOCITY_DRIVEN : 0);
					pflags = FG_INLET | FG_OUTLET | VELOCITY_DRIVEN_FLAG;
					// TODO FIXME: check compatibility with new non-SA inlets
					ptype = PT_BOUNDARY;
					boundary_parts += current_geometry_particles;
					break;
				}
			// TODO: nothing else is possible since this is checked while adding the
			// geometry, should we double-check again? And a default

			// utility pointer to the first ReadParticle
			const ReadParticles *xyzParticle = m_geometries[g]->xyz_reader->buf;

			// add every particle
			for (uint i = tot_parts; i < tot_parts + current_geometry_particles; ++i, ++xyzParticle) {
				// compute particle info, local pos, cellhash
				// NOTE: using explicit constructor make_particleinfo_by_ids() since some flags may
				// be set afterward (e.g. in initializeParticles() callback)
				info[i] = make_particleinfo_by_ids(ptype, 0, object_id, i);

				// set appropriate particle flags
				SET_FLAG(info[i], pflags);

				// NOTE: reading the mass from the object, even if it is an empty STL
				Point tmppoint = Point(xyzParticle->Coords_0, xyzParticle->Coords_1, xyzParticle->Coords_2,
					m_geometries[g]->ptr->GetPartMass());
				calc_localpos_and_hash(tmppoint, info[i], pos[i], hash[i]);
				globalPos[i] = tmppoint.toDouble4();

				// Compute density for hydrostatic filling. FIXME for multifluid
				float rho = atrest_density(0);
				if (m_hydrostaticFilling && (ptype == PT_FLUID || ptype == PT_VERTEX || simparams()->boundarytype == DYN_BOUNDARY))
					rho = hydrostatic_density(m_waterLevel - vertical_coord(globalPos[i], dims), 0);
				vel[i] = make_float4(0, 0, 0, rho);

				// Update boundary particles counters for rb indices
				// NOTE: the same check will be done for non-HDF5 bodies
				if (ptype == PT_BOUNDARY && m_geometries[g]->measure_forces) {
					current_geometry_num_boundary_parts++;
					if (current_geometry_first_boundary_id == UINT_MAX)
						current_geometry_first_boundary_id = id(info[i]); // which should be == i
				}

				if (eulerVel)
					eulerVel[i] = make_float4(0);

				// store particle mass for current type, if it was not store already
				if (ptype == PT_FLUID && !isfinite(fluid_part_mass))
					fluid_part_mass = pos[i].w;
				else
				if (ptype == PT_BOUNDARY && !isfinite(boundary_part_mass))
					boundary_part_mass = pos[i].w;
				// no else supported yet

				// also set rigid_body_part_mass, which is orthogonal the the previous values
				if ((m_geometries[g]->type == GT_FLOATING_BODY ||
					 m_geometries[g]->type == GT_MOVING_BODY) &&
					 !isfinite(rigid_body_part_mass))
					rigid_body_part_mass = pos[i].w;

			} // for every particle in the XYZ buffer

		} // if (m_geometries[g]->has_xyz_file)

		// copy particles from the point vector of objects which have not been loaded from file
		if ( (m_geometries[g]->type == GT_FLOATING_BODY || m_geometries[g]->type == GT_MOVING_BODY)
				&& !(m_geometries[g]->has_hdf5_file) && !(m_geometries[g]->has_xyz_file)) {
			// not loading from file: take object vector
			PointVect & rbparts = m_geometries[g]->ptr->GetParts();
			current_geometry_particles = rbparts.size();
			// copy particles
			for (uint i = tot_parts; i < tot_parts + current_geometry_particles; i++) {
				// TODO FIXME MERGE
				// NOTE: using explicit constructor make_particleinfo_by_ids() since some flags may
				// be set afterward (e.g. in initializeParticles() callback)
				info[i] = make_particleinfo_by_ids(PT_BOUNDARY, 0, object_id, i);
				calc_localpos_and_hash(rbparts[i - tot_parts], info[i], pos[i], hash[i]);
				globalPos[i] = rbparts[i - tot_parts].toDouble4();
				// Compute density for hydrostatic filling. FIXME for multifluid
				float rho = atrest_density(0);
				if (m_hydrostaticFilling && simparams()->boundarytype == DYN_BOUNDARY)
					rho = hydrostatic_density(m_waterLevel - vertical_coord(globalPos[i], dims), 0);
				vel[i] = make_float4(0, 0, 0, rho);
				if (eulerVel)
					// there should be no eulerVel with LJ bounds, but it is safe to init the array anyway
					eulerVel[i] = make_float4(0);
				// NOTE: setting/showing rigid_body_part_mass only makes sense with non-SA bounds
				if (m_geometries[g]->type == GT_FLOATING_BODY && !isfinite(rigid_body_part_mass))
					rigid_body_part_mass = pos[i].w;
				// set appropriate particle flags
				switch (m_geometries[g]->type) {
					case GT_MOVING_BODY:
						SET_FLAG(info[i], FG_MOVING_BOUNDARY);
						if (m_geometries[g]->measure_forces)
							SET_FLAG(info[i], FG_COMPUTE_FORCE);
						break;
					case GT_FLOATING_BODY:
						SET_FLAG(info[i], FG_MOVING_BOUNDARY | FG_COMPUTE_FORCE);
						break;
					// not possible
					//case GT_OPENBOUNDARY:
					//	break;
				}

				// Update boundary particles counters for rb indices
				// NOTE: this is the safest way to update the counters, although
				// with LJ boundaries we could directly set the values afterwards
				// instead of checking every particle
				if (COMPUTE_FORCE(info[i])) {
					current_geometry_num_boundary_parts++;
					if (current_geometry_first_boundary_id == UINT_MAX)
						current_geometry_first_boundary_id = id(info[i]); // which should be == i
				}

			} // for every particle of body
		} // if current geometry is a body and is not loaded from file

		// settings related to objects for which we compute the forces, regardless they were loaded from file or not
		if (m_geometries[g]->measure_forces) {
			// In s_hRbFirstIndex it is stored the id of the first particle of current body (changed
			// in sign, since it is used as an offset) plus the number of previously filled object
			// particles. The former addendum is set here, the latter will be added later (when we'll
			// know the number of particles of all the bodies).
			gdata->s_hRbFirstIndex[object_id] = - (int)current_geometry_first_boundary_id;

			// update counter of rigid body particles
			body_particle_counters[object_id] = current_geometry_num_boundary_parts;

			// recap on stdout
			cout << "Rigid body " << forces_bodies_incremental << ": " << current_geometry_particles <<
				" parts, mass " << rigid_body_part_mass << ", object mass " << m_geometries[g]->ptr->GetMass() << "\n";

			// reset value to spot possible anomalies in next bodies
			rigid_body_part_mass = NAN;
		}

		// update object num parts
		if (m_geometries[g]->type == GT_FLOATING_BODY ||
			m_geometries[g]->type == GT_MOVING_BODY) {
			// set numParts, which will be read while allocating device buffers for obj parts
			// NOTE: this is strictly necessary only for hdf5-loaded objects, because
			// when numparts==0, Object uses rbparts.size().
			m_geometries[g]->ptr->SetNumParts(current_geometry_num_boundary_parts);
		}

		// update global particle counter
		tot_parts += current_geometry_particles;
	} // for each geometry

	// Now we fix s_hRbFirstIndex and fill s_hRbLastIndex. We iterate on all the
	// forces bodies by means of their object id, which is basically the insertion
	// order after being sorted by body type, to keep and incremental particle counter.
	uint incremental_bodies_part_counter = 0;
	for (uint obj_id = 0; obj_id < m_numForcesBodies; obj_id++) {
		// s_hRbFirstIndex is currently -first_bound_id; shift it further according to the previous bodies
		gdata->s_hRbFirstIndex[obj_id] += incremental_bodies_part_counter;
		// now let's increment incremental_bodies_part_counter with current body
		incremental_bodies_part_counter += body_particle_counters[obj_id];
		// memo: s_hRbLastIndex, as used in the reduction, is inclusive (thus -1)
		gdata->s_hRbLastIndex[obj_id] = incremental_bodies_part_counter - 1;
#if 0 // DBG info
		printf(" DBG: s_hRbFirstIndex[%u] = %d, s_hRbLastIndex[%u] = %u\n",
			obj_id, gdata->s_hRbFirstIndex[obj_id], obj_id, gdata->s_hRbLastIndex[obj_id]);
#endif
	}
	delete [] body_particle_counters;

	// fix connectivity by replacing Crixus' AbsoluteIndex with local index
	// TODO: instead of iterating on all the particles, we could create a list of boundary particles while
	// loading them from file, and here iterate only on that vector
	if (simparams()->boundarytype == SA_BOUNDARY && hdf5_loaded_parts > 0) {
		cout << "Fixing connectivity..." << flush;
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
		cout << "DONE" << "\n";
		hdf5idx_to_idx_map.clear();
	}

	// FIXME: move this somewhere else
	printf("Open boundaries: %zu\n", m_numOpenBoundaries);

	cout << "Fluid: " << fluid_parts << " parts, mass " << fluid_part_mass << "\n";
	cout << "Boundary: " << boundary_parts << " parts, mass " << boundary_part_mass << "\n";
	if (simparams()->boundarytype == SA_BOUNDARY)
		cout << "Vertices: " << vertex_parts << " parts, mass " << vertex_part_mass << "\n";
	cout << "Testpoint: " << testpoint_parts << " parts\n";
	cout << "Tot: " << tot_parts << " particles\n";
	flush(cout);

	if (tot_parts != gdata->totParticles)
		throw logic_error("particle count mismatch: fill = " + to_string(gdata->totParticles) +
			", copy =  " + to_string(tot_parts));
}

// callback for filtering out points before they become particles (e.g. unfills/cuts)
void ProblemAPI<1>::filterPoints(PointVect &fluidParts, PointVect &boundaryParts)
{
		// Default: do nothing

	/*
	// Example usage
	// TODO
	*/
}

