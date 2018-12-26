/*  Copyright 2013 Alexis Herault, Giuseppe Bilotta, Robert A.
 	Dalrymple, Eugenio Rustico, Ciro Del Negro

	Conservatoire National des Arts et Metiers, Paris, France

	Istituto Nazionale di Geofisica e Vulcanologia,
    Sezione di Catania, Catania, Italy

    Universita di Catania, Catania, Italy

    Johns Hopkins University, Baltimore, MD

	This file is part of GPUSPH.

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
 * Core Problem class implementation
 */

#include <sstream>
#include <stdexcept>
#include <string>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

// shared_ptr
#include <memory>

#include "Problem.h"
#include "vector_math.h"
#include "vector_print.h"
#include "utils.h"

// here we need the complete definition of the GlobalData struct
#include "GlobalData.h"

// COORD1, COORD2, COORD3
#include "linearization.h"

#if USE_CHRONO
#include "chrono/physics/ChSystemNSC.h"
#include "chrono/solver/ChSolver.h"
#endif

// Enable to get/set envelop and margin (mainly debug)
/*
#ifndef USE_CMAKE
#include "chrono_select.opt"
#endif // USE_CMAKE
#if USE_CHRONO == 1
#include "chrono/collision/ChCCollisionModel.h"
#endif
*/

using namespace std;

Problem::Problem(GlobalData *_gdata) :
	m_problem_dir(_gdata->clOptions->dir),
	m_dem(NULL),
	m_physparams(new PhysParams()),
	m_simframework(NULL),
	m_size(make_double3(NAN, NAN, NAN)),
	m_origin(make_double3(NAN, NAN, NAN)),
	m_deltap(NAN),
	gdata(_gdata),
	m_options(_gdata->clOptions),
	m_bodies_storage(NULL)
{
#if USE_CHRONO == 1
	m_bodies_physical_system = NULL;
#endif
}

bool
Problem::initialize()
{
	SimParams const* _sp(simparams());
	PhysParams const* _pp(physparams());

	if (_sp->gage.size() > 0 && !m_simframework->hasPostProcessEngine(SURFACE_DETECTION)) {
		printf("Wave gages present: force-enabling surface detection\n");
		m_simframework->addPostProcessEngine(SURFACE_DETECTION);
	}

	const bool multi_fluid_flag = IS_MULTIFLUID(_sp->simflags);
	const bool has_multiple_fluids = (_pp->numFluids() > 1);

	if (has_multiple_fluids && !multi_fluid_flag) {
		throw invalid_argument(to_string(_pp->numFluids()) +
			" fluids defined, but ENABLE_MULTIFLUID missing from simulation flags");
	}

	if (multi_fluid_flag && !has_multiple_fluids) {
		fprintf(stderr, "WARNING: multi-fluid support enabled, but only one fluid defined\n"
			"Viscous computation will not be optimized\n");
	}

	// run post-construction functions
	check_dt();
	check_neiblistsize();
	calculateDensityDiffusionCoefficient();

	/* Set artificial viscosity epsilon to h^2/10 if not set by the user.
	 * For simplicity, we do this regardless of the viscosity model used,
	 * it'll just be ignored otherwise */
	if (simparams()->turbmodel == ARTIFICIAL && isnan(physparams()->epsartvisc)){
		physparams()->epsartvisc = 0.01*simparams()->slength*simparams()->slength;
		printf("ARTVISC epsilon is not set, using default value: %e\n", physparams()->epsartvisc);		
	}

	if (simparams()->sph_formulation == SPH_GRENIER && isnan(physparams()->epsinterface)){
		physparams()->epsinterface = 0.05;
		printf("Grenier's interface epsilon is not set, using default value: %e\n", physparams()->epsinterface);
	}

	create_problem_dir();

	printf("Problem calling set grid params\n");
	set_grid_params();

	return true;
}

Problem::~Problem(void)
{
	delete [] m_bodies_storage;
	delete m_simframework;
	delete m_physparams;
}

void
Problem::InitializeChrono()
{
#if USE_CHRONO == 1
	m_bodies_physical_system = new ::chrono::ChSystemNSC();
	m_bodies_physical_system->Set_G_acc(::chrono::ChVector<>(m_physparams->gravity.x, m_physparams->gravity.y,
		m_physparams->gravity.z));
	m_bodies_physical_system->SetMaxItersSolverSpeed(100);
	m_bodies_physical_system->SetSolverType(::chrono::ChSolver::Type::SOR);
	// For debug purposes
	/*
	const double chronoSuggEnv = ::chrono::collision::ChCollisionModel::GetDefaultSuggestedEnvelope();
	const double chronoSuggMarg = ::chrono::collision::ChCollisionModel::GetDefaultSuggestedMargin();
	printf("Default envelop: %g, default margin: %g\n", chronoSuggEnv, chronoSuggMarg);
	::chrono::collision::ChCollisionModel::SetDefaultSuggestedEnvelope(chronoSuggEnv / 10.0);
	::chrono::collision::ChCollisionModel::SetDefaultSuggestedMargin(chronoSuggMarg / 10.0);
	*/
#else
	throw runtime_error ("Problem::InitializeChrono Trying to use Chrono without USE_CHRONO defined !\n");
#endif
}

void Problem::FinalizeChrono(void)
{
#if USE_CHRONO == 1
	if (m_bodies_physical_system)
		delete m_bodies_physical_system;
#else
	throw runtime_error ("Problem::FinalizeChrono Trying to use Chrono without USE_CHRONO defined !\n");
#endif
}

// callback for initializing joints between Chrono bodies
void Problem::initializeObjectJoints()
{
	// Default: do nothing

	// See also: http://api.chrono.projectchrono.org/links.html
}

/// Allocate storage required for the integration of the kinematic data
/// of moving bodies.
void
Problem::allocate_bodies_storage()
{
	const uint nbodies = simparams()->numbodies;

	if (nbodies) {
		// TODO: this should depend on the integration scheme
		m_bodies_storage = new KinematicData[nbodies];
	}
}

void
Problem::add_moving_body(Object* object, const MovingBodyType mbtype)
{
	// Moving bodies are put at the end of the bodies vector,
	// ODE bodies and moving bodies for which we want a force feedback
	// are put at the beginning of the bodies vector (ODE bodies first).
	// The reason behind this ordering is the way the forces on bodies
	// are reduced by a parallel prefix sum: all the bodies that require
	// force computing must have consecutive ids.
	const uint index = m_bodies.size();
	if (index >= MAX_BODIES)
		throw runtime_error ("Problem::add_moving_body Number of moving bodies superior to MAX_BODIES. Increase MAXBODIES\n");
	MovingBodyData *mbdata = new MovingBodyData;
	mbdata->index = index;
	mbdata->type = mbtype;
	mbdata->object = object;
	mbdata->kdata.crot = object->GetCenterOfGravity();
	mbdata->kdata.lvel = make_double3(0.0f);
	mbdata->kdata.avel = make_double3(0.0f);
	mbdata->kdata.orientation = object->GetOrientation();
	switch (mbdata->type) {
		case MB_FLOATING : {
#if USE_CHRONO == 1
			std::shared_ptr< ::chrono::ChBody > body = object->GetBody();
			::chrono::ChVector<> vec = body->GetPos();
			mbdata->kdata.crot = make_double3(vec.x(), vec.y(), vec.z());
			vec = body->GetPos_dt();
			mbdata->kdata.lvel = make_double3(vec.x(), vec.y(), vec.z());
			vec = body->GetWvel_par();
			mbdata->kdata.avel = make_double3(vec.x(), vec.y(), vec.z());
			::chrono::ChQuaternion<> quat = body->GetRot();
			m_bodies.insert(m_bodies.begin() + simparams()->numODEbodies, mbdata);
			simparams()->numODEbodies++;
			simparams()->numforcesbodies++;
#else
			throw runtime_error ("Problem::add_moving_body Cannot add a floating body without CHRONO\n");
#endif
			break;
		}

		case MB_FORCES_MOVING:
			m_bodies.insert(m_bodies.begin() + simparams()->numforcesbodies, mbdata);
			simparams()->numforcesbodies++;
			break;

		case MB_MOVING:
			m_bodies.push_back(mbdata);
			break;
	}

	// Setting body id after insertion
	for (uint id = 0; id < m_bodies.size(); id++)
		m_bodies[id]->id = id;

	mbdata->initial_kdata = mbdata->kdata;

	simparams()->numbodies = m_bodies.size();
}

void
Problem::restore_moving_body(const MovingBodyData & saved_mbdata, const uint numparts, const int firstindex, const int lastindex)
{
	const uint id = saved_mbdata.id;
	MovingBodyData *mbdata = m_bodies[id];
	mbdata->object->SetNumParts(numparts);
	mbdata->initial_kdata = saved_mbdata.initial_kdata;
	mbdata->kdata = saved_mbdata.kdata;
	if (mbdata->type == MB_FORCES_MOVING || mbdata->type == MB_FLOATING) {
		gdata->s_hRbFirstIndex[id] = firstindex;
		gdata->s_hRbLastIndex[id] = lastindex;
	}

	if (mbdata->type == MB_FLOATING) {
#if USE_CHRONO == 1
		std::shared_ptr< ::chrono::ChBody > body = mbdata->object->GetBody();
		body->SetPos(::chrono::ChVector<>(mbdata->kdata.crot.x, mbdata->kdata.crot.y, mbdata->kdata.crot.z));
		body->SetPos_dt(::chrono::ChVector<>(mbdata->kdata.lvel.x, mbdata->kdata.lvel.y, mbdata->kdata.lvel.z));
		body->SetWvel_par(::chrono::ChVector<>(mbdata->kdata.avel.x, mbdata->kdata.avel.y, mbdata->kdata.avel.z));
		body->SetRot(mbdata->kdata.orientation.ToChQuaternion());
#else
		throw runtime_error ("Problem::restore_moving_body Cannot restore a floating body without CHRONO\n");
#endif
		}
}

MovingBodyData *
Problem::get_mbdata(const uint index)
{
	if (index >= m_bodies.size()) {
		stringstream ss;
		ss << "get_body: body number " << index << " >= numbodies";
		throw runtime_error(ss.str());
	}
	for (vector<MovingBodyData *>::iterator it = m_bodies.begin() ; it != m_bodies.end(); ++it) {
		if ((*it)->index == index)
			return *it;
	}
	return NULL;
}


MovingBodyData *
Problem::get_mbdata(const Object* object)
{
	for (vector<MovingBodyData *>::iterator it = m_bodies.begin() ; it != m_bodies.end(); ++it) {
		if ((*it)->object == object)
			return *it;
	}
	throw runtime_error("get_body: invalid object\n");
	return NULL;
}

size_t
Problem::get_bodies_numparts(void)
{
	size_t total_parts = 0;
	for (vector<MovingBodyData *>::iterator it = m_bodies.begin() ; it != m_bodies.end(); ++it) {
		total_parts += (*it)->object->GetNumParts();
	}

	return total_parts;
}


size_t
Problem::get_forces_bodies_numparts(void)
{
	size_t total_parts = 0;
	for (vector<MovingBodyData *>::iterator it = m_bodies.begin() ; it != m_bodies.end(); ++it) {
		if ((*it)->type == MB_FLOATING || (*it)->type == MB_FORCES_MOVING)
			total_parts += (*it)->object->GetNumParts();
	}
	return total_parts;
}


size_t
Problem::get_body_numparts(const int index)
{
	return m_bodies[index]->object->GetNumParts();
}


size_t
Problem::get_body_numparts(const Object* object)
{
	return get_mbdata(object)->object->GetNumParts();
}

void
Problem::calc_grid_and_local_pos(double3 const& globalPos, int3 *gridPos, float3 *localPos) const
{
	int3 _gridPos = calc_grid_pos(globalPos);
	*gridPos = _gridPos;
	*localPos = make_float3(globalPos - m_origin -
		(make_double3(_gridPos) + 0.5)*m_cellsize);
}

void
Problem::get_bodies_cg(void)
{
	for (uint i = 0; i < simparams()->numbodies; i++) {
		calc_grid_and_local_pos(m_bodies[i]->kdata.crot,
			gdata->s_hRbCgGridPos + i,
			gdata->s_hRbCgPos + i);
		cout << "Body: " << i << endl;
		cout << "\t Cg grid pos: " << gdata->s_hRbCgGridPos[i].x << " " << gdata->s_hRbCgGridPos[i].y << " " << gdata->s_hRbCgGridPos[i].z << endl;
		cout << "\t Cg pos: " << gdata->s_hRbCgPos[i].x << " " << gdata->s_hRbCgPos[i].y << " " << gdata->s_hRbCgPos[i].z << endl;
	}
}


void
Problem::set_body_cg(const double3& crot, MovingBodyData* mbdata) {
	mbdata->kdata.crot = crot;
}


void
Problem::set_body_cg(const uint index, const double3& crot) {
	set_body_cg(crot, m_bodies[index]);
}


void
Problem::set_body_cg(const Object *object, const double3& crot) {
	set_body_cg(crot, get_mbdata(object));
}


void
Problem::set_body_linearvel(const double3& lvel, MovingBodyData* mbdata) {
	mbdata->kdata.lvel = lvel;
}


void
Problem::set_body_linearvel(const uint index, const double3& lvel) {
	set_body_linearvel(lvel, m_bodies[index]);
}


void
Problem::set_body_linearvel(const Object *object, const double3& lvel)
{
	set_body_linearvel(lvel, get_mbdata(object));
}


void
Problem::set_body_angularvel(const double3& avel, MovingBodyData* mbdata) {
	mbdata->kdata.avel = avel;
}

void
Problem::set_body_angularvel(const uint index, const double3& avel) {
	set_body_angularvel(avel, m_bodies[index]);
}


void
Problem::set_body_angularvel(const Object *object, const double3& avel)
{
	set_body_angularvel(avel, get_mbdata(object));
}


void
Problem::bodies_forces_callback(const double t0, const double t1, const uint step, float3 *forces, float3 *torques)
{ /* default does nothing */ }


void
Problem::post_timestep_callback(const double t)
{ /* default does nothing */ }


void
Problem::moving_bodies_callback(const uint index, Object* object, const double t0, const double t1,
		const float3& force, const float3& torque, const KinematicData& initial_kdata,
		KinematicData& kdata, double3& dx, EulerParameters& dr)
{ /* default does nothing */ }

// input: force, torque, step number, dt
// output: cg, trans, steprot (can be input uninitialized)
void
Problem::bodies_timestep(const float3 *forces, const float3 *torques, const int step,
		const double dt, const double t,
		int3 * & cgGridPos, float3 * & cgPos, float3 * & trans, float * & steprot,
		float3 * & linearvel, float3 * & angularvel)
{
	// Compute time step and time according to the integration scheme
	// TODO: must be done according to the integration scheme
	double dt1 = dt;
	if (step == 1)
		dt1 /= 2.0;
	double t0 = t;
	double t1 = t + dt1;

	//#define _DEBUG_OBJ_FORCES_
	bool there_is_at_least_one_chrono_body = false;
	// For Chrono bodies apply forces and torques
	for (size_t i = 0; i < m_bodies.size(); i++) {
		// Shortcut to body data
		MovingBodyData* mbdata = m_bodies[i];
		// Store kinematic data at the beginning of the time step
		if (step == 1)
			m_bodies_storage[i] = mbdata->kdata;
		// Restore kinematic data from the value stored at the beginning of the time step
		if (step == 2)
			mbdata->kdata = m_bodies_storage[i];
#if USE_CHRONO == 1
		// If current body has a Chrono body associated (no matter whether type moving or floating), we
		// want to copy its parameters (velocities, position, etc.) from its kdata to Chrono
		if (mbdata->object->HasBody()) {
			there_is_at_least_one_chrono_body = true;
			std::shared_ptr< ::chrono::ChBody > body = mbdata->object->GetBody();
			// For step 2 restore cg, lvel and avel to the value at the beginning of
			// the timestep
			if (step == 2) {
				body->SetPos(::chrono::ChVector<>(mbdata->kdata.crot.x, mbdata->kdata.crot.y, mbdata->kdata.crot.z));
				body->SetPos_dt(::chrono::ChVector<>(mbdata->kdata.lvel.x, mbdata->kdata.lvel.y, mbdata->kdata.lvel.z));
				body->SetWvel_par(::chrono::ChVector<>(mbdata->kdata.avel.x, mbdata->kdata.avel.y, mbdata->kdata.avel.z));
				body->SetRot(mbdata->kdata.orientation.ToChQuaternion());
			}

			body->Empty_forces_accumulators();
			body->Accumulate_force(::chrono::ChVector<>(forces[i].x, forces[i].y, forces[i].z), body->GetPos(), false);
			body->Accumulate_torque(::chrono::ChVector<>(torques[i].x, torques[i].y, torques[i].z), false);


			if (false) {
				cout << "Before dWorldStep, object " << i << "\tt = " << t << "\tdt = " << dt <<"\n";
				//mbdata->object->ODEPrintInformation(false);
				printf("   F:	%e\t%e\t%e\n", forces[i].x, forces[i].y, forces[i].z);
				printf("   T:	%e\t%e\t%e\n", torques[i].x, torques[i].y, torques[i].z);
			}
		}
#endif
	}

#if USE_CHRONO == 1
	// Call Chrono solver. Should it be called only if there are floating ones?
	if (there_is_at_least_one_chrono_body) {
		m_bodies_physical_system->DoStepDynamics(dt1);
	}
#endif

	// Walk trough all moving bodies :
	// updates bodies center of rotation, linear and angular velocity and orientation
	for (size_t i = 0; i < m_bodies.size(); i++) {
		// Shortcut to MovingBodyData
		MovingBodyData* mbdata = m_bodies[i];
		// New center of rotation, linear and angular velocity and orientation
		double3 new_trans = make_double3(0.0);
		EulerParameters new_orientation, dr;
#if USE_CHRONO == 1
		// For floating bodies, new center of rotation position, linear and angular velocity
		// and new orientation have been computed by Chrono. So let's read them and copy to kdata
		if (mbdata->type == MB_FLOATING) {
			std::shared_ptr< ::chrono::ChBody > body = mbdata->object->GetBody();
			::chrono::ChVector<> vec = body->GetPos();
			const double3 new_crot = make_double3(vec.x(), vec.y(), vec.z());
			new_trans = new_crot - mbdata->kdata.crot;
			mbdata->kdata.crot = new_crot;
			vec = body->GetPos_dt();
			mbdata->kdata.lvel = make_double3(vec.x(), vec.y(), vec.z());
			vec = body->GetWvel_par();
			mbdata->kdata.avel = make_double3(vec.x(), vec.y(), vec.z());
			::chrono::ChQuaternion<> quat = body->GetRot();
			const EulerParameters new_orientation = EulerParameters(quat.e0(), quat.e1(), quat.e2(), quat.e3());
			dr = new_orientation*mbdata->kdata.orientation.Inverse();
			mbdata->kdata.orientation = new_orientation;
		}
#endif
		// If the body is not floating, the user is probably providing linear and angular velocity trough a call back
		// function.
		if (mbdata->type != MB_FLOATING) {
			const uint index = mbdata->index;
			// Get linear and angular velocities at t + dt/2.O for step 1 or t + dt for step 2
			float3 force = make_float3(0.0f);
			float3 torque = make_float3(0.0f);
			if (mbdata->type == MB_FORCES_MOVING) {
				force = forces[i];
				torque = torques[i];
			}

			moving_bodies_callback(index, mbdata->object, t0, t1, force, torque, mbdata->initial_kdata,
					mbdata->kdata, new_trans, dr);
		}

		calc_grid_and_local_pos(mbdata->kdata.crot, cgGridPos + i, cgPos + i);
		trans[i] = make_float3(new_trans);
		linearvel[i] = make_float3(mbdata->kdata.lvel);
		angularvel[i] = make_float3(mbdata->kdata.avel);

		// Compute and relative rotation respect to the beginning of time step
		float *base_addr = steprot + 9*i;
		dr.ComputeRot();
		dr.GetRotation(base_addr);

		if (false) {
			if (i == 1 && trans[i].x != 0.0) {
				cout << "After dWorldStep, object "  << i << "\tt = " << t << "\tdt = " << dt <<"\n";
			mbdata->object->BodyPrintInformation(false);
			printf("   lvel: %e\t%e\t%e\n", linearvel[i].x, linearvel[i].y, linearvel[i].z);
			printf("   avel: %e\t%e\t%e\n", angularvel[i].x, angularvel[i].y, angularvel[i].z);
			printf("    pos: %g\t%g\t%g\n", mbdata->kdata.crot.x, mbdata->kdata.crot.y, mbdata->kdata.crot.z);
			printf("   gpos: %d\t%d\t%d\n", cgGridPos[i].x, cgGridPos[i].y, cgGridPos[i].z);
			printf("   lpos: %e\t%e\t%e\n", cgPos[i].x, cgPos[i].y, cgPos[i].z);
			printf("   trans:%e\t%e\t%e\n", trans[i].x, trans[i].y, trans[i].z);
			printf("   n_ep: %e\t%e\t%e\t%e\n", mbdata->kdata.orientation(0), mbdata->kdata.orientation(1),
					mbdata->kdata.orientation(2), mbdata->kdata.orientation(3));
			printf("   dr: %e\t%e\t%e\t%e\n", dr(0), dr(1),dr(2), dr(3));
			printf("   SR:   %e\t%e\t%e\n", base_addr[0], base_addr[1], base_addr[2]);
			printf("         %e\t%e\t%e\n", base_addr[3], base_addr[4], base_addr[5]);
			printf("         %e\t%e\t%e\n\n", base_addr[6], base_addr[7], base_addr[8]);
			}
		}
	}
}

// Copy planes for upload
void
Problem::copy_planes(PlaneList& planes)
{
	return;
}

void
Problem::check_dt(void)
{
	float dt_from_sspeed = INFINITY;
	for (uint f = 0 ; f < physparams()->numFluids(); ++f) {
		float sspeed = physparams()->sscoeff[f];
		dt_from_sspeed = fmin(dt_from_sspeed, simparams()->slength/sspeed);
	}
	dt_from_sspeed *= simparams()->dtadaptfactor;

	float dt_from_gravity = sqrt(simparams()->slength/length(physparams()->gravity));
	dt_from_gravity *= simparams()->dtadaptfactor;

	float dt_from_visc = NAN;
	if (simparams()->rheologytype != INVISCID) {
		for (uint f = 0; f < physparams()->numFluids(); ++f)
			dt_from_visc = fminf(dt_from_visc, simparams()->slength*simparams()->slength/physparams()->kinematicvisc[f]);
		dt_from_visc *= 0.125f; // TODO this should be configurable
	}

	float cfl_dt = fminf(dt_from_sspeed, fminf(dt_from_gravity, dt_from_visc));

	if (simparams()->dt > cfl_dt) {
		fprintf(stderr, "WARNING: dt %g bigger than %g imposed by CFL conditions (sspeed: %g, gravity: %g, viscosity: %g)\n",
			simparams()->dt, cfl_dt,
			dt_from_sspeed, dt_from_gravity, dt_from_visc);
	} else if (!simparams()->dt) { // dt wasn't set
			simparams()->dt = cfl_dt;
			printf("setting dt = %g from CFL conditions (soundspeed: %g, gravity: %g, viscosity: %g)\n",
				simparams()->dt,
				dt_from_sspeed, dt_from_gravity, dt_from_visc);
	} else {
			printf("dt = %g (CFL conditions from soundspeed: %g, from gravity %g, from viscosity %g)\n",
				simparams()->dt,
				dt_from_sspeed, dt_from_gravity, dt_from_visc);
	}

}

void
Problem::check_neiblistsize(void)
{
	// kernel radius times smoothing factor, rounded to the next integer
	double r = simparams()->sfactor*simparams()->kernelradius;
	r = ceil(r);

	// volumes are computed using a coefficient which is sligthly more than PI
#define PI_PLUS_EPS 3.2
	double vol = 4*PI_PLUS_EPS*r*r*r/3;
	// and rounded up
	vol = ceil(vol);

	// neibsboundpos is obtained rounding up the volume to the next
	// multiple of 32
	uint neiblistsize = round_up((uint)vol, 32U);

	// more in general, it's possible to have different particle densities for the
	// boundaries even with other boundary conditions. we do not have a universal
	// parameter that marks the inter-particle distance for boundary particles,
	// although we know that r0 is normally used for this too.
	// TODO FIXME when the double meaning of r0 as inter-particle distance for
	// boundary particles and as fluid-boundary distance is split into separate
	// variables, the inter-particle distance should be used in the next formula

	// The formula we use is based on the following:
	// 1. a half-sphere has (3/2) pi r^3 particle
	// 2. a full circle has pi (r/q)^2 particles, if q is the ratio beween
	//   the inter-particle distance on the full circle and the inter-particle
	//   distance used in the fluid
	// * the number of neighbors that are seen by a particle which is near
	//   a boundary plane with q*dp interparticle-distance is augmented the number
	//   in 2. over the number in 1., giving (3/2) (1/q)^2 (1/r)
	// * of course this does not affect the entire neighborhood, but only the part
	//   which is close to a boundary, which we estimate to be at most 2/3rds of
	//   the neighborhood, which cancels with the (3/2) factor
	//   TODO check if we should assume 7/8ths instead (particle near vertex
	//   only has 1/8th of a sphere in the fluid, the rest is all boundaries).
	double qq = m_deltap/physparams()->r0; // 1/q
	// double ratio = fmax((21*qq*qq)/(16*r), 1.0); // if we assume 7/8
	double ratio = fmax((qq*qq)/r, 1.0); // only use this if it gives us _more_ particles
	// increase neib list size as appropriate
	neiblistsize = (uint)ceil(ratio*neiblistsize);
	// round up to multiple of 32
	neiblistsize = round_up(neiblistsize, 32U);
	uint neibboundpos = neiblistsize - 1;

	// with semi-analytical boundaries, boundary particles
	// are doubled, so we expand by a factor of 1.5,
	// again rounding up
	// TODO: optimize the 1.5 factor
	if (simparams()->boundarytype == SA_BOUNDARY)
		neiblistsize = round_up(3*neiblistsize/2, 32U);

	// if the neib list was user-set, check against computed minimum
	if (simparams()->neiblistsize) {
		if (simparams()->neiblistsize < neiblistsize) {
			cerr << "WARNING: problem-set neib list size too low! " <<
				simparams()->neiblistsize << "<" << neiblistsize << "\n";
		} else {
			cout << "Using problem-set neib list size " << simparams()->neiblistsize <<
					" (safe computed value was " << neiblistsize << ")\n";
		}
	} else {
		cout << "Using computed max neib list size " << neiblistsize << "\n";
		simparams()->neiblistsize = neiblistsize;
	}

	// if the neib bound pos was user-set, check against computed minimum
	if (simparams()->neibboundpos) {
		if (simparams()->neibboundpos < neibboundpos) {
			cerr << "WARNING: problem-set neib bound pos too low! " <<
					simparams()->neibboundpos << "<" << neibboundpos << "\n";
		} else {
			cout << "Using problem-set neib bound pos " << simparams()->neibboundpos <<
					" (safe computed value was " << neibboundpos << ")\n";
		}
	} else {
		cout << "Using computed neib bound pos " << neibboundpos << "\n";
		simparams()->neibboundpos = neibboundpos;
	}
}

float
Problem::hydrostatic_density(float h, int i) const
{
	float density = atrest_density(i);

	if (h > 0) {
		float g = fabsf(length(physparams()->gravity));
		// TODO g*rho0*h/B could be simplified to g*h*gamma/(c0*c0)
		density = pow(g*physparams()->rho0[i]*h/physparams()->bcoeff[i] + 1,1/physparams()->gammacoeff[i])-1.0;

		}

	return density;
}

float
Problem::density_for_pressure(float P, int i) const
{
	return  pow(P/physparams()->bcoeff[i] + 1,1/physparams()->gammacoeff[i])-1.0;
}

float
Problem::soundspeed(float rho_tilde, int i) const
{
	const float rho_ratio = rho_tilde + 1;

    return physparams()->sscoeff[i]*pow(rho_ratio, physparams()->sspowercoeff[i]);
}

float
Problem::pressure(float rho_tilde, int i) const
{
	const float rho_ratio = rho_tilde + 1;

	return physparams()->bcoeff[i]*(pow(rho_ratio, physparams()->gammacoeff[i]) - 1);
}

float
Problem::physical_density( float rho_tilde, int i) const
{
	return (rho_tilde + 1)*physparams()->rho0[i];
}

float
Problem::numerical_density( float rho, int i) const
{
	return rho/physparams()->rho0[i] - 1;
}

void
Problem::add_gage(double3 const& pt)
{
	simparams()->gage.push_back(make_double4(pt.x, pt.y, 0., pt.z));
}

plane_t
Problem::implicit_plane(double4 const& p)
{
	const double4 midPoint = make_double4(m_origin + m_size/2, 1.0);

	plane_t plane;
	const double norm = length3(p);
	const double3 normal = as_double3(p)/norm;
	plane.normal = make_float3(normal);

	/* For the plane point, we pick the one closest to the center of the domain
	 * TODO find a better logic ? */

	const double midDist = dot(midPoint, p)/norm;
	double3 planePoint = as_double3(midPoint) - midDist*normal;

	calc_grid_and_local_pos(planePoint, &plane.gridPos, &plane.pos);

	return plane;
}

plane_t
Problem::make_plane(Point const& pt, Vector const& normal)
{
	plane_t plane;

	plane.normal = make_float3(normal);
	calc_grid_and_local_pos(make_double3(pt), &plane.gridPos, &plane.pos);

	return plane;
}

string const&
Problem::create_problem_dir(void)
{
	// if no data save directory was specified, default to a name
	// composed of problem name followed by date and time
	if (m_problem_dir.empty()) {
		time_t  rawtime;
		char	time_str[18];

		time(&rawtime);
		strftime(time_str, 18, "_%Y-%m-%dT%Hh%M", localtime(&rawtime));
		time_str[17] = '\0';
		// if "./tests/" doesn't exist yet...
		mkdir("./tests/", S_IRWXU | S_IRWXG | S_IRWXO);
		m_problem_dir = "./tests/" + m_name + string(time_str);
	}

	// TODO it should be possible to specify a directory with %-like
	// replaceable strings, such as %{problem} => problem name,
	// %{time} => launch time, etc.

	mkdir(m_problem_dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);

	return m_problem_dir;
}

void
Problem::add_writer(WriterType wt, double freq)
{
	m_writers.push_back(make_pair(wt, freq));
}


// override in problems where you want to save
// at specific times regardless of standard conditions
bool
Problem::need_write(double t) const
{
	return false;
}

// overridden in subclasses if they want to write custom stuff
// using the CALLBACKWRITER
void
Problem::writer_callback(CallbackWriter *,
	uint numParts, BufferList const&, uint node_offset, double t,
	const bool testpoints) const
{
	fprintf(stderr, "WARNING: CallbackWriter is being used, but writer_callback wasn't implemented\n");
}


// is the simulation finished at the given time?
bool
Problem::finished(double t) const
{
	double tend(simparams()->tend);
	return tend && (t > tend);
}


float3
Problem::g_callback(const double t)
{
	throw std::runtime_error("default g_callback invoked! did you forget to override g_callback(double)");
}


// Fill the device map with "devnums" (*global* device ids) in range [0..numDevices[.
// Default algorithm: split along the longest axis
void Problem::fillDeviceMap()
{
	fillDeviceMapByAxis(LONGEST_AXIS);
}

// partition by splitting the cells according to their linearized hash.
void Problem::fillDeviceMapByCellHash()
{
	uint cells_per_device = gdata->nGridCells / gdata->totDevices;
	for (uint i=0; i < gdata->nGridCells; i++)
		// guaranteed to fit in a devcount_t due to how it's computed
		gdata->s_hDeviceMap[i] = devcount_t(min( int(i/cells_per_device), gdata->totDevices-1));
}

// partition by splitting along the specified axis
void Problem::fillDeviceMapByAxis(SplitAxis preferred_split_axis)
{
	// select the longest axis
	if (preferred_split_axis == LONGEST_AXIS) {
		if (	gdata->worldSize.x >= gdata->worldSize.y &&
				gdata->worldSize.x >= gdata->worldSize.z)
			preferred_split_axis = X_AXIS;
		else
		if (	gdata->worldSize.y >= gdata->worldSize.z)
			preferred_split_axis = Y_AXIS;
		else
			preferred_split_axis = Z_AXIS;
	}
	uint cells_per_split_axis = 0;
	switch (preferred_split_axis) {
		case X_AXIS:
			cells_per_split_axis = gdata->gridSize.x;
			break;
		case Y_AXIS:
			cells_per_split_axis = gdata->gridSize.y;
			break;
		case Z_AXIS:
			cells_per_split_axis = gdata->gridSize.z;
			break;
	}

	// Check that we have enough cells along the split axis. This check should
	// be performed in all split algorithms
	if (cells_per_split_axis / (double) gdata->totDevices < 3.0)
		throw runtime_error ("FATAL: not enough cells along the split axis. Aborting.\n");

	uint cells_per_device_per_split_axis = (uint)round(cells_per_split_axis / (double)gdata->totDevices);

	/*
	printf("Splitting domain along axis %s, %u cells per part\n",
		(preferred_split_axis == X_AXIS ? "X" : (preferred_split_axis == Y_AXIS ? "Y" : "Z") ), cells_per_device_per_split_axis);
	*/
	for (uint cx = 0; cx < gdata->gridSize.x; cx++)
		for (uint cy = 0; cy < gdata->gridSize.y; cy++)
			for (uint cz = 0; cz < gdata->gridSize.z; cz++) {
				uint axis_coordinate;
				switch (preferred_split_axis) {
					case X_AXIS: axis_coordinate = cx; break;
					case Y_AXIS: axis_coordinate = cy; break;
					case Z_AXIS: axis_coordinate = cz; break;
				}
				// everything is just a preparation for the following line
				devcount_t dstDevice = devcount_t(axis_coordinate / cells_per_device_per_split_axis);
				// handle the case when cells_per_split_axis multiplies cells_per_split_axis
				dstDevice = (devcount_t)min(int(dstDevice), gdata->totDevices - 1);
				// compute cell address
				uint cellLinearHash = gdata->calcGridHashHost(cx, cy, cz);
				// assign it
				gdata->s_hDeviceMap[cellLinearHash] = dstDevice;
			}
}

// Like fillDeviceMapByAxis(), but splits are proportional to the contained fluid particles
void Problem::fillDeviceMapByAxisBalanced(SplitAxis preferred_split_axis)
{
	// Select the longest axis
	if (preferred_split_axis == LONGEST_AXIS) {
		if (	gdata->worldSize.x >= gdata->worldSize.y &&
				gdata->worldSize.x >= gdata->worldSize.z)
			preferred_split_axis = X_AXIS;
		else
		if (	gdata->worldSize.y >= gdata->worldSize.z)
			preferred_split_axis = Y_AXIS;
		else
			preferred_split_axis = Z_AXIS;
	}

	// Set some aux variables - axis 1 is the split axis
	uint cells_per_axis1 = 0;
	uint cells_per_axis2 = 0;
	uint cells_per_axis3 = 0;
	uint cx = 0, cy = 0, cz = 0; // cell coordinates
	uint *c1, *c2, *c3; // abstract from cell coordinates
	uint *axisParticleCounter = NULL;
	switch (preferred_split_axis) {
		case X_AXIS:
			cells_per_axis1 = gdata->gridSize.x;
			cells_per_axis2 = gdata->gridSize.y;
			cells_per_axis3 = gdata->gridSize.z;
			c1 = &cx;
			c2 = &cy;
			c3 = &cz;
			axisParticleCounter = gdata->s_hPartsPerSliceAlongX;
			break;
		case Y_AXIS:
			cells_per_axis1 = gdata->gridSize.y;
			cells_per_axis2 = gdata->gridSize.x;
			cells_per_axis3 = gdata->gridSize.z;
			c1 = &cy;
			c2 = &cx;
			c3 = &cz;
			axisParticleCounter = gdata->s_hPartsPerSliceAlongY;
			break;
		case Z_AXIS:
			cells_per_axis1 = gdata->gridSize.z;
			cells_per_axis2 = gdata->gridSize.x;
			cells_per_axis3 = gdata->gridSize.y;
			c1 = &cz;
			c2 = &cx;
			c3 = &cy;
			axisParticleCounter = gdata->s_hPartsPerSliceAlongZ;
			break;
	}

	// Check that we have enough cells along the split axis. This check should
	// be performed in all split algorithms
	if (cells_per_axis1 / (double) gdata->totDevices < 3.0)
		throw runtime_error ("FATAL: not enough cells along the split axis. Aborting.\n");

	// Compute ideal split values
	const uint particles_per_device = gdata->totParticles / gdata->totDevices;
	const uint particles_per_slice = gdata->totParticles / cells_per_axis1;

	// If a device has "almost" particles_per_device particles, next slice will assign particles_per_slice more
	// particles and make it "overflow" the ideal number; we will instead stop before, at this threshold
	const uint particles_per_device_threshold = particles_per_device - (particles_per_slice / 2);

	// printf("Splitting domain along axis %s, ~%u particles per device\n",
	//	(preferred_split_axis == X_AXIS ? "X" : (preferred_split_axis == Y_AXIS ? "Y" : "Z") ), (uint)particles_per_device);

	// We need at least 3 cells per device, regardless the distribution of fluid; so we track the
	// remaining cells which need to be "reserved" for device numbers yet to be analyzed, excluding
	// the first.
	uint reserved_cells =  3 * (gdata->totDevices - 1);

	// We will iterate on the cells and increase the current device number
	uint currentDevice = 0;
	uint currentDeviceParticles = 0;

	// NOTE: not using "*cx++" since post-increment has precedence over deference
	for (*c1 = 0; *c1 < cells_per_axis1; (*c1)++) {
		// We must increase the current device only if:
		// 1. This is not the last device (i.e. should always be currentDevice < totDevices), and
		// 2a. we got enough particles in previous iteration, or
		// 2b. we reached the reserved cells (thus, we must leave them for next devices)
		if ( (currentDevice < gdata->totDevices - 1) &&
			(currentDeviceParticles >= particles_per_device_threshold ||
			*c1 >= cells_per_axis1 - reserved_cells - 1) ) {
			// switch to next device: reset counter,
			currentDeviceParticles = 0;
			// increase device,
			currentDevice++;
			// update reserved_cells (minus mine)
			reserved_cells -= 3;
		}

		// add particles in current slice
		currentDeviceParticles += axisParticleCounter[ *c1 ];

		// assign all the cells of the current slice to current device
		for (*c2 = 0; *c2 < cells_per_axis2; (*c2)++)
			for (*c3 = 0; *c3 < cells_per_axis3; (*c3)++) {
				// we are actually using c1, c2, c3 in a proper order
				const uint cellLinearHash = gdata->calcGridHashHost(cx, cy, cz);
				// assign it
				gdata->s_hDeviceMap[cellLinearHash] = currentDevice;
			}
	} // iterate on split axis
}

void Problem::fillDeviceMapByEquation()
{
	// 1st equation: diagonal plane. (x+y+z)=coeff
	//uint longest_grid_size = max ( max( gdata->gridSize.x, gdata->gridSize.y), gdata->gridSize.z );
	uint coeff = (gdata->gridSize.x + gdata->gridSize.y + gdata->gridSize.z) / gdata->totDevices;
	// 2nd equation: sphere. Sqrt(cx²+cy²+cz²)=radius
	uint diagonal = (uint) sqrt(	gdata->gridSize.x * gdata->gridSize.x +
									gdata->gridSize.y * gdata->gridSize.y +
									gdata->gridSize.z * gdata->gridSize.z) / 2;
	uint radius_part = diagonal /  gdata->totDevices;
	for (uint cx = 0; cx < gdata->gridSize.x; cx++)
		for (uint cy = 0; cy < gdata->gridSize.y; cy++)
			for (uint cz = 0; cz < gdata->gridSize.z; cz++) {
				uint dstDevice;
				// 1st equation: rough oblique plane split --
				dstDevice = (cx + cy + cz) / coeff;
				// -- end of 1st eq.
				// 2nd equation: spheres --
				//uint distance_from_origin = (uint) sqrt( cx * cx + cy * cy + cz * cz);
				// comparing directly the square would be more efficient but could require long uints
				//dstDevice = distance_from_origin / radius_part;
				// -- end of 2nd eq.
				// handle special cases at the edge
				dstDevice = min(int(dstDevice), gdata->totDevices - 1);
				// compute cell address
				uint cellLinearHash = gdata->calcGridHashHost(cx, cy, cz);
				// assign it
				gdata->s_hDeviceMap[cellLinearHash] = (uchar)dstDevice;
			}
}

// Partition by performing the splitting the domain in the specified number of slices for each axis.
// Values must be > 0. The number of devices will be the product of the input values.
// This is not meant to be called directly by a problem since the number of splits (and thus the devices)
// would be hardocded. A wrapper method (like fillDeviceMapByRegularGrid) can provide an algorithm to
// properly factorize a given number of GPUs in 2 or 3 values.
void Problem::fillDeviceMapByAxesSplits(uint Xslices, uint Yslices, uint Zslices)
{
	// is any of these zero?
	if (Xslices * Yslices * Zslices == 0)
		printf("WARNING: fillDeviceMapByAxesSplits() called with zero values, using 1 instead");

	if (Xslices == 0) Xslices = 1;
	if (Yslices == 0) Yslices = 1;
	if (Zslices == 0) Zslices = 1;

	// divide and round
	uint devSizeCellsX = (gdata->gridSize.x + Xslices - 1) / Xslices ;
	uint devSizeCellsY = (gdata->gridSize.y + Yslices - 1) / Yslices ;
	uint devSizeCellsZ = (gdata->gridSize.z + Zslices - 1) / Zslices ;

	// iterate on all cells
	for (uint cx = 0; cx < gdata->gridSize.x; cx++)
			for (uint cy = 0; cy < gdata->gridSize.y; cy++)
				for (uint cz = 0; cz < gdata->gridSize.z; cz++) {

				// where are we in the 3D grid of devices?
				uint whichDevCoordX = (cx / devSizeCellsX);
				uint whichDevCoordY = (cy / devSizeCellsY);
				uint whichDevCoordZ = (cz / devSizeCellsZ);

				// round if needed
				if (whichDevCoordX == Xslices) whichDevCoordX--;
				if (whichDevCoordY == Yslices) whichDevCoordY--;
				if (whichDevCoordZ == Zslices) whichDevCoordZ--;

				// compute dest device
				uint dstDevice = whichDevCoordZ * Yslices * Xslices + whichDevCoordY * Xslices + whichDevCoordX;
				// compute cell address
				uint cellLinearHash = gdata->calcGridHashHost(cx, cy, cz);
				// assign it
				gdata->s_hDeviceMap[cellLinearHash] = (uchar)dstDevice;
			}
}

// Wrapper for fillDeviceMapByAxesSplits() computing the number of cuts along each axis.
// WARNING: assumes the total number of devices is divided by a combination of 2, 3 and 5
void Problem::fillDeviceMapByRegularGrid()
{
	float Xsize = gdata->worldSize.x;
	float Ysize = gdata->worldSize.y;
	float Zsize = gdata->worldSize.z;
	devcount_t cutsX = 1;
	devcount_t cutsY = 1;
	devcount_t cutsZ = 1;
	devcount_t remaining_factors = gdata->totDevices;

	// define the product of non-zero cuts to keep track of current number of parallelepipeds
//#define NZ_PRODUCT	((cutsX > 0? cutsX : 1) * (cutsY > 0? cutsY : 1) * (cutsZ > 0? cutsZ : 1))

	while (cutsX * cutsY * cutsZ < gdata->totDevices) {
		devcount_t factor = 1;
		// choose the highest factor among 2, 3 and 5 which divides remaining_factors
		if (remaining_factors % 5 == 0) factor = 5; else
		if (remaining_factors % 3 == 0) factor = 3; else
		if (remaining_factors % 2 == 0) factor = 2; else {
			factor = remaining_factors;
			printf("WARNING: splitting by regular grid but %u is not divided by 2,3,5!\n", remaining_factors);
		}
		// choose the longest axis to split along
		if (Xsize >= Ysize && Xsize >= Zsize) {
			Xsize /= factor;
			cutsX *= factor;
		} else
		if (Ysize >= Xsize && Ysize >= Zsize) {
			Ysize /= factor;
			cutsY *= factor;
		} else {
			Zsize /= factor;
			cutsZ *= factor;
		}
	}

	// should always hold, but double check for bugs
	if (cutsX * cutsY * cutsZ != gdata->totDevices)
		printf("WARNING: splitting by regular grid but final distribution (%u, %u, %u) does not produce %u parallelepipeds!\n",
			cutsX, cutsY, cutsZ, gdata->totDevices);

	fillDeviceMapByAxesSplits(cutsX, cutsY, cutsZ);
}



uint
Problem::max_parts(uint numParts)
{
	if (!(simparams()->simflags & ENABLE_INLET_OUTLET))
		return numParts;

	// we assume that we can't have more particles than by filling the whole domain:
	// if the user knows how many particles there are going to be he should implement
	// his own version of this function
	double3 range = get_worldsize();
	range /= m_deltap; // regular fill
	uint wparts = max(range.x, double(1))*max(range.y, double(1))*max(range.z, double(1));
	printf("  estimating %u particles to fill the world\n", wparts);

	return wparts;
}

// This function computes the density diffusion coefficient.
//
// For the Ferrari diffusion this coefficient is based on a lenght-scale. The formula for the coefficient
// is L/(1000 * deltap), see Mayrhofer et al., 2013. If the length scale is not set then the ferrari coefficient
// will be taken as it is, regardless of whether it is set or not (default value = 0)
//
// The Brezzi coefficient is 0 by default.
//
// The Colagrossi coefficient is 0.1 by default and is pre-multiplied with 2*h in order to avoid this
// multiplication during the kernel runs.
void
Problem::calculateDensityDiffusionCoefficient()
{
	switch (simparams()->densitydiffusiontype)
	{
	case FERRARI:
		if (isnan(simparams()->densityDiffCoeff)) {
			if (isnan(simparams()->ferrariLengthScale)) {
				simparams()->densityDiffCoeff = 0.0f;
				printf("Ferrari diffusion coefficient: %e (default value, disabled)\n", simparams()->densityDiffCoeff);
				fprintf(stderr, "WARNING: Ferrari density diffusion enabled, but no coefficient or length scale given!\n");
				break;
			}
			else {
				simparams()->densityDiffCoeff = simparams()->ferrariLengthScale*1e-3f/m_deltap;
				printf("Ferrari diffusion coefficient: %e (computed from length scale: %e)\n", simparams()->densityDiffCoeff, simparams()->ferrariLengthScale);
				break;
			}
		}
		printf("Ferrari diffusion coefficient: %e\n", simparams()->densityDiffCoeff);
		break;
	case BREZZI:
		if (isnan(simparams()->densityDiffCoeff)) {
			simparams()->densityDiffCoeff = 0.0f;
			printf("Brezzi diffusion coefficient: %e (default value, disabled)\n", simparams()->densityDiffCoeff);
			fprintf(stderr, "WARNING: Brezzi density diffusion enabled, but no coefficient given!\n");
			break;
		}
		printf("Brezzi diffusion coefficient = %e\n", simparams()->densityDiffCoeff);
		break;
	case COLAGROSSI:
		if (isnan(simparams()->densityDiffCoeff)) {
			simparams()->densityDiffCoeff = 0.1f;
			printf("Colagrossi diffusion coefficient: %e (default value)\n", simparams()->densityDiffCoeff);
			fprintf(stderr, "WARNING: Colagrossi density diffusion enabled, but no coefficient given!\n");
		}
		else
			printf("Colagrossi diffusion coefficient ξ = %e\n", simparams()->densityDiffCoeff);
		// pre-multiply xi with 2*h
		simparams()->densityDiffCoeff *= 2.0f*simparams()->slength;
		break;
	default:

		break;
	}
	return;
}


/*! Compute grid and cell size from the kernel influence radius
 * The number of cell is obtained as the ratio between the domain size and the
 * influence radius, rounded down to the closest integer.
 * The reason for rounding down is that we want the cell size to be no smaller
 * than the influence radius, to guarantee that all neighbors of a particle are
 * found at most one cell away in each direction.
 */
void
Problem::set_grid_params(void)
{
	/* When using periodicity, it's important that the world size in the periodic
	 * direction is an exact multiple of the deltap: if this is not the case,
	 * fluid filling might use an effective inter-particle distance which is
	 * “significantly” different from deltap, which would lead particles near
	 * periodic boundaries to have distance _exactly_ deltap across the boundary,
	 * but “significantly” different on the same side. While this in general would not
	 * be extremely important, it can have a noticeable effect at the beginning of the
	 * simulation, when particles are distributed quite regularly and the difference
	 * between effective (inner) distance and cross-particle distance can create
	 * a rather annoying discontinuity.
	 * So warn if m_size.{x,y,z} is not a multiple of deltap in case of periodicity.
	 * TODO FIXME this would not be needed if filling was made taking into account
	 * periodicity and spaced particles accordingly.
	 */
	if (simparams()->periodicbound & PERIODIC_X && !is_multiple(m_size.x, m_deltap))
		fprintf(stderr, "WARNING: problem is periodic in X, but X world size %.9g is not a multiple of deltap (%.g)\n",
			m_size.x, m_deltap);
	if (simparams()->periodicbound & PERIODIC_Y && !is_multiple(m_size.y, m_deltap))
		fprintf(stderr, "WARNING: problem is periodic in Y, but Y world size %.9g is not a multiple of deltap (%.g)\n",
			m_size.y, m_deltap);
	if (simparams()->periodicbound & PERIODIC_Z && !is_multiple(m_size.z, m_deltap))
		fprintf(stderr, "WARNING: problem is periodic in X, but Z world size %.9g is not a multiple of deltap (%.g)\n",
			m_size.z, m_deltap);

	const double influenceRadius = simparams()->influenceRadius;
	const double nlInfluenceRadius = simparams()->nlInfluenceRadius;

	if (nlInfluenceRadius < influenceRadius) {
		stringstream ss;
		ss << "neighbor search radius " << nlInfluenceRadius <<
			" < kernel influence radius " << influenceRadius;
		throw runtime_error(ss.str());
	}

	// with semi-analytical boundaries, we want a cell size which is
	// deltap/2 + the usual influence radius
	double cellSide = nlInfluenceRadius;
	if (simparams()->boundarytype == SA_BOUNDARY)
		cellSide += m_deltap/2.0f;

	m_gridsize.x = (uint)floor(m_size.x / cellSide);
	m_gridsize.y = (uint)floor(m_size.y / cellSide);
	m_gridsize.z = (uint)floor(m_size.z / cellSide);

	// While trying to run a simulation at very low resolution, the user might
	// set a deltap so large that cellSide is bigger than m_size.{x,y,z}, resulting
	// in a corresponding gridsize of 0. Check for this case (by checking if any
	// of the gridsize components are zero) and throw.

	if (!m_gridsize.x || !m_gridsize.y || !m_gridsize.z) {
		stringstream ss;
		ss << "resolution " << simparams()->slength << " is too low! Resulting grid size would be "
			<< m_gridsize;
		throw runtime_error(ss.str());
	}

	m_cellsize.x = m_size.x / m_gridsize.x;
	m_cellsize.y = m_size.y / m_gridsize.y;
	m_cellsize.z = m_size.z / m_gridsize.z;

	/*
	printf("set_grid_params->t:\n");
	printf("Domain size\t: (%f, %f, %f)\n", m_size.x, m_size.y, m_size.z);
	*/
	printf("Influence radius / neighbor search radius / expected cell side\t: %g / %g / %g\n", influenceRadius, nlInfluenceRadius, cellSide);
	/*
	printf("Grid   size\t: (%d, %d, %d)\n", m_gridsize.x, m_gridsize.y, m_gridsize.z);
	printf("Cell   size\t: (%f, %f, %f)\n", m_cellsize.x, m_cellsize.y, m_cellsize.z);
	printf("       delta\t: (%.2f%%, %.2f%%, %.2f%%)\n",
		(m_cellsize.x - cellSide)*100/cellSide,
		(m_cellsize.y - cellSide)*100/cellSide,
		(m_cellsize.z - cellSide)*100/cellSide);
	*/
}


// Compute position in uniform grid (clamping to edges)
int3
Problem::calc_grid_pos(const Point& pos) const
{
	int3 gridPos;
	gridPos.x = (int)floor((pos(0) - m_origin.x) / m_cellsize.x);
	gridPos.y = (int)floor((pos(1) - m_origin.y) / m_cellsize.y);
	gridPos.z = (int)floor((pos(2) - m_origin.z) / m_cellsize.z);
	gridPos.x = min(max(0, gridPos.x), int(m_gridsize.x-1));
	gridPos.y = min(max(0, gridPos.y), int(m_gridsize.y-1));
	gridPos.z = min(max(0, gridPos.z), int(m_gridsize.z-1));

	return gridPos;
}

/// Compute the uniform grid components of a vector
int3
Problem::calc_grid_offset(double3 const& vec) const
{
	int3 gridOff;
	gridOff = make_int3(floor(vec/m_cellsize));

	return gridOff;
}

/// Compute the local (fractional grid cell) components of a vector,
/// given the vector and its grid offset
double3
Problem::calc_local_offset(double3 const& vec, int3 const& gridOff) const
{
	return vec - (make_double3(gridOff) + 0.5)*m_cellsize;
}


// Compute address in grid from position
uint
Problem::calc_grid_hash(int3 gridPos) const
{
	return gridPos.COORD3 * m_gridsize.COORD2 * m_gridsize.COORD1 + gridPos.COORD2 * m_gridsize.COORD1 + gridPos.COORD1;
}


void
Problem::calc_localpos_and_hash(const Point& pos, const particleinfo& info, float4& localpos, hashKey& hash) const
{
	static bool warned_out_of_bounds = false;
	// check if the particle is actually inside the domain
	if (!warned_out_of_bounds &&
		(pos(0) < m_origin.x || pos(0) > m_origin.x + m_size.x ||
		 pos(1) < m_origin.y || pos(1) > m_origin.y + m_size.y ||
		 pos(2) < m_origin.z || pos(2) > m_origin.z + m_size.z))
	{
		const uint pid = id(info);
		stringstream errmsg;
		errmsg << "Particle " << pid << " position " << make_double4(pos)
			<< " is outside of the domain " << m_origin << "--" << (m_origin+m_size) ;
		warned_out_of_bounds = true;
		if (gdata->debug.validate_init_positions)
			throw std::out_of_range(errmsg.str());
		else
			cerr << errmsg.str() << endl;
	}

	int3 gridPos = calc_grid_pos(pos);

	// automatically choose between long hash (cellHash + particleId) and short hash (cellHash)
	hash = calc_grid_hash(gridPos);

	localpos.x = float(pos(0) - m_origin.x - (gridPos.x + 0.5)*m_cellsize.x);
	localpos.y = float(pos(1) - m_origin.y - (gridPos.y + 0.5)*m_cellsize.y);
	localpos.z = float(pos(2) - m_origin.z - (gridPos.z + 0.5)*m_cellsize.z);
	localpos.w = float(pos(3));
}

/* Initialize the particle volumes from their masses and densities. */
void
Problem::init_volume(BufferList &buffers, uint numParticles)
{
	const float4 *pos = buffers.getConstData<BUFFER_POS>();
	const float4 *vel = buffers.getConstData<BUFFER_VEL>();
	float4 *vol = buffers.getData<BUFFER_VOLUME>();

	const particleinfo *info = buffers.getConstData<BUFFER_INFO>();

	for (uint i = 0; i < numParticles; ++i) {
		float4 pvol;
		// .x: initial volume, .w current volume.
		// at the beginning they are both equal to mass/density
		pvol.x = pvol.w = pos[i].w/physical_density(vel[i].w,fluid_num(info[i]));
		// .y is the log of current/initial
		pvol.y = 0;
		// .z is unused, set to zero
		pvol.z = 0;

		vol[i] = pvol;
	}
}

/* Initialize the particle internal energy. */
void
Problem::init_internal_energy(BufferList &buffers, uint numParticles)
{
	float *int_eng = buffers.getData<BUFFER_INTERNAL_ENERGY>();

	for (uint i = 0; i < numParticles; ++i) {
		int_eng[i]=0;
	}

}

/* Default initialization for k and epsilon  */
void
Problem::init_keps(BufferList &buffers, uint numParticles)
{
	float *k = buffers.getData<BUFFER_TKE>();
	float *e = buffers.getData<BUFFER_EPSILON>();

	const float Lm = fmax(2*m_deltap, 1e-5f);
	const float k0 = pow(0.002f*physparams()->sscoeff[0], 2);
	const float e0 = 0.16f*pow(k0, 1.5f)/Lm;

	for (uint i = 0; i < numParticles; i++) {
		k[i] = k0;
		e[i] = e0;
	}
}

/*!
 * Initialize eddy viscosity from k and epsilon
 * TODO this is now called whenever the user selects a(n actual) turbulence
 * mode, but it only does something in the KEPSILON case
 */
void
Problem::init_turbvisc(BufferList &buffers, uint numParticles)
{
	if (simparams()->turbmodel != KEPSILON)
		return;

	const float *k = buffers.getConstData<BUFFER_TKE>();
	const float *e = buffers.getConstData<BUFFER_EPSILON>();
	float *turbVisc = buffers.getData<BUFFER_TURBVISC>();

	for (uint i = 0; i < numParticles; ++i) {
		float ki = k[i];
		float ei = e[i];
		turbVisc[i] = 0.9*ki*ki/ei;
	}

}


void
Problem::imposeBoundaryConditionHost(
			BufferList&		bufwrite,
			BufferList const&	bufread,
					uint*			IOwaterdepth,
			const	float			t,
			const	uint			numParticles,
			const	uint			numOpenBoundaries,
			const	uint			particleRangeEnd)
{
	fprintf(stderr, "WARNING: open boundaries are present, but imposeBoundaryCondtionHost was not implemented\n");
	return;
}

void Problem::imposeForcedMovingObjects(
			float3	&gravityCenters,
			float3	&translations,
			float*	rotationMatrices,
	const	uint	ob,
	const	double	t,
	const	float	dt)
{
	// not implemented
	return;
}

void Problem::calcPrivate(
	flag_t options,
	BufferList const& bufread,
	BufferList & bufwrite,
	uint const *cellStart,
	uint numParticles,
	uint particleRangeEnd,
	uint deviceIndex,
	const GlobalData * const gdata)
{
	throw invalid_argument("CALC_PRIVATE requested, but calcPrivate() not implemented in problem");
}

std::string Problem::get_private_name(flag_t buffer) const
{
	switch (buffer) {
	case BUFFER_PRIVATE:
		return "Private";
	case BUFFER_PRIVATE2:
		return "Private2";
	case BUFFER_PRIVATE4:
		return "Private4";
	default:
		/* This shouldn't happen */
		return "Buffer" + to_string(buffer);
	}
}

void Problem::PlaneCut(PointVect& points, const double a, const double b,
			const double c, const double d)
{
	PointVect new_points;
	new_points.reserve(points.size());
	//const double norm_factor = sqrt(a*a + b*b + c*c);
	for (uint i = 0; i < points.size(); i++) {
		const Point & p = points[i];
		const double dist = a*p(0) + b*p(1) + c*p(2) + d;

		if (dist >= 0)
			new_points.push_back(p);
	}

	points.clear();
	points = new_points;
}
