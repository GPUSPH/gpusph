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
 * Core Problem class implementation
 */

#include <sstream>
#include <stdexcept>
#include <string>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

// For the automatic name determination
#include <typeinfo>
#include <cxxabi.h>

// shared_ptr
#include <memory>

#include "ProblemCore.h"
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
#include "chrono/solver/ChIterativeSolverLS.h"
#include "chrono/solver/ChDirectSolverLS.h"
#include "chrono/fea/ChNodeFEAxyzD.h"
#include "chrono/fea/ChLinkPointFrame.h"
#include "chrono/fea/ChLinkDirFrame.h"
#endif

// Enable to get/set envelop and margin (mainly debug)
/*
#include "chrono_select.opt"
#if USE_CHRONO == 1
#include "chrono/collision/ChCCollisionModel.h"
#endif
*/

using namespace std;

ProblemCore::ProblemCore(GlobalData *_gdata) :
	m_problem_dir(_gdata->clOptions->dir),
	m_dem(NULL),
	m_physparams(NULL),
	m_simframework(NULL),
	m_size(make_double3(NAN, NAN, NAN)),
	m_origin(make_double3(NAN, NAN, NAN)),
	m_deltap(NAN),
	m_hydrostaticFilling(true),
	m_waterLevel(NAN),
	gdata(_gdata),
	m_options(_gdata->clOptions),
	m_bodies_storage(NULL)
{
#if USE_CHRONO == 1
	m_chrono_system = NULL;
#endif
}

bool
ProblemCore::initialize()
{
	if (m_name.empty()) {
		m_name = abi::__cxa_demangle(typeid(*this).name(), NULL, 0, NULL);
		printf("Problem name set to '%s' automatically\n", m_name.c_str());
	}

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

	/* Set LJ parameters that were not set by the user */
	if (simparams()->boundarytype == LJ_BOUNDARY)
	{
		if (isnan(physparams()->r0)) {
			printf("LJ r0 not set, using %g\n", m_deltap);
			physparams()->r0 = m_deltap;
		}
		if (isnan(physparams()->dcoeff)) {
			// this would need knowledge about the max fall, but this isn't present in the problem core
			// so assume H = 1. Problem API 1 works around this issue by having or computing the maximum
			// fall height
			physparams()->dcoeff = 5.0f*length(physparams()->gravity);
			printf("LJ D coefficient not set, using %g\n", physparams()->dcoeff);
		}
	}

	/* Set MK parameters that were not set by the user */
	if (simparams()->boundarytype == MK_BOUNDARY)
	{
		if (isnan(physparams()->MK_d)) {
			physparams()->MK_d = 1.1*m_deltap/physparams()->MK_beta;
			printf("MK D coefficient not set, using %g\n", physparams()->MK_d);
		}
		if (isnan(physparams()->MK_K)) {
			// this would need knowledge about the max fall, but this isn't present in the problem core
			// so assume H = 1. Problem API 1 works around this issue by having or computing the maximum
			// fall height
			physparams()->MK_K = length(physparams()->gravity);
			printf("MK K coefficient not set, using %g\n", physparams()->MK_K);
		}
	}

	/* Set artificial viscosity epsilon to h^2/10 if not set by the user.
	 * For simplicity, we do this regardless of the viscosity model used,
	 * it'll just be ignored otherwise */
	if (isnan(physparams()->epsartvisc)){
		physparams()->epsartvisc = 0.01*simparams()->slength*simparams()->slength;
		printf("Artificial viscosity epsilon is not set, using default value: %e\n", physparams()->epsartvisc);
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

ProblemCore::~ProblemCore(void)
{
	delete [] m_bodies_storage;
	delete m_simframework;
	delete m_physparams;
}

void
ProblemCore::InitializeChrono()
{
#if USE_CHRONO == 1

	cout << "Initializing Chrono FEA... " << endl;

	// initialize FEA system
	m_chrono_system = new ::chrono::ChSystemNSC(); // FIXME NOT necessary the NSC

	m_chrono_system->Set_G_acc(::chrono::ChVector<>(m_physparams->gravity.x, m_physparams->gravity.y,
		m_physparams->gravity.z));


#define SOLVER_TYPE 7

	// Set FEA solver (the way of computing FEM forces, time bottleneck for FEA)
#if SOLVER_TYPE == 0	//  choose between MINRES or MKL
	auto minres_solver = chrono_types::make_shared<::chrono::ChSolverMINRES>();
	m_chrono_system->SetSolver(minres_solver);

	minres_solver->EnableDiagonalPreconditioner(true);
	minres_solver->SetMaxIterations(100);
	minres_solver->SetTolerance(1e-10);
	minres_solver->SetVerbose(false);

	m_chrono_system->SetTimestepperType(::chrono::ChTimestepper::Type::HHT); // HHT is an implicit integration scheme.
	auto mystepper = std::static_pointer_cast<::chrono::ChTimestepperHHT>(m_chrono_system->GetTimestepper());
	mystepper->SetMaxiters(100);
	mystepper->SetAbsTolerances(1e-5);
	mystepper->SetMode(::chrono::ChTimestepperHHT::POSITION);
	mystepper->SetScaling(true);
#endif

#if SOLVER_TYPE == 1
	auto mkl_solver = chrono_types::make_shared<::chrono::ChSolverMKL>();
	mkl_solver->UseSparsityPatternLearner(false);
	mkl_solver->LockSparsityPattern(false);
	mkl_solver->SetVerbose(false);
	m_chrono_system->SetSolver(mkl_solver);

	m_chrono_system->Update();

	m_chrono_system->SetTimestepperType(::chrono::ChTimestepper::Type::EULER_IMPLICIT_LINEARIZED);
#endif

#if SOLVER_TYPE == 2
	auto solver = chrono_types::make_shared<::chrono::ChSolverMINRES>();
	m_chrono_system->SetSolver(solver);
	solver->SetMaxIterations(100);
	solver->SetTolerance(1e-8);
	solver->EnableDiagonalPreconditioner(true);

	m_chrono_system->SetSolverForceTolerance(1e-10);

	m_chrono_system->SetTimestepperType(::chrono::ChTimestepper::Type::EULER_IMPLICIT);
#endif

#if SOLVER_TYPE == 3
    auto solver = chrono_types::make_shared<::chrono::ChSolverMINRES>();
    my_system.SetSolver(solver);
    solver->SetMaxIterations(40);
    solver->SetTolerance(1e-8);

    my_system.SetSolverForceTolerance(1e-10);

    my_system.SetTimestepperType(::chrono::ChTimestepper::Type::EULER_IMPLICIT_LINEARIZED);
#endif

#if SOLVER_TYPE == 4
	auto solver = chrono_types::make_shared<::chrono::ChSolverMINRES>();
	m_chrono_system->SetSolver(solver);
	solver->SetMaxIterations(10000);
	solver->SetTolerance(1e-3);
	solver->EnableDiagonalPreconditioner(true);

	m_chrono_system->SetSolverForceTolerance(1e-3);

	m_chrono_system->SetTimestepperType(::chrono::ChTimestepper::Type::NEWMARK);

	if (auto timestepper = std::dynamic_pointer_cast<::chrono::ChTimestepperHHT>(m_chrono_system->GetTimestepper())) {
		cout << " Setting damping to Project Chrono timestepper" << endl;
		timestepper->SetAlpha(-0.2);
	}
#endif

#if SOLVER_TYPE == 5	// Recommended by Mike Taylor from Project Chrono 
	auto mkl_solver = chrono_types::make_shared<::chrono::ChSolverMKL>();
	mkl_solver->UseSparsityPatternLearner(false);
	mkl_solver->LockSparsityPattern(false);
	mkl_solver->SetVerbose(false);
	m_chrono_system->SetSolver(mkl_solver);

	m_chrono_system->SetTimestepperType(::chrono::ChTimestepper::Type::HHT); // HHT is an implicit integration scheme.
	auto mystepper = std::static_pointer_cast<::chrono::ChTimestepperHHT>(m_chrono_system->GetTimestepper());
	mystepper->SetMaxiters(100);
	mystepper->SetAbsTolerances(1e-5);
	mystepper->SetMode(::chrono::ChTimestepperHHT::ACCELERATION);
	mystepper->SetScaling(true);

	m_chrono_system->Update();
#endif

#if SOLVER_TYPE == 6	// Recommended by Mike Taylor from Project Chrono 
	auto mkl_solver = chrono_types::make_shared<::chrono::ChSolverSparseLU>();
	mkl_solver->UseSparsityPatternLearner(false);
	mkl_solver->LockSparsityPattern(false);
	mkl_solver->SetVerbose(false);
	m_chrono_system->SetSolver(mkl_solver);

	m_chrono_system->SetTimestepperType(::chrono::ChTimestepper::Type::HHT); // HHT is an implicit integration scheme.
	auto mystepper = std::static_pointer_cast<::chrono::ChTimestepperHHT>(m_chrono_system->GetTimestepper());
	mystepper->SetMaxiters(100);
	mystepper->SetAbsTolerances(1e-5);
	mystepper->SetMode(::chrono::ChTimestepperHHT::ACCELERATION);
	mystepper->SetScaling(true);

	m_chrono_system->Update();
#endif

#if SOLVER_TYPE == 7
	/*
	auto minres_solver = chrono_types::make_shared<::chrono::ChSolverMINRES>();
	m_chrono_system->SetSolver(minres_solver);

	minres_solver->EnableDiagonalPreconditioner(true);
	minres_solver->SetMaxIterations(100);
	minres_solver->SetTolerance(1e-5);
	minres_solver->SetVerbose(false);
	*/

	auto mkl_solver = chrono_types::make_shared<::chrono::ChSolverSparseQR>();
	mkl_solver->UseSparsityPatternLearner(true);
	mkl_solver->LockSparsityPattern(true);
	mkl_solver->SetVerbose(false);
	m_chrono_system->SetSolver(mkl_solver);
	m_chrono_system->SetTimestepperType(::chrono::ChTimestepper::Type::EULER_IMPLICIT); // HHT is an implicit integration scheme.
/*	auto mystepper = std::static_pointer_cast<::chrono::ChTimestepperHHT>(m_chrono_system->GetTimestepper());
	mystepper->SetMaxiters(100);
	mystepper->SetAbsTolerances(1e-5);
	//mystepper->SetMode(::chrono::ChTimestepperHHT::POSITION);
	mystepper->SetScaling(true);
	*/
#endif

#else
	throw runtime_error ("ProblemCore::InitializeChrono Trying to use Chrono without USE_CHRONO defined !\n");
#endif
}

void
ProblemCore::SetFeaReady()
{
#if USE_CHRONO == 1
	//m_chrono_system->SetupInitial();

	// If there are nodes to write create the output file

	//m_chrono_system->Update();

	if (simparams()->numNodesToWrite)
		create_fea_nodes_file();
	if (simparams()->numConstraintsToWrite)
		create_fea_constr_file();

#endif
}

void ProblemCore::FinalizeChrono(void)
{
#if USE_CHRONO == 1
	if (m_chrono_system)
		delete m_chrono_system;
	if (m_fea_nodes_file)
		m_fea_nodes_file.close();
#else
	throw runtime_error ("ProblemCore::FinalizeChrono Trying to use Chrono without USE_CHRONO defined !\n");
#endif
}

// callback for initializing joints between Chrono bodies
void ProblemCore::initializeObjectJoints()
{
	// Default: do nothing

	// See also: http://api.chrono.projectchrono.org/links.html
}

/// Allocate storage required for the integration of the kinematic data
/// of moving bodies.
void
ProblemCore::allocate_bodies_storage()
{
	const uint nbodies = simparams()->numbodies;

	if (nbodies) {
		// TODO: this should depend on the integration scheme
		m_bodies_storage = new KinematicData[nbodies];
	}
}

void
ProblemCore::add_moving_body(Object* object, const MovingBodyType mbtype)
{
	// Moving bodies are put at the end of the bodies vector,
	// ODE bodies and moving bodies for which we want a force feedback
	// are put at the beginning of the bodies vector (ODE bodies first).
	// The reason behind this ordering is the way the forces on bodies
	// are reduced by a parallel prefix sum: all the bodies that require
	// force computing must have consecutive ids.
	const uint index = m_bodies.size();
	if (index >= MAX_BODIES)
		throw runtime_error ("ProblemCore::add_moving_body Number of moving bodies superior to MAX_BODIES. Increase MAXBODIES\n");
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
			throw runtime_error ("ProblemCore::add_moving_body Cannot add a floating body without CHRONO\n");
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

// Add a body to the fea system (so far used as joint)
void
ProblemCore::add_fea_body(Object* object)
{
	cout << "adding fea object " << endl;

	const uint index = m_bodies.size();
	if (index >= MAX_BODIES) // FIXME do we need an independent  MAX_FEA_BODIES?
		throw runtime_error ("ProblemCore::add_fea_body Number of fea bodies superior to MAX_BODIES. Increase MAXBODIES\n");

	FeaBodyData *feadata = new FeaBodyData;
	feadata->index = index;
	feadata->object = object;

	m_fea_bodies.push_back(feadata);

	// Setting body id after insertion
	for (uint id = 0; id < m_fea_bodies.size(); id++) {
		m_fea_bodies[id]->id = id;
	}

	simparams()->numfeabodies++;
}

#if USE_CHRONO == 1
// Simulate a ground where to attach FEA structures: defined the ground (a plane, analytically) set
// fixed all the nodes that are in the negative side of the plane
void
ProblemCore::groundFeaNodes(std::shared_ptr<::chrono::fea::ChMesh> fea_mesh)
{
	// four coefficients of the plane equation
	float4 plane = physparams()->feaGround;

	if ((plane.x != 0)  && (plane.y != 0) && (plane.z != 0) && (plane.w != 0)) // no ground defined
		return;

	cout << "grounding FEA body using a = " << plane.x << " b = " << plane.y << " c = " << plane.z << " d = " << plane.w << endl;

	uint nodes_num = fea_mesh->GetNnodes();
	shared_ptr<::chrono::fea::ChNodeFEAxyzD> node;
	Point p;

	//TODO give the possibility to set more planes as grounds

	// iterate over the FEA nodes in the current mesh (a single GPUSPH geometry)
	for (int i = 0; i < nodes_num; ++i) {

		// get node
		node = dynamic_pointer_cast<::chrono::fea::ChNodeFEAxyzD>(fea_mesh->GetNode(i));

		// get coordinates of the node
		p(0) = node->GetPos().x();
		p(1) = node->GetPos().y();
		p(2) = node->GetPos().z();

		/*Any point in the negative side of the plane is grounded*/
		const float prod = (plane.x * p(0) + plane.y * p(1) + plane.z * p(2) - plane.w);

		if (prod < 0) {
			node->SetFixed(true);
			cout << "grounded (mesh local index) " << i << endl;
		}
	}
}
#endif

void
ProblemCore::restore_moving_body(const MovingBodyData & saved_mbdata, const uint numparts, const int firstindex, const int lastindex)
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
		throw runtime_error ("ProblemCore::restore_moving_body Cannot restore a floating body without CHRONO\n");
#endif
		}
}

MovingBodyData *
ProblemCore::get_mbdata(const uint index)
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
ProblemCore::get_mbdata(const Object* object)
{
	for (vector<MovingBodyData *>::iterator it = m_bodies.begin() ; it != m_bodies.end(); ++it) {
		if ((*it)->object == object)
			return *it;
	}
	throw runtime_error("get_body: invalid object\n");
	return NULL;
}

size_t
ProblemCore::get_bodies_numparts(void)
{
	size_t total_parts = 0;
	for (vector<MovingBodyData *>::iterator it = m_bodies.begin() ; it != m_bodies.end(); ++it) {
		total_parts += (*it)->object->GetNumParts();
	}

	return total_parts;
}


size_t
ProblemCore::get_forces_bodies_numparts(void)
{
	size_t total_parts = 0;
	for (vector<MovingBodyData *>::iterator it = m_bodies.begin() ; it != m_bodies.end(); ++it) {
		if ((*it)->type == MB_FLOATING || (*it)->type == MB_FORCES_MOVING)
			total_parts += (*it)->object->GetNumParts();
	}
	return total_parts;
}

size_t
ProblemCore::get_fea_objects_numparts(void)
{
	size_t total_parts = 0;
	for (vector<FeaBodyData *>::iterator it = m_fea_bodies.begin() ; it != m_fea_bodies.end(); ++it) {
		total_parts += (*it)->object->GetNumParts();
	}

	//FIXME all this should not stay here, should have a proper location
	if (gdata->iterations == 0 && total_parts && gdata->s_hFeaNatCoords == NULL) {
		gdata->s_hFeaNatCoords = new float4 [total_parts];
		cout << "Created host buffer for Nat coords (" << total_parts <<" particles)" << endl;
	}

	//FIXME all this should not stay here, should have a proper location
	if (gdata->iterations == 0 && total_parts && gdata->s_hFeaOwningNodes == NULL) {
		gdata->s_hFeaOwningNodes = new uint4 [total_parts];
		cout << "Created host buffer for owning nodes (" << total_parts <<" particles)" << endl;
	}

	return total_parts;
}

size_t
ProblemCore::get_fea_objects_numnodes(void)
{
	size_t total_parts = 0;

#if USE_CHRONO == 1
	for (vector<FeaBodyData *>::iterator it = m_fea_bodies.begin() ; it != m_fea_bodies.end(); ++it) {
		total_parts += (*it)->object->GetNumFeaNodes();
	}

	//FIXME all this should not stay here, should have a proper location
	if (gdata->iterations == 0 && total_parts) {
		m_old_fea_vel.resize(total_parts);
	}
#endif

	return total_parts;
}

size_t
ProblemCore::get_body_numparts(const int index)
{
	return m_bodies[index]->object->GetNumParts();
}


size_t
ProblemCore::get_body_numparts(const Object* object)
{
	return get_mbdata(object)->object->GetNumParts();
}

void
ProblemCore::calc_grid_and_local_pos(double3 const& globalPos, int3 *gridPos, float3 *localPos) const
{
	int3 _gridPos = calc_grid_pos(globalPos);
	*gridPos = _gridPos;
	*localPos = make_float3(globalPos - m_origin -
		(make_double3(_gridPos) + 0.5)*m_cellsize);
}

/*This is the writer for the FEA nodes. We don't use a regular GPUSPH writer
 * since running a writer causes the dumping of the data from device. FEA nodes
 * are already stored on host, so no dumping needed.*/
void
ProblemCore::write_fea_nodes(const double t)
{
	// Printing position of nodes in a GT_FEA_WRITE
	m_fea_nodes_file << t;
	for (int i = 0; i < simparams()->numNodesToWrite; ++i) {

		shared_ptr<::chrono::fea::ChNodeFEAxyz> node;

		/*We read from the array of nodes to be wirtten. This array
		 * is set at the beginning of the simulation in ProblemAPI_1
		 * when defining GT_FEA_WRITE geometries*/
		node = gdata->s_hWriteFeaNodesPointers[i];

		// print nodes position to file
		m_fea_nodes_file << '\t' <<
		node->GetPos().x() << '\t' <<
		node->GetPos().y() << '\t' <<
		node->GetPos().z();
	}
	m_fea_nodes_file << endl;

	// Printing reactions on constraints
	m_fea_constr_file << t;
	for (int i = 0; i < simparams()->numConstraintsToWrite; ++i) {

		::chrono::ChVector<> force = gdata->s_hWriteFeaPointConstrPointers[i]->Get_react_force();
		::chrono::ChVector<> torque = gdata->s_hWriteFeaDirConstrPointers[i]->Get_react_torque();

		m_fea_constr_file << '\t' <<
		force.x() << '\t' <<
		force.y() << '\t' <<
		force.z() << '\t' <<
		torque.x() << '\t' <<
		torque.y() << '\t' <<
		torque.z() << '\t' <<
		gdata->total_fea_force.x << '\t' <<
		gdata->total_fea_force.y << '\t' <<
		gdata->total_fea_force.z;
	}
	m_fea_constr_file << endl;
}

/*Multi-device: collect from the devices the forces applied to the FEA nodes*/
void
ProblemCore::reduce_fea_forces(BufferList &buffers, const uint numFeaNodes)
{
	float4 *forces = buffers.getData<BUFFER_FEA_EXCH>(); // contains forces from (FSI)

	//number of devices
	uint n_devs = gdata->devices;

	for (uint i = 0; i < numFeaNodes; i++) {

		float4 f_sum = forces[i];

		for (uint d = 1; d < n_devs; d++)
			f_sum += forces[d*numFeaNodes + i];

		forces[i] = f_sum;
	}
}

// Set external forces and Fluid-Solid Interaction (FSI) forces to FEM nodes
void
ProblemCore::fea_init_step(BufferList &buffers, const uint numFeaParts, const double t,  const int step)
{
#if USE_CHRONO == 1

	/* The BUFFER_FEA_EXCH is used to exchange data between the FEM
	 * and SPH models. It transfers FSI forces to the FEM model and
	 * nodes velocities to the SPH model.*/
	const float4 *forces = buffers.getConstData<BUFFER_FEA_EXCH>(); // contains forces from FSI now
	const particleinfo *info = buffers.getConstData<BUFFER_INFO>();

	shared_ptr<::chrono::fea::ChNodeFEAxyzD> node;

	/* In Project Chrono, nodes are locally indexed within each mesh
	 * (we associate a mesh to each deformable object)*/
	uint n = 0; // node index within an individual mesh
	uint o = 0; // index of meshes (associated to defomable objects) starting from the one with index 0

	/* We perform a temporal averaging of the forces in order to reduce noise
	 * therefore we store the previous forces in a matrix.*/
	uint av_idx = gdata->averager_index; // Index of the current sample in the averager matrix

	/* Sum up all the forces applied to the FEM system for printing purpose */
	float3 total_force = make_float3(0.0, 0.0, 0.0);


	for(uint i = 0; i < numFeaParts; ++i) {

		// get number of nodes in the current mesh
		uint nnodes = m_fea_bodies[o]->object->GetNumFeaNodes();
		// get nth node within the current mesh
		node = dynamic_pointer_cast<::chrono::fea::ChNodeFEAxyzD>
			(m_fea_bodies[o]->object->GetFeaMesh()->GetNode(n));


		/* -- Time averaging -- */

		// Divide the new force by the number of samples in the
		// averaging time window
		float3 new_f  = as_float3(forces[i])/600.0f;

		// Retrieve the averaged force at previous instant
		// (stored at the end of the row)
		float3 node_f = gdata->forces_averager[i][600];

		// Add new force and subtract oldest force to update the
		// average
		node_f += (new_f - gdata->forces_averager[i][av_idx]);

		// Replace the oldest force in matrix with the newest one
		gdata->forces_averager[i][av_idx] = new_f;

		// Store the new averaged value
		gdata->forces_averager[i][600] = node_f;

		/* -- End of averaging -- */


		// Add external forces (from ext_force_callback), if any
		// (s_hFeaExtForce[i] is true if node i has an external force applied)
		//
		if (simparams()->fcallback && gdata->s_hFeaExtForce[i])
			node_f += gdata->s_FeaExtForce;

		// Apply the force to the FEM node
		node->SetForce(::chrono::ChVector<>(node_f.x, node_f.y, node_f.z));

		// Add the force for the current node to the total forces computation
		total_force += node_f;


		// tracking the nodes in the current mesh
		n++;

		// reset mesh nodes tacking when we reach the nuber of nodes in the mesh
		// and pass to next mesh
		// TODO FIXME check how this complies with node recycling
		if (n == nnodes) {
			n = 0;
			o ++;
		}
	}

	gdata->total_fea_force = total_force;

	// Manage averaging index
	av_idx ++;
	if (av_idx == 600) av_idx = 0;
	gdata->averager_index = av_idx;

#endif
}

/* Do FEA and get displacements during timestep */
void
ProblemCore::fea_do_step(BufferList &buffers, const uint numFeaParts, const  double dt, const bool dofea, const uint fea_every)
{
#if USE_CHRONO == 1

	const particleinfo *info = buffers.getConstData<BUFFER_INFO>();
	// Now we are going to put into BUFFER_FEA_EXCH the data from FEA
	// (node velocities) that will be used to move the associated particles
	float4 *fea_vel = buffers.getData<BUFFER_FEA_EXCH>();

	if (dofea) {
		// - we perform FEA during the predictor, where dt is half the complete
		//   time-step, then we use 2*dt
		// - we perform FEA every fea_every SPH steps, then (assuming dt constant
		//   over this time) we perform FEA using a factor fea_every on the time-step
		m_chrono_system->DoStepDynamics(2*dt*fea_every);


		shared_ptr<::chrono::fea::ChNodeFEAxyzD> node;
		float4 node_vel = make_float4(0.0); //FIXME it could be float3

		/*In Project Chrono, nodes are locally indexed within each mesh (each deformable object)*/
		uint n = 0; //node in the object
		uint o = 0; // we go through all the meshes (defomable objects) starting from the one with index 0


		for(uint i = 0; i < numFeaParts; ++i) {

			uint nnodes = m_fea_bodies[o]->object->GetNumFeaNodes();
			node = dynamic_pointer_cast<::chrono::fea::ChNodeFEAxyzD>
				(m_fea_bodies[o]->object->GetFeaMesh()->GetNode(n));

			// get updated node velocity from Chrono
			node_vel.x = node->GetPos_dt().x();
			node_vel.y = node->GetPos_dt().y();
			node_vel.z = node->GetPos_dt().z();

			// to be sent to the SPH particle system
			fea_vel[i] = node_vel;

			// store the most recent fea vel to be used dusing the fea_every SPH steps
			// TODO we could send it once and reuse in the gpu, but we would apply a 
			// conditional in gpu side and we should keep track of the current iteration.
			// SEE WHAT IS CONVENIENT
			m_old_fea_vel[i] = node_vel;

			// tracking the nodes in the current mesh
			n++;

			// reset mesh nodes tacking when we reach the nuber of nodes in the mesh
			// and pass to next mesh
			// TODO FIXME check how this complies with node recycling
			if (n == nnodes) {
				n = 0;
				o++;
			}
		}
	} else {
		for(uint i = 0; i < numFeaParts; ++i) {
			// when we do not compute a new FEA we send the newest stored velocity
			fea_vel[i] = m_old_fea_vel[i];
		}


		/*TODO DISCUSS alternatively we could fo FEA for the SPH dt and fea_every times the displacement*/
	}
#endif
}

void
ProblemCore::get_bodies_cg(void)
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
ProblemCore::set_body_cg(const double3& crot, MovingBodyData* mbdata) {
	mbdata->kdata.crot = crot;
}


void
ProblemCore::set_body_cg(const uint index, const double3& crot) {
	set_body_cg(crot, m_bodies[index]);
}


void
ProblemCore::set_body_cg(const Object *object, const double3& crot) {
	set_body_cg(crot, get_mbdata(object));
}


void
ProblemCore::set_body_linearvel(const double3& lvel, MovingBodyData* mbdata) {
	mbdata->kdata.lvel = lvel;
}


void
ProblemCore::set_body_linearvel(const uint index, const double3& lvel) {
	set_body_linearvel(lvel, m_bodies[index]);
}


void
ProblemCore::set_body_linearvel(const Object *object, const double3& lvel)
{
	set_body_linearvel(lvel, get_mbdata(object));
}


void
ProblemCore::set_body_angularvel(const double3& avel, MovingBodyData* mbdata) {
	mbdata->kdata.avel = avel;
}

void
ProblemCore::set_body_angularvel(const uint index, const double3& avel) {
	set_body_angularvel(avel, m_bodies[index]);
}


void
ProblemCore::set_body_angularvel(const Object *object, const double3& avel)
{
	set_body_angularvel(avel, get_mbdata(object));
}


void
ProblemCore::bodies_forces_callback(const double t0, const double t1, const uint step, float3 *forces, float3 *torques)
{ /* default does nothing */ }


void
ProblemCore::post_timestep_callback(const double t)
{ /* default does nothing */ }


void
ProblemCore::moving_bodies_callback(const uint index, Object* object, const double t0, const double t1,
		const float3& force, const float3& torque, const KinematicData& initial_kdata,
		KinematicData& kdata, double3& dx, EulerParameters& dr)
{ /* default does nothing */ }

// input: force, torque, step number, dt
// output: cg, trans, steprot (can be input uninitialized)
void
ProblemCore::bodies_timestep(const float3 *forces, const float3 *torques, const int step,
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
	bool fea_is_enabled = simparams()->simflags & ENABLE_FEA;
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
			if (step == 2 && !fea_is_enabled) {
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
	if (there_is_at_least_one_chrono_body && !fea_is_enabled) {
		m_chrono_system->DoStepDynamics(dt1);
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
ProblemCore::copy_planes(PlaneList& planes)
{
	return;
}

//! No yield strength contribution
template<RheologyType rheologytype>
enable_if_t< yield_strength_type<rheologytype>() == NO_YS, float >
getInitViscYieldTerm(const SimParams * simparams, const PhysParams * physparams)
{
	return 0.0f;
}

//! Standard contribution from the yield strength
/*! In this case, if there is a yield strength, the maximum viscosity can be infinite
 * (for null shear-rate), so we pick the limiting value
 */
template<RheologyType rheologytype>
enable_if_t< yield_strength_type<rheologytype>() == STD_YS, float >
getInitViscYieldTerm(const SimParams * simparams, const PhysParams * physparams)
{
	float maxViscYieldTerm = 0.0f;
	for (uint f = 0; f < physparams->numFluids(); ++f)
	{
		if (physparams->yield_strength[f] != 0)
			return physparams->limiting_kinvisc;
	}
	return maxViscYieldTerm;
}

//! Regularized contribution from the yield strength
/*! In this case, the yield strength contribution to the viscosity is computed as m*tau/rho
 */
template<RheologyType rheologytype>
enable_if_t< yield_strength_type<rheologytype>() == REG_YS, float >
getInitViscYieldTerm(const SimParams * simparams, const PhysParams * physparams)
{
	float maxViscYieldTerm = 0.0f;
	for (uint f = 0; f < physparams->numFluids(); ++f)
	{
		maxViscYieldTerm = fmaxf(maxViscYieldTerm, physparams->visc_regularization_param[f]*physparams->yield_strength[f]/physparams->rho0[f]);
	}
	return maxViscYieldTerm;
}

//! Shear rate contribution in the linear case
/** For NEWTON, BINGHAM and PAPANASTASIOU */
template<RheologyType rheologytype>
enable_if_t<not NONLINEAR_RHEOLOGY(rheologytype), float >
getInitViscShearTerm(const SimParams * simparams, const PhysParams * physparams, const float max_shear_rate)
{
	float maxViscShearTerm = 0.0f;
	for (uint f = 0; f < physparams->numFluids(); ++f)
	{
		maxViscShearTerm = fmaxf(maxViscShearTerm, physparams->kinematicvisc[f]);
	}
	return maxViscShearTerm;
}

//! Shear rate contribution in the power-law case
/** For POWER_LAW, HERSCHEL_BULKLEY and ALEXANDROU */
template<RheologyType rheologytype>
enable_if_t<POWERLAW_RHEOLOGY(rheologytype), float >
getInitViscShearTerm(const SimParams * simparams, const PhysParams * physparams, const float max_shear_rate)
{
	float maxViscShearTerm = 0.0f;
	for (uint f = 0; f < physparams->numFluids(); ++f)
	{
		// exponent of the power law
		const float n = physparams->visc_nonlinear_param[f];
		// actually linear case
		if (n == 1)
			maxViscShearTerm = fmaxf(maxViscShearTerm, physparams->kinematicvisc[f]);
		// dilatant fluid (n > 1): visc grows with growing shear rate, limit according to the maximum (expected) shear rate
		else if (n > 1)
			maxViscShearTerm = fmaxf(maxViscShearTerm, physparams->kinematicvisc[f]*pow(max_shear_rate, n));
		// shear-thinning fluid (n < 1): visc becomes infinite for vanishing shear rate
		else
			return physparams->limiting_kinvisc;
	}
	return maxViscShearTerm;
}

//! Shear rate contribution in the exponential case
/** For DEKEE_TURCOTTE and ZHU */
template<RheologyType rheologytype>
enable_if_t<EXPONENTIAL_RHEOLOGY(rheologytype), float >
getInitViscShearTerm(const SimParams * simparams, const PhysParams * physparams, const float max_shear_rate)
{
	float maxViscShearTerm = 0.0f;
	for (uint f = 0; f < physparams->numFluids(); ++f)
	{
		const float t1 = physparams->visc_nonlinear_param[f];
		// linear case and shear-thinning (t1 > 0) case: maximum viscosity is achieved for vanishing shear rate,
		// and it's equal to the kinematic coefficient
		if (t1 >= 0)
			maxViscShearTerm = fmaxf(maxViscShearTerm, physparams->kinematicvisc[f]);
		// dilatant fluid (t1 < 0): visc grows with growing shear rate, limit according to the maximum (expected) shear rate
		else
			maxViscShearTerm = fmaxf(maxViscShearTerm, physparams->kinematicvisc[f]*exp(-t1*max_shear_rate));
	}
	return maxViscShearTerm;
}


template<RheologyType rheologytype>
float get_dt_from_visc(const SimParams *simparams, const PhysParams *physparams)
{
	float effvisc = getInitViscYieldTerm<rheologytype>(simparams, physparams);
	if (effvisc < physparams->limiting_kinvisc) {
		float max_shear_rate = 0;
		// we assume that the maximum shear rate we can expect is given by
		// a velocity difference equal to the speed of sound, over a smoothing length
		// (given that the maximum speed should be an order of magnitude lower than the speed of sound
		// of any fluid, this shouldn't be a strong assumption)
		for (uint f = 0; f < physparams->numFluids(); ++f)
			max_shear_rate = fmaxf(max_shear_rate, physparams->sscoeff[f]/simparams->slength);

		printf("Expected maximum shear rate: %g 1/s\n", max_shear_rate);
		effvisc += getInitViscShearTerm<rheologytype>(simparams, physparams, max_shear_rate);
	}
	effvisc = fminf(effvisc, physparams->limiting_kinvisc);
	return simparams->slength*simparams->slength/effvisc;
}

void
ProblemCore::check_dt(void)
{
	// warn if dt was set by the user but adaptive dt is enable
	if (simparams()->dt && simparams()->simflags & ENABLE_DTADAPT)
	{
		fprintf(stderr, "WARNING: dt %g will be used only for the first iteration because adaptive dt is enabled\n",
			simparams()->dt);
	}

	float dt_from_sspeed = INFINITY;
	for (uint f = 0 ; f < physparams()->numFluids(); ++f) {
		float sspeed = physparams()->sscoeff[f];
		dt_from_sspeed = fmin(dt_from_sspeed, simparams()->slength/sspeed);
	}
	dt_from_sspeed *= simparams()->dtadaptfactor;

	float dt_from_gravity = sqrt(simparams()->slength/length(physparams()->gravity));
	dt_from_gravity *= simparams()->dtadaptfactor;

	float dt_from_visc = NAN;
	if (simparams()->rheologytype != INVISCID)
	{
		const SimParams *sim = simparams();
		const PhysParams *phys = physparams();

#define GET_DT_FROM_VISC(rheologytype) case rheologytype: dt_from_visc = get_dt_from_visc<rheologytype>(sim, phys); break;
		switch(simparams()->rheologytype)
		{
			GET_DT_FROM_VISC(NEWTONIAN)
			GET_DT_FROM_VISC(BINGHAM)
			GET_DT_FROM_VISC(PAPANASTASIOU)
			GET_DT_FROM_VISC(POWER_LAW)
			GET_DT_FROM_VISC(HERSCHEL_BULKLEY)
			GET_DT_FROM_VISC(ALEXANDROU)
			GET_DT_FROM_VISC(DEKEE_TURCOTTE)
			GET_DT_FROM_VISC(ZHU)
		}
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
ProblemCore::check_neiblistsize(void)
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
ProblemCore::hydrostatic_density(float h, int i) const
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
ProblemCore::density_for_pressure(float P, int i) const
{
	return  pow(P/physparams()->bcoeff[i] + 1,1/physparams()->gammacoeff[i])-1.0;
}

float
ProblemCore::soundspeed(float rho_tilde, int i) const
{
	const float rho_ratio = rho_tilde + 1;

    return physparams()->sscoeff[i]*pow(rho_ratio, physparams()->sspowercoeff[i]);
}

float
ProblemCore::pressure(float rho_tilde, int i) const
{
	const float rho_ratio = rho_tilde + 1;

	return physparams()->bcoeff[i]*(pow(rho_ratio, physparams()->gammacoeff[i]) - 1);
}

float
ProblemCore::physical_density( float rho_tilde, int i) const
{
	return (rho_tilde + 1)*physparams()->rho0[i];
}

float
ProblemCore::numerical_density( float rho, int i) const
{
	return rho/physparams()->rho0[i] - 1;
}

void
ProblemCore::add_gage(double3 const& pt)
{
	simparams()->gage.push_back(make_double4(pt.x, pt.y, 0., pt.z));
}

plane_t
ProblemCore::implicit_plane(double4 const& p)
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
ProblemCore::make_plane(Point const& pt, Vector const& normal)
{
	plane_t plane;

	plane.normal = make_float3(normal);
	calc_grid_and_local_pos(make_double3(pt), &plane.gridPos, &plane.pos);

	return plane;
}

string const&
ProblemCore::create_problem_dir(void)
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
ProblemCore::create_fea_nodes_file(void)
{
	string filename = m_problem_dir + "/data/fea_nodes.txt";

	std::cout << "Opening: " << filename << endl;
	m_fea_nodes_file.open(filename);

	// add columns description
	m_fea_nodes_file << "time[s]";

	for (int n = 0; n < simparams()->numNodesToWrite; ++n) {
		int node_id = gdata->s_hWriteFeaNodesIndices[n];
		m_fea_nodes_file <<
			"\tNode_" << node_id << "_x[m]"
#if 0 //write only x component
			<<
			"\tNode_" << node_id << "_y [m]"
			<<
			"\tNode_" << node_id << "_z [m]"
#endif
			;
	}
	m_fea_nodes_file << endl;
}

void
ProblemCore::create_fea_constr_file(void)
{
	string filename = m_problem_dir + "/data/fea_dynamometer.txt";

	std::cout << "Opening: " << filename << endl;
	m_fea_constr_file.open(filename);

	// add columns description
	m_fea_constr_file << "time[s]";

	for (int n = 0; n < simparams()->numConstraintsToWrite; ++n) {
		m_fea_constr_file <<
			"\tF_" << n << "_x[N]"
			<<
			"\tF_" << n  << "_y[N]"
			<<
			"\tF_" << n << "_z[N]"
			<<
			"\tT_" << n << "_x[N*m]"
			<<
			"\tT_" << n  << "_y[N*m]"
			<<
			"\tT_" << n << "_z[N*m]"
			<<
			"\tFapp_" << n  << "_x[N]"
			<<
			"\tFapp_" << n  << "_y[N]"
			<<
			"\tFapp_" << n  << "_z[N]"
			;
	}
	m_fea_constr_file << endl;
}

void
ProblemCore::add_writer(WriterType wt, double freq)
{
	m_writers.push_back(make_pair(wt, freq));
}


// override in problems where you want to save
// at specific times regardless of standard conditions
bool
ProblemCore::need_write(double t) const
{
	return false;
}

// overridden in subclasses if they want to write custom stuff
// using the CALLBACKWRITER
void
ProblemCore::writer_callback(CallbackWriter *,
	uint numParts, BufferList const&, uint node_offset, double t,
	const bool testpoints) const
{
	fprintf(stderr, "WARNING: CallbackWriter is being used, but writer_callback wasn't implemented\n");
}


// is the simulation finished at the given time?
bool
ProblemCore::finished(double t) const
{
	double tend(simparams()->tend);
	return tend && (t > tend);
}

float3
ProblemCore::ext_force_callback(const double t)
{
	throw std::runtime_error("default ext_force_callback invoked! did you forget to override ext_force_callback(double)");
}

float3
ProblemCore::g_callback(const double t)
{
	throw std::runtime_error("default g_callback invoked! did you forget to override g_callback(double)");
}


// Fill the device map with "devnums" (*global* device ids) in range [0..numDevices[.
// Default algorithm: split along the longest axis
void ProblemCore::fillDeviceMap()
{
	fillDeviceMapByAxis(LONGEST_AXIS);
}

// partition by splitting the cells according to their linearized hash.
void ProblemCore::fillDeviceMapByCellHash()
{
	uint cells_per_device = gdata->nGridCells / gdata->totDevices;
	for (uint i=0; i < gdata->nGridCells; i++)
		// guaranteed to fit in a devcount_t due to how it's computed
		gdata->s_hDeviceMap[i] = devcount_t(min( int(i/cells_per_device), gdata->totDevices-1));
}

// partition by splitting along the specified axis
void ProblemCore::fillDeviceMapByAxis(SplitAxis preferred_split_axis)
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
void ProblemCore::fillDeviceMapByAxisBalanced(SplitAxis preferred_split_axis)
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

void ProblemCore::fillDeviceMapByEquation()
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
void ProblemCore::fillDeviceMapByAxesSplits(uint Xslices, uint Yslices, uint Zslices)
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
void ProblemCore::fillDeviceMapByRegularGrid()
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
ProblemCore::max_parts(uint numParts)
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
ProblemCore::calculateDensityDiffusionCoefficient()
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
ProblemCore::set_grid_params(void)
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
ProblemCore::calc_grid_pos(const Point& pos) const
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
ProblemCore::calc_grid_offset(double3 const& vec) const
{
	int3 gridOff;
	gridOff = make_int3(floor(vec/m_cellsize));

	return gridOff;
}

/// Compute the local (fractional grid cell) components of a vector,
/// given the vector and its grid offset
double3
ProblemCore::calc_local_offset(double3 const& vec, int3 const& gridOff) const
{
	return vec - (make_double3(gridOff) + 0.5)*m_cellsize;
}


// Compute address in grid from position
uint
ProblemCore::calc_grid_hash(int3 gridPos) const
{
	return gridPos.COORD3 * m_gridsize.COORD2 * m_gridsize.COORD1 + gridPos.COORD2 * m_gridsize.COORD1 + gridPos.COORD1;
}


void
ProblemCore::calc_localpos_and_hash(const Point& pos, const particleinfo& info, float4& localpos, hashKey& hash) const
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
ProblemCore::init_volume(BufferList &buffers, uint numParticles)
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
ProblemCore::init_internal_energy(BufferList &buffers, uint numParticles)
{
	float *int_eng = buffers.getData<BUFFER_INTERNAL_ENERGY>();

	for (uint i = 0; i < numParticles; ++i) {
		int_eng[i]=0;
	}

}

/* Default initialization for k and epsilon  */
void
ProblemCore::init_keps(BufferList &buffers, uint numParticles)
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
ProblemCore::init_turbvisc(BufferList &buffers, uint numParticles)
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

/*!
 * Initialize effective pressure for GRANULAR rheology
 */
void
ProblemCore::init_effpres(BufferList &buffers, uint numParticles)
{
	if (simparams()->rheologytype != GRANULAR)
		return;

	float *effpres = buffers.getData<BUFFER_EFFPRES>();

	for (uint i = 0; i < numParticles; ++i) {
		effpres[i] = 0.;
	}
}

void
ProblemCore::imposeBoundaryConditionHost(
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

void ProblemCore::imposeForcedMovingObjects(
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

void ProblemCore::calcPrivate(
	flag_t options,
	BufferList const& bufread,
	BufferList & bufwrite,
	uint numParticles,
	uint particleRangeEnd,
	uint deviceIndex,
	const GlobalData * const gdata)
{
	throw invalid_argument("CALC_PRIVATE requested, but calcPrivate() not implemented in problem");
}

std::string ProblemCore::get_private_name(flag_t buffer) const
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

void ProblemCore::PlaneCut(PointVect& points, const double a, const double b,
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

// callback for initializing particles with custom values
void ProblemCore::initializeParticles(BufferList &buffers, const uint numParticles)
{
	// Default: do nothing

	/*
	// Example usage

	// 1. warn the user if this is expected to take much time
	printf("Initializing particles velocity...\n");

	// 2. grab the particle arrays from the buffer list
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();

	// 3. iterate on the particles
	for (uint i = 0; i < numParticles; i++) {
		// 4. optionally grep with custom filters (e.g. type, size, position, etc.)
		if (FLUID(info[i]))
			// 5. set in loco the desired values
			vel[i].x = 0.1;
	}
	*/
}

void ProblemCore::resetBuffers(BufferList &buffers, const uint numParticles)
{
	particleinfo *info = buffers.getData<BUFFER_INFO>();
	double4 *globalPos = buffers.getData<BUFFER_POS_GLOBAL>();
	float4 *vel = buffers.getData<BUFFER_VEL>();
	float4 *eulerVel = buffers.getData<BUFFER_EULERVEL>();

	for (uint i = 0; i < numParticles; i++) {
		// Compute density for hydrostatic filling.
		float rho = physical_density(physparams()->rho0[0],fluid_num(info[i]));
		if (m_hydrostaticFilling)
			rho = hydrostatic_density(m_waterLevel - globalPos[i].z, fluid_num(info[i]));
		vel[i] = make_float4(0, 0, 0, rho);
		if (eulerVel)
			eulerVel[i] = make_float4(0);
	}

	// initialize values of k and e for k-e model
	if (simparams()->turbmodel == KEPSILON)
		init_keps(buffers, numParticles);

	// call user-set initialization routine, if any
	initializeParticles(buffers, numParticles);
}
