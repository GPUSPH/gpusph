/*  Copyright 2011-2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

    Istituto Nazionale di Geofisica e Vulcanologia
        Sezione di Catania, Catania, Italy

    Universitˆ di Catania, Catania, Italy

    Johns Hopkins University, Baltimore, MD

  Ê This file is part of GPUSPH.

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

#include <iostream>
#include <cmath>
#include <stdexcept>
#include "CSIDevice.h"
#include "GlobalData.h"
#include "cudasimframework.cu"
#include "particledefine.h"
#include "Cube.h"
#include "Sphere.h"
#include "Cylinder.h"
#include "Point.h"
#include "Vector.h"
#include "vector_math.h"
#include "vector_print.h"
#include "CallbackWriter.h"

#define MK_par 2

CSIDevice::CSIDevice(GlobalData *_gdata) : Problem(_gdata)
{
	parser = DataFileParser("obx.txt");
	parser.printVariables();

	// Size and origin of the simulation domain
	lx = parser.getD("tank_x");
	ly = parser.getD("tank_y");
	lz = parser.getD("tank_z");

	const double tower_height = parser.getD("tower_height");
	m_size = make_double3(lx + 0.5, ly, (lz + tower_height)*1.2);
	m_origin = make_double3(-0.5, 0.0, - 0.2);

	// Data for problem setup
	beta = atan(0.25);
	double h_length_factor = parser.getD("h_length_factor");
	h_length = h_length_factor * lx;
	slope_length = lx - h_length;
	H = parser.getD("water_height");

	SETUP_FRAMEWORK(
		viscosity<ARTVISC>,
		//viscosity<KINEMATICVISC>,
		//viscosity<SPSVISC>,
		boundary<DYN_BOUNDARY>,
		//boundary<MK_BOUNDARY>
		periodicity<PERIODIC_Y>
		);

	//addFilter(SHEPARD_FILTER, 27);

	// SPH parameters
	set_deltap(parser.getF("deltap"));
	simparams()->tend = parser.getD("duration");
	simparams()->dt = 0.00003;
	simparams()->dtadaptfactor = 0.3;
	simparams()->buildneibsfreq = 10;

	simparams()->neiblistsize = 192;

	// Physical parameters
	m_physparams->gravity = make_float3(0.0, 0.0, -9.81);
	float g = length(m_physparams->gravity);

	add_fluid(1000.0);
	set_equation_of_state(0, 7.0f, 30.0f);


	m_physparams->r0 = m_deltap;

	m_physparams->artvisccoeff = 0.1f;
    m_physparams->smagfactor = 0.12*0.12*m_deltap*m_deltap;
	m_physparams->kspsfactor = (2.0/3.0)*0.0066*m_deltap*m_deltap;
	m_physparams->epsartvisc = 0.01*simparams()->slength*simparams()->slength;
	// BC when using LJ
	m_physparams->dcoeff = 5.0*g*H;

	//	Wave paddle definition:  location, start & stop times, stroke and frequency (2 \pi/period)
	const double r0 = m_deltap;
	paddle_length = parser.getF("paddle_length") * lz;
	paddle_width = m_size.y - 2*r0;
	paddle_origin = make_double3(parser.getD("paddle_origin_x") * lx, 0.0, parser.getD("paddle_origin_z") * lz);
	paddle_tstart = parser.getF("paddle_time_start");
	paddle_tend = simparams()->tend;
	// The stroke value is given at free surface level H
	float stroke = parser.getF("paddle_stroke");
	// m_mbamplitude is the maximal angular value par paddle angle
	// Paddle angle is in [-m_mbamplitude, m_mbamplitude]
	paddle_amplitude = atan(stroke / (2.0 * (H - paddle_origin.z)));
	paddle_omega = 2.0 * M_PI / parser.getD("paddle_period"); // period T = 1.5 s

	// Initializing mooring points
	mooring[0] = make_double3(parser.getD("mooring_anchor0_x"), parser.getD("mooring_anchor0_y"), parser.getD("mooring_anchor0_z"));
	mooring[1] = make_double3(parser.getD("mooring_anchor1_x"), parser.getD("mooring_anchor1_y"), parser.getD("mooring_anchor1_z"));
	mooring[2] = make_double3(parser.getD("mooring_anchor2_x"), parser.getD("mooring_anchor2_y"), parser.getD("mooring_anchor2_z"));
	mooring[3] = make_double3(parser.getD("mooring_anchor3_x"), parser.getD("mooring_anchor3_y"), parser.getD("mooring_anchor3_z"));
	// chain data
	chain_uw = parser.getD("chain_uw");
	// and chain lengths
	chain_length[0] = parser.getD("chain_length0");
	chain_length[1] = parser.getD("chain_length1");
	chain_length[2] = parser.getD("chain_length2");
	chain_length[3] = parser.getD("chain_length3");

	// Initialize ODE
	dInitODE();
	m_ODEWorld = dWorldCreate();
	m_ODESpace = dHashSpaceCreate(0);
	m_ODEJointGroup = dJointGroupCreate(0);
	dWorldSetGravity(m_ODEWorld, m_physparams->gravity.x, m_physparams->gravity.y, m_physparams->gravity.z);	// Set gravity(x, y, z)

	// Drawing and saving times
	add_writer(VTKWRITER, 0.1);
	add_writer(HOTWRITER, 0.1);
	add_writer(CALLBACKWRITER, 0.0);

	// Name of problem used for directory creation
	m_name = "CSIDevice";
}

CSIDevice::~CSIDevice(void)
{
	release_memory();
}

void
CSIDevice::release_memory(void) {
	parts.clear();
}


// Return f(alpha, lt, lb) where alpha = T/uw, lt = total length of the chain
// lb = chain projected length on the (x,y) plane, h = height.
// Since uw, lt, lb and h are known, solving f = 0 for alpha gives the tension T0
// at the sea bed.
double
CSIDevice::f(const double alpha, const double lt, const double lb, const double h) {
	return cosh(sqrt(alpha*h*(alpha*h + 2)) - alpha*(lt - lb)) - alpha*h - 1.;
}

// Derivative of f respect to alpha. Used in find_tension.
double
CSIDevice::dfdalpha(const double alpha, const double lt, const double lb, const double h) {
	const double temp = sqrt(alpha*h*(alpha*h + 2));
	return (h*(alpha*h + 1)/temp - (lt - lb))*sinh(temp - alpha*(lt - lb)) - h;
}


// Solve f(alpha, lt, lb, h) = 0 for alpha with Newton method.
double
CSIDevice::find_tension(const double Ti, const double lt, const double lb, const double h) {
	double alpha = chain_uw/Ti;
	double alphas = alpha;

	do {
		alphas = alpha;
		alpha = alpha - f(alpha, lt, lb, h)/dfdalpha(alpha, lt, lb, h);
	} while (abs((alpha - alphas)/alpha) > FLT_EPSILON);

	return chain_uw/alpha;
}

// This method is called before the computation of the platform movement by ODE.
// We will apply here the forces and moments due to mooring updating the forces and torques
// array.
void
CSIDevice::bodies_forces_callback(const double t0, const double t1, const uint step, float3 *forces, float3 *torques) {
	MovingBodyData* mbdata = m_bodies[0];

	// Get approximate water height at the body
	double height = mbdata->kdata.crot.z;

	// Solve mooring for each attachment point
	for (int i = 0; i < 4; i++) {
		// First compute the position of the attachment point in the rest frame
		double3 p = mbdata->kdata.orientation.Rot(attachment_object_frame[i]);
		attachment_rest_frame[i] = p + mbdata->kdata.crot;
		double2 hp = make_double2(mooring[i] - attachment_rest_frame[i]);

		// Height of the mooring
		double h = attachment_rest_frame[i].z - mooring[i].z;
		//cout << "h = " << h << "\n";

		// Now compute mooring line projected length
		const double lb = length(hp);
		//cout << "lb = " << lb << "\n";

		// Solve for tension
		const double T0 = find_tension(mooring_tension[i], chain_length[i], lb, h);
		//cout << "T0 = " << T0 << "\n";

		// Computing force and torque on floating body
		double3 F;

		// Tension at the upper point of the mooring line (bottom of floating body)
		const double T = T0 + chain_uw*h;

		// Effective length of the mooring line
		const double leff = sqrt(h*(h + 2*T0/chain_uw));
		chain_leff[i] = leff;

		// Compute length of the sea bed resting portion of the chain
		chain_hlength[i] = chain_length[i] - leff;
		//cout << "lb = " << chain_hlength[i] << "\n";

		// First vertical component of force
		// Angle between mooring line and horizontal plane
		const double theta = atan(chain_uw*leff/T0);
		F.z = -T*sin(theta);
		const double Fh = T*cos(theta);
		hp = normalize(hp);
		F.x = T*hp.x;
		F.y = T*hp.y;

		// Save tension mooring tension and force
		mooring_tension[i] = T0;
		mooring_force[i] = F;

		// Update forces and torques on the floating platform
		forces[0] += make_float3(F);
		torques[0] += make_float3(cross(p, F));
	}
}

// This method is called at each time step. We use it to output the joint forces
// on the swinging sphere and the tension of each mooring cable at each time step
// in the joint_feedback.txt file.
// For visualisation purposes we also write the attachment points in mooring.txt
// but only when we output a VTK file.
void
CSIDevice::writer_callback(CallbackWriter *cbw,
	uint numParts, BufferList const&, uint node_offset, double t,
	const bool testpoints) const
{
	static std::ofstream& main_out = cbw->open_data_file("joint_feedback", string(), ".txt");
	static bool first_write = false;

	if (!first_write) {
		main_out << "t\tFx\tFy\tFz\tMx\tMy\tMz\tT0\tT1\tT2\tT3" << endl;
		first_write = true;
	}
	// Write joint feed back and tension of mooring lines at each time step
	main_out << t << "\t" << jointFb.f1[0] << "\t" << jointFb.f1[1] << "\t" << jointFb.f1[2] << "\t";
	main_out << jointFb.t1[0] << "\t" << jointFb.t1[1] << "\t" << jointFb.t1[2];

	for (int i = 0; i < 4; i++)
		main_out << "\t" << mooring_force[i].x << "\t" << mooring_force[i].y << "\t" << mooring_force[i].z << "\n";

	const Writer *vtk = cbw->get_other_writer(VTKWRITER);

	// Each time we output a VTK file we write mooring points on sea bed and on platform
	// respect to the rest frame in the mooring.txt file. The later will be used in Paraview
	// to render the mooring lines.
	if (vtk) {
		// VTK has written something
		std::ofstream& vtk_out = cbw->open_data_file("mooring", vtk->last_filenum(), ".txt");
		vtk_out << chain_uw << "," << 0 << "," << 0 << "\n";

		for (int i = 0; i < 4; i++)
			vtk_out << mooring[i].x << "," << mooring[i].y << "," << mooring[i].z << "\n";

		for (int i = 0; i < 4; i++)
			vtk_out << attachment_rest_frame[i].x << "," << attachment_rest_frame[i].y << "," << attachment_rest_frame[i].z << "\n";

		for (int i = 0; i < 4; i++)
			vtk_out << mooring_tension[i] << "," << 0 << "," << 0 << "\n";

		for (int i = 0; i < 4; i++)
			vtk_out << chain_length[i] << "," << 0 << "," << 0 << "\n";

		for (int i = 0; i < 4; i++)
			vtk_out << chain_leff[i] << "," << 0 << "," << 0 << "\n";

		vtk_out.close();
	}
}


// Build the platform with base + two cylinder and a swinging sphere
void
CSIDevice::build() {
	const int layers = 3;

	double water_level = parser.getD("water_height");

	// get floating platform metrics
	double platform_x = parser.getD("platform_x");
	double platform_y = parser.getD("platform_y");
	double platform_l = parser.getD("platform_l");
	double platform_w = parser.getD("platform_w");
	double platform_h = parser.getD("platform_h");
	double platform_sg = parser.getD("platform_sg");

	// Mooring points on the platform in local coordinate system
	// i.e. respect to the geometrical center
	attachment_object_frame[0] = make_double3(-platform_l/2.0, platform_w/2.0, -platform_h/2.0);
	attachment_object_frame[1] = make_double3(platform_l/2.0, platform_w/2.0, -platform_h/2.0);
	attachment_object_frame[2] = make_double3(platform_l/2.0, -platform_w/2.0, -platform_h/2.0);
	attachment_object_frame[3] = make_double3(-platform_l/2.0, -platform_w/2.0, -platform_h/2.0);

	// get tower metrics
	double tower_diameter = parser.getD("tower_diameter");
	double tower_height = parser.getD("tower_height");
	double tower_width = parser.getD("tower_width");

	// swing metrics
	double swing_diameter = parser.getD("swing_diameter");
	double swing2_diameter = parser.getD("swing2_diameter");

	// create the main floating platform
	platform = Cube(Point(platform_x, platform_y, water_level), platform_l, platform_w, platform_h);
	platform.SetPartMass(m_deltap, m_physparams->rho0[0]);
	platform.SetMass(m_deltap, m_physparams->rho0[0]*platform_sg);
	platform.Unfill(parts, m_deltap);
	platform.FillIn(platform.GetParts(), m_deltap, layers);
	platform.ODEBodyCreate(m_ODEWorld, m_deltap);
	//platform.ODEGeomCreate(m_ODESpace, m_deltap);
	dBodySetLinearVel(platform.ODEGetBody(), 0.0, 0.0, 0.0);
	dBodySetAngularVel(platform.ODEGetBody(), 0.0, 0.0, 0.0);
	add_moving_body(&platform, MB_ODE);

	// create first tower
	double xloc = platform_x + (platform_l / 2.0);
	double yloc = platform_y + (platform_w / 2.0) - (tower_width / 2.0);
	tower0 = Cylinder(Point(xloc, yloc, water_level + platform_h), tower_diameter/2.0, tower_height);
	tower0.SetPartMass(m_deltap, m_physparams->rho0[0]);
	tower0.SetMass(m_deltap, m_physparams->rho0[0]);
	tower0.Unfill(parts, m_deltap);
	tower0.FillIn(tower0.GetParts(), m_deltap, layers);
	tower0.ODEBodyCreate(m_ODEWorld, m_deltap);
	//mtower0.ODEGeomCreate(m_ODESpace, m_deltap);
	add_moving_body(&tower0, MB_ODE);
	dJointID bid = dJointCreateFixed(m_ODEWorld, 0);
	joints.push_back(bid);
	dJointAttach(bid, tower0.m_ODEBody,  platform.m_ODEBody);
	dJointSetFixed(bid);

	// create second tower
	yloc = platform_y + (platform_w / 2.0) + (tower_width / 2.0);
	tower1 = Cylinder(Point(xloc, yloc, water_level + platform_h), tower_diameter/2.0, tower_height);
	tower1.SetPartMass(m_deltap, m_physparams->rho0[0]);
	tower1.SetMass(m_deltap, m_physparams->rho0[0]);
	tower1.Unfill(parts, m_deltap);
	tower1.FillIn(tower1.GetParts(), m_deltap, layers);
	tower1.ODEBodyCreate(m_ODEWorld, m_deltap);
	//mtower0.ODEGeomCreate(m_ODESpace, m_deltap);
	add_moving_body(&tower1, MB_ODE);
	bid = dJointCreateFixed(m_ODEWorld, 0);
	joints.push_back(bid);
	dJointAttach(bid, tower1.m_ODEBody,  platform.m_ODEBody);
	dJointSetFixed(bid);

	// crossbar
	crossbar = Cylinder(Point(xloc, yloc-tower_width, water_level + platform_h + tower_height),
			tower_diameter/2.0, tower_width, EulerParameters(Vector(1, 0, 0), -M_PI/2.0));
	crossbar.SetPartMass(m_deltap, m_physparams->rho0[0]);
	crossbar.SetMass(m_deltap, m_physparams->rho0[0]);
	crossbar.Unfill(parts, m_deltap);
	crossbar.FillIn(crossbar.GetParts(), m_deltap, layers);
	crossbar.ODEBodyCreate(m_ODEWorld, m_deltap);
	//crossbar.ODEGeomCreate(m_ODESpace, m_deltap);
	add_moving_body(&crossbar, MB_ODE);

	// attach crossbar
	bid = dJointCreateFixed(m_ODEWorld, 0);
	joints.push_back(bid);
	dJointAttach(bid, crossbar.m_ODEBody, tower0.m_ODEBody);
	dJointSetFixed(bid);
	bid = dJointCreateFixed(m_ODEWorld, 0);
	joints.push_back(bid);
	dJointAttach(bid, crossbar.m_ODEBody, tower1.m_ODEBody);
	dJointSetFixed(bid);

	// Create swing
	yloc = platform_y + (platform_w / 2.0);
	double swing_length = parser.getD("swing_length");
	double zloc = water_level + platform_h + tower_height - swing_length - swing_diameter/2.0;
	swing1 = Sphere(Point(xloc, yloc, zloc), swing_diameter/2.0);
	swing1.SetPartMass(m_deltap, m_physparams->rho0[0]);
	swing1.SetMass(m_deltap, m_physparams->rho0[0]*parser.getD("swing_sg"));
	swing1.Unfill(parts, m_deltap);
	swing1.FillIn(swing1.GetParts(), m_deltap, layers);
	swing1.ODEBodyCreate(m_ODEWorld, m_deltap);
	//swing.ODEGeomCreate(m_ODESpace, m_deltap);
	dBodySetLinearVel(swing1.ODEGetBody(), 0.0, 0.0, 0.0);
	dBodySetAngularVel(swing1.ODEGetBody(), 0.0, 0.0, 0.0);
	add_moving_body(&swing1, MB_ODE);

	// Attach swing
	bid = dJointCreateBall(m_ODEWorld, 0);
	dJointAttach(bid, swing1.m_ODEBody, crossbar.m_ODEBody);
	joints.push_back(bid);
	dJointSetFeedback(bid, &jointFb);
	zloc = water_level + platform_h + tower_height - (tower_diameter / 2.0);
	dJointSetBallAnchor(bid, xloc, yloc, zloc);

	// Compute mooring at t = 0
	float3 forces, torques;
	// Initialize mooring tension
	for (int i = 0; i < 4; i++)
		mooring_tension[i] = chain_uw;
	bodies_forces_callback(0, 0, 0, &forces, &torques);
}


void
CSIDevice::moving_bodies_callback(const uint index, Object* object, const double t0, const double t1,
			const float3& force, const float3& torque, const KinematicData& initial_kdata,
			KinematicData& kdata, double3& dx, EulerParameters& dr)
{
	dx = make_double3(0.0);
	kdata.lvel = make_double3(0.0);
	if (t1 >= paddle_tstart) {
		kdata.avel = make_double3(0.0, paddle_amplitude*sin(paddle_omega*(t1)), 0.0);
		EulerParameters depdt  = 0.5*EulerParameters(kdata.avel)*kdata.orientation;
		EulerParameters new_orientation = kdata.orientation + (t1 - t0)*depdt;
		new_orientation.Normalize();
		dr = EulerParameters::Identity() + (t1 - t0)*depdt*kdata.orientation.Inverse();
		dr.Normalize();
		kdata.orientation = new_orientation;
	}
}


void CSIDevice::ODE_near_callback(void *data, dGeomID o1, dGeomID o2)
{
	const int N = 10;
	dContact contact[N];

	int n = dCollide(o1, o2, N, &contact[0].geom, sizeof(dContact));
	for (int i = 0; i < n; i++) {
		contact[i].surface.mode = dContactBounce;
		contact[i].surface.mu = dInfinity;
		contact[i].surface.bounce = 0.0; // (0.0~1.0) restitution parameter
		contact[i].surface.bounce_vel = 0.0; // minimum incoming velocity for bounce
		dJointID c = dJointCreateContact(m_ODEWorld, m_ODEJointGroup, &contact[i]);
		dJointAttach (c, dGeomGetBody(contact[i].geom.g1), dGeomGetBody(contact[i].geom.g2));
	}
}


int CSIDevice::fill_parts()
{
	const float dp = m_deltap;
	const float width = ly;
	const int layers = 4;		// Number of dynamic particles layers

	boundary_parts.reserve(1000);
	parts.reserve(34000);

	paddle = Cube(Point(paddle_origin.x - (layers - 1)*dp, paddle_origin.y, paddle_origin.z + layers *dp),
		(layers - 1)*dp, width, 0.7*lz);

	paddle.SetPartMass(dp, m_physparams->rho0[0]);
#if 0 // fixed paddle
	paddle.Fill(boundary_parts, dp, true);
#else // moving paddle
	paddle.Fill(paddle.GetParts(), dp, true);
	add_moving_body(&paddle, MB_MOVING);
	set_body_cg(&paddle, paddle_origin);
#endif

	// Filling the horizontal portion of bottom boundary with layers number of particles
	Cube h_bound = Cube(Point(-0.5, 0, 0), h_length + 2*dp + 0.5, width, (layers - 1)*dp, EulerParameters());
	h_bound.SetPartMass(dp, m_physparams->rho0[0]);
	h_bound.Fill(boundary_parts, dp, true);

	// Filling the sloped portion of bottom boundary with layers number of particles
	Cube slope_bound = Cube(Point(h_length, 0, 0), slope_length/cos(beta), width, (layers - 1)*dp,
			EulerParameters(Vector(0, 1, 0), -beta));
	// Unfilling the intersection of the horizontal part and sloped one before
	// filling the sloped part
	slope_bound.Unfill(boundary_parts, 0.4*dp);
	slope_bound.SetPartMass(dp, m_physparams->rho0[0]);
	slope_bound.Fill(boundary_parts, dp, true);

	// Filling the fluid part
	Rect fluid;
	double z = 0;
	int n = 0;
	double l = 0;
	while (z < H ) {
		//while (z < H && l < 2.0) {
		z = n*dp + layers*dp;
		double x = paddle_origin.x + dp;
		l = h_length + z/tan(beta) - x;
		fluid = Rect(Point(x, 0, z),
			Vector(0, width, 0), Vector(l, 0, 0));
		fluid.SetPartMass(m_deltap, m_physparams->rho0[0]);
		fluid.Fill(parts, m_deltap, true);
		n++;
	}
	H = z;

	slope_bound.Unfill(parts, m_physparams->r0);

	build();

	return parts.size() + boundary_parts.size() + get_bodies_numparts();

}

void CSIDevice::copy_to_array(BufferList &buffers)
{
	float4 *pos = buffers.getData<BUFFER_POS>();
	hashKey *hash = buffers.getData<BUFFER_HASH>();
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();

	std::cout << "\nBoundary parts: " << boundary_parts.size() << "\n";
		std::cout << "      "<< 0  <<"--"<< boundary_parts.size() << "\n";
	for (uint i = 0; i < boundary_parts.size(); i++) {
		float ht = H - boundary_parts[i](2);
		if (ht < 0)
			ht = 0.0;
		float rho = density(ht, 0);
		vel[i] = make_float4(0, 0, 0, rho);
		info[i]= make_particleinfo(PT_BOUNDARY, 0, i);  // first is type, object, 3rd id
		calc_localpos_and_hash(boundary_parts[i], info[i], pos[i], hash[i]);
	}
	int j = boundary_parts.size();
	std::cout << "Boundary part mass:" << pos[j-1].w << "\n";

	uint object_particle_counter = 0;
	for (uint k = 0; k < m_bodies.size(); k++) {
		PointVect & rbparts = m_bodies[k]->object->GetParts();
		std::cout << "Rigid body " << k << ": " << rbparts.size() << " particles ";
		for (uint i = 0; i < rbparts.size(); i++) {
			uint ij = i + j;
			float ht = H - rbparts[i](2);
			if (ht < 0)
				ht = 0.0;
			float rho = density(ht, 0);
			rho = m_physparams->rho0[0];
			vel[ij] = make_float4(0, 0, 0, rho);
			uint ptype = (uint) PT_BOUNDARY;
			switch (m_bodies[k]->type) {
				case MB_ODE:
					ptype |= FG_MOVING_BOUNDARY | FG_COMPUTE_FORCE;
					break;
				case MB_FORCES_MOVING:
					ptype |= FG_COMPUTE_FORCE | FG_MOVING_BOUNDARY;
					break;
				case MB_MOVING:
					ptype |= FG_MOVING_BOUNDARY;
					break;
			}
			info[ij] = make_particleinfo(ptype, k, ij);
			calc_localpos_and_hash(rbparts[i], info[ij], pos[ij], hash[ij]);
		}
		if (k < simparams()->numforcesbodies) {
			gdata->s_hRbFirstIndex[k] = -j + object_particle_counter;
			gdata->s_hRbLastIndex[k] = object_particle_counter + rbparts.size() - 1;
			object_particle_counter += rbparts.size();
		}
		j += rbparts.size();
		std::cout << ", part mass: " << pos[j-1].w << "\n";
	}

	std::cout << "\nFluid parts: " << parts.size() << "\n";
	std::cout << "      "<< j  <<"--"<< j+ parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		float ht = H - parts[i-j](2);
		if (ht < 0)
			ht = 0.0;
		float rho = density(ht, 0);
		vel[i] = make_float4(0, 0, 0, rho);
		info[i]= make_particleinfo(PT_FLUID,0,i);
		calc_localpos_and_hash(parts[i - j], info[i], pos[i], hash[i]);
	}
	j += parts.size();
	std::cout << "Fluid part mass:" << pos[j-1].w << "\n";

	std::cout << " Everything uploaded" <<"\n";
	std::flush(std::cout);
}
#undef MK_par
