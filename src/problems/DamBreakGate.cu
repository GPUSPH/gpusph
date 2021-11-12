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

#include <iostream>

#include "DamBreakGate.h"
#include "cudasimframework.cu"

#define SIZE_X		(1.60)
#define SIZE_Y		(0.67)
#define SIZE_Z		(0.40)

// default: origin in 0,0,0
#define ORIGIN_X	(0)
#define ORIGIN_Y	(0)
#define ORIGIN_Z	(0)


DamBreakGate::DamBreakGate(GlobalData *_gdata) : Problem(_gdata)
{
	// command-line options and gate setup
	const int ppH = get_option("ppH", 32);

	H = 0.4;

	// initial (from tstart) gate velocity (v0)
	// TODO we might want to “smooth in” from 0 to gate_velocity between 0 and tstart
	gate_vel = get_option("gate-velocity", 0.0);
	// gate acceleration at tstart (a)
	gate_accel = get_option("gate-accel", 4.0);
	gate_start = get_option("gate-start", 0.1); // gives time to settle
	// after tstart, the gate's motion is defined by
	// vz = v0 + a (t - tstart)
	// rz = r0 + v0 (t - start) + a/2 (t - tstart)^2
	// we want to stop when rz - r0 = H. Simple calculations give us
	// a (t - tstart)^2 + 2v0 (t - tstart) - 2H = 0
	// i.e.
	// t = tstart + sqrt(2 H/a + (v0/a)^2) - (v0/a)
	{
		const double gate_v_a_ratio = gate_vel/gate_accel;
		gate_end = get_option("gate-end", gate_start + (
				gate_accel == 0 ?
				(gate_vel == 0 ? 0 : H/gate_vel) :
				(sqrt(2*H/gate_accel + gate_v_a_ratio*gate_v_a_ratio) - gate_v_a_ratio)
			)
		);
	}


	SETUP_FRAMEWORK(
		viscosity<ARTVISC>,//SPSVISC, or DYNAMICVISC
		boundary<LJ_BOUNDARY>,
		//boundary<DYN_BOUNDARY>,
		add_flags<ENABLE_MOVING_BODIES>
	);

	// geometry
	const double water_length = 0.4;
	const double obstacle_width = 0.12;
	const double obstacle_center = 0.9 + obstacle_width/2;

	// SPH parameters
	set_deltap(H/ppH);
	simparams()->dtadaptfactor = 0.3;
	simparams()->buildneibsfreq = 10;
	simparams()->tend = 4;

	//addFilter(MLS_FILTER, 10);

	// Physical parameters
	set_gravity(-9.81f);
	add_fluid(1000.0);
	set_equation_of_state(0,  7.0f, NAN); // auto compute speed of sound

	set_kinematic_visc(0, 1.0e-2f);

	{
		const double max_gate_vel = gate_vel + gate_accel*(gate_end - gate_start);
		// let the API know that we're going to get at least this fast
		setMaxParticleSpeed(max_gate_vel);
	}


	// Drawing and saving times
	add_writer(VTKWRITER, 0.005f);

	// Building the geometry
	const Point corner(ORIGIN_X, ORIGIN_Y, ORIGIN_Z);
	const Point gate_corner(corner + Vector(water_length, 0, 0));
	const Point obstacle_pos(ORIGIN_X + obstacle_center, SIZE_Y/2, 0);

	setFillingMethod(Object::BORDER_TANGENT);
	setPositioning(PP_CORNER);

	// The experiment box is composed of a box, minus the top, which is unfilled with
	// a second geometry
	// TODO Problem API 2 should expose filling options such as the specification of which walls to fill
	addBox(GT_FIXED_BOUNDARY, FT_OUTER_BORDER, corner, SIZE_X, SIZE_Y, SIZE_Z + m_deltap);
	// cutting plane
	addPlane(0, 0, -1, ORIGIN_Z + SIZE_Z + m_deltap/2, FT_UNFILL);

	// Gate: note that this is OUTER border because it should fill out outwards wrt to the fluid
	auto gate = addRect(GT_MOVING_BODY, FT_OUTER_BORDER, gate_corner, H, SIZE_Y);
	rotate(gate, 0, M_PI/2, 0);
	// Since the gate moves up, we need to expand the domain in the z direction.
	// addExtraWorldMargin() adds room everywhere, so instead we add a small cube at the height
	// where the gate will finish its motion
	auto padder = addCube(GT_FIXED_BOUNDARY, FT_SOLID, gate_corner + Vector(0, 0, 2*H), 0);

	// Obstacle
	setPositioning(PP_BOTTOM_CENTER);
	addBox(GT_FIXED_BOUNDARY, FT_INNER_BORDER, obstacle_pos, obstacle_width, obstacle_width, H);

	// Fluid
	setPositioning(PP_CORNER);
	addBox(GT_FLUID, FT_SOLID, corner, water_length, SIZE_Y, H);

	bool wet = false;	// set wet to true have a wet bed experiment
	if (wet) {
		const double wet_height = 0.3;
		addBox(GT_FLUID, FT_SOLID, gate_corner, SIZE_X - water_length, SIZE_Y, wet_height);
	}

}

void
DamBreakGate::moving_bodies_callback(const uint index, Object* object, const double t0, const double t1,
			const float3& force, const float3& torque, const KinematicData& initial_kdata,
			KinematicData& kdata, double3& dx, EulerParameters& dr)
{
	// Computing, at t = t1, new position of center of rotation (here only translation)
	// along with linear velocity and displacement of center of rotation
	// These are non-zero only during motion, i.e. when the “next” time is after the beginning,
	// and the “prev” time is before the end
	if (t1 >= gate_start && t0 <= gate_end) {
		// we need to clamp t1 and t0 between tstart and tend
		const double clamped_t1 = min(t1, gate_end);
		const double clamped_t0 = max(t0, gate_start);
		const double dt1 = clamped_t1 - gate_start;
		const double dt0 = clamped_t0 - gate_start;

		kdata.lvel = make_double3(0.0, 0.0, gate_vel + gate_accel*dt1);
		kdata.crot.z = initial_kdata.crot.z + gate_vel*dt1 + gate_accel*0.5*dt1*dt1;

		// displacement of center of rotation between t1 and t0
		dx.z = gate_vel*(clamped_t1 - clamped_t0) + gate_accel*0.5*(dt1*dt1 - dt0*dt0);
	} else {
		kdata.lvel = make_double3(0.0f);
		dx.z = 0;
	}

	// Setting angular velocity at t = t1 and the rotation between t = t0 and t = 1.
	// Here we have a simple translation movement so the angular velocity is null and
	// the rotation between t0 and t1 equal to identity.
	kdata.avel = make_double3(0.0f);
	dr.Identity();
}
