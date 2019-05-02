/*  Copyright 2011-2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

    Istituto Nazionale di Geofisica e Vulcanologia
        Sezione di Catania, Catania, Italy

    Università di Catania, Catania, Italy

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

#include <iostream>

#include "OpenChannel.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

// TODO this problem should be turned into a generic validation problem
// similar to Poiseuille:
// * allow runtime selection of density, kinematic visc and driving force;
// * only use driving force in the direction of the flow (no vertical component)
// * script to check the resulting profile against the expected (analytical) one
// * when generalized Newtonian rheologies get implemented, we should make this
//   default to a non-Newtonian rheology, so that checking all problems will
//   have one such problem to verify against too
OpenChannel::OpenChannel(GlobalData *_gdata) : XProblem(_gdata)
{
	use_side_walls = get_option("sidewalls", true);

	SETUP_FRAMEWORK(
		//viscosity<ARTVISC>,
		viscosity<KINEMATICVISC>,
		boundary<DYN_BOUNDARY>,
		periodicity<PERIODIC_XY>
	).select_options(
		use_side_walls, periodicity<PERIODIC_X>()
	);

	// SPH parameters
	set_deltap(0.02f);
	set_timestep(0.00004f);
	simparams()->dtadaptfactor = 0.3;
	simparams()->buildneibsfreq = 10;
	simparams()->tend = 20;

	H = 0.5; // water level

	if (simparams()->boundarytype == DYN_BOUNDARY) {
		dyn_layers = ceil(simparams()->influenceRadius/m_deltap) + 1;
		// no extra offset in the X direction, since we have periodicity there
		// no extra offset in the Y direction either if we do NOT have side walls
		dyn_offset = dyn_layers*make_double3(0,
			use_side_walls ? m_deltap : 0,
			m_deltap);
		margin = make_double3(0., use_side_walls ? 0.1 : 0, 0.1);
	} else {
		dyn_layers = 0;
		dyn_offset = make_double3(0.0);
		margin = make_double3(0.0);
	}

	// Size and origin of the simulation domain
	a = round_up(1.0, m_deltap);
	h = round_up(H*1.4, m_deltap);
	l = round_up(15*simparams()->influenceRadius, m_deltap);

	m_size = make_double3(l, a, h) + 2*make_double3(margin.x, margin.y, margin.z);
	m_origin = make_double3(0.0, 0.0, 0.0) - make_double3(margin.x, margin.y, margin.z);

	// Physical parameters
	const double angle = 4.5; // angle in degrees
	const float g = 9.81f;
	set_gravity(g*sin(M_PI*angle/180), 0.0, -g*cos(M_PI*angle/180));

	add_fluid(2650.0f);
	set_equation_of_state(0,  2.0f, 20.f);
	set_dynamic_visc(0, 110.f);

	physparams()->dcoeff = 5.0f*g*H;

	physparams()->epsartvisc = 0.01*simparams()->slength*simparams()->slength;
	//set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5

	// Drawing and saving times
	add_writer(VTKWRITER, 0.5);

	// Name of problem used for directory creation
	m_name = "OpenChannel";

	// Building the geometry
	setPositioning(PP_CORNER);
	const float r0 = m_deltap;
	// gap due to periodicity
	const double3 periodicity_gap = make_double3(m_deltap/2,
		use_side_walls ? 0 : m_deltap/2, 0);

	if (use_side_walls) {
		// side walls: shifted by dyn_offset, and with opposite orientation so that
		// they "fill in" towards the outside
		GeometryID sideWall1 = addBox(GT_FIXED_BOUNDARY, FT_BORDER,
			periodicity_gap + make_double3(0, a, 0),
			l - r0, 4*margin.y, h - r0);
		disableCollisions(sideWall1);

		GeometryID sideWall2 = addBox(GT_FIXED_BOUNDARY, FT_BORDER,
			periodicity_gap + make_double3(0, -dyn_offset.y, 0),
			l - r0, 1.5*margin.y, h - r0);
		disableCollisions(sideWall2);

		if (simparams()->boundarytype == DYN_BOUNDARY) {
			GeometryID unfill_sideWall1 = addBox(GT_FIXED_BOUNDARY, FT_NOFILL,
					periodicity_gap + make_double3(0, a + dyn_offset.y + r0, 0),
					l - r0, 4*margin.y, h - r0);
			disableCollisions(unfill_sideWall1);
			setEraseOperation(unfill_sideWall1, ET_ERASE_BOUNDARY);
			GeometryID unfill_sideWall2 = addBox(GT_FIXED_BOUNDARY, FT_NOFILL,
					periodicity_gap + make_double3(0, r0, 0),
					l - r0, 1.5*margin.y, h - r0);
			disableCollisions(unfill_sideWall2);
			setEraseOperation(unfill_sideWall2, ET_ERASE_BOUNDARY);
		}
	}

	// bottom: it must cover the whole bottom floor, including under the walls,
	// hence it must not be shifted by dyn_offset in the y direction. In the
	// Y-periodic case (no side walls), the Y length must be decreased by
	// a deltap to account for periodicity (start at deltap/2, end deltap/2 before the end)
	GeometryID bottom = addBox(GT_FIXED_BOUNDARY, FT_BORDER,
		periodicity_gap + make_double3(0, -dyn_offset.y, -2*dyn_offset.z),
		l - r0, a + 2*dyn_offset.y, 2*dyn_offset.z);
	disableCollisions(bottom);
	if (simparams()->boundarytype == DYN_BOUNDARY) {
		GeometryID unfill_bottom = addBox(GT_FIXED_BOUNDARY, FT_NOFILL,
				periodicity_gap + make_double3(0, -dyn_offset.y, -2*dyn_offset.z),
				l - r0, a + 2*dyn_offset.y, dyn_offset.z);
		disableCollisions(unfill_bottom);
		setEraseOperation(unfill_bottom, ET_ERASE_BOUNDARY);
	}

	if (simparams()->periodicbound == PERIODIC_XY) {
		addBox(GT_FLUID, FT_SOLID,
				periodicity_gap + make_double3(0, 0, r0), l - r0, a - r0, H - r0) ;
	} else {
		addBox(GT_FLUID, FT_SOLID,
				periodicity_gap + make_double3(0, r0, r0), l - r0, a - 2*r0, H - r0) ;
	}

}

