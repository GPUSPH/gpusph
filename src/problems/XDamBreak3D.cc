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

#include <cmath>
#include <iostream>

#include "XDamBreak3D.h"
#include "Cube.h"
#include "Point.h"
#include "Vector.h"
#include "GlobalData.h"

XDamBreak3D::XDamBreak3D(GlobalData *_gdata) : XProblem(_gdata)
{
	// viscosities: ARTVISC, KINEMATICVISC, DYNAMICVISC, SPSVISC, KEPSVISC
	// boundary types: LJ_BOUNDARY, MK_BOUNDARY, SA_BOUNDARY, DYN_BOUNDARY
	SETUP_FRAMEWORK(
		viscosity<ARTVISC>,
		boundary<LJ_BOUNDARY>
	);

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
	//m_simparams.dtadapt = true;

	// *** Other parameters and settings
	add_writer(VTKWRITER, 1e-2f);
	m_name = "XDamBreak3D";

	const double dimX = 4;
	const double dimY = 2;
	const double dimZ = 2;
	const double obstacle_radius = 0.1;
	const double sphere_radius = 0.2;
	const double water_length = dimX / 4;
	const double water_height = dimZ / 2;

	// set positioning policy to PP_CORNER: given point will be the corner of the geometry
	setPositioning(PP_CORNER);

	// add water
	addBox(GT_FLUID, FT_SOLID, Point(0,0,0), water_length, dimY, water_height);

	// set positioning policy to PP_BOTTOM_CENTER: given point will be the center of the base
	setPositioning(PP_BOTTOM_CENTER);

	// add a few of obstacles
	const uint NUM_OBSTACLES = 3;
	const double Y_DISTANCE = dimY / (NUM_OBSTACLES + 1);
	// will rotate along Y axis by...
	const double Y_ANGLE = - M_PI / 8;

	for (uint i = 0; i < NUM_OBSTACLES; i++) {
		GeometryID obstacle =
			addCylinder(GT_FIXED_BOUNDARY, FT_BORDER, Point(dimX/2, Y_DISTANCE * (i+1), 0), obstacle_radius, dimZ);
		rotate(obstacle, 0, Y_ANGLE, 0);
	}

	// set positioning policy to PP_CENTER: given point will be the geometrical center of the object
	setPositioning(PP_CENTER);

	// now add a floating (actually drowning...) object on the floor
	GeometryID floating_obj =
		addSphere(GT_FLOATING_BODY, FT_BORDER, Point(water_length, dimY/2, water_height), sphere_radius);
	// half water density to make it float
	setMassByDensity(floating_obj, m_physparams.rho0[0] / 2);
	setParticleMassByDensity(floating_obj, m_physparams.rho0[0] / 2);
	// disable collisions: will only interact with fluid
	disableCollisions(floating_obj);

	// note: if we do not use makeUniverseBox(), origin and size will be computed automatically
	m_origin = make_double3(0, 0, 0);
	m_size = make_double3(dimX, dimY, dimZ);

	// limit domain with 6 planes
	makeUniverseBox(m_origin, m_origin + m_size);
}


void XDamBreak3D::fillDeviceMap()
{
	fillDeviceMapByAxis(Y_AXIS);
}
