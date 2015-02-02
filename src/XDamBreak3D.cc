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

XDamBreak3D::XDamBreak3D(const GlobalData *_gdata) : XProblem(_gdata)
{
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
	m_simparams.boundarytype = LJ_BOUNDARY;

	// *** Other parameters and settings
	add_writer(VTKWRITER, 1e-2f);
	m_name = "XDamBreak3D";

	const double dimX = 4;
	const double dimY = 2;
	const double dimZ = 2;
	const double obstacle_side = 0.2;
	const double water_length = dimX / 4;
	const double water_height = dimZ / 4;

	// set positioning policy
	setPositioning(PP_CORNER);

	GeometryID water =
		addBox(GT_FLUID, FT_SOLID, Point(0,0,0), water_length, dimY, water_height);

	setPositioning(PP_BOTTOM_CENTER);

	GeometryID obstacle =
		addCylinder(GT_FIXED_BOUNDARY, FT_BORDER, Point(dimX/2, dimY/2, 0), obstacle_side, dimZ);

	// cetering policy is still PP_BOTTOM_CENTER
	//setPositioning(PP_CENTER);

	GeometryID floating_obj =
		addSphere(GT_FLOATING_BODY, FT_BORDER, Point(water_length,dimY/2,m_deltap), obstacle_side);
	setMassByDensity(floating_obj, m_physparams.rho0[0] / 2);

	// limit domain with 6 planes
	m_origin = make_double3(0, 0, 0);
	m_size = make_double3(dimX, dimY, dimZ);
	// note: if we do not use makeUniverseBox(), origin and size will be computed automatically
	makeUniverseBox(m_origin, m_origin + m_size);
}


void XDamBreak3D::fillDeviceMap()
{
	fillDeviceMapByAxis(Y_AXIS);
}
