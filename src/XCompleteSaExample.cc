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

#include "XCompleteSaExample.h"
#include "Cube.h"
#include "Point.h"
#include "Vector.h"
#include "GlobalData.h"

XCompleteSaExample::XCompleteSaExample(GlobalData *_gdata) : XProblem(_gdata)
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
	m_simparams.visctype = DYNAMICVISC;
	// boundary types: LJ_BOUNDARY, MK_BOUNDARY, SA_BOUNDARY, DYN_BOUNDARY
	m_simparams.boundarytype = SA_BOUNDARY;

	// *** Other parameters and settings
	add_writer(VTKWRITER, 1e-2f);
	m_name = "XCompleteSaExample";

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


	m_origin = make_double3(-1, -1, -1);
	m_size = make_double3(3, 3, 3);

	// Set world size and origin like CompleteSaExample, instead of computing automatically.
	// Also, HDF5 file loading does not support bounding box detection yet
	const double MARGIN = 0.1;
	const double INLET_BOX_LENGTH = 0.25;
	// size of the main cube, exlcuding the inlet and any margin
	double box_l, box_w, box_h;
	box_l = box_w = box_h = 1.0;
	// world size
	double world_l = box_l + INLET_BOX_LENGTH + 2 * MARGIN; // length is 1 (box) + 0.2 (inlet box length)
	double world_w = box_w + 2 * MARGIN;
	double world_h = box_h + 2 * MARGIN;
	m_origin = make_double3(- INLET_BOX_LENGTH - MARGIN, - MARGIN, - MARGIN);
	m_size = make_double3(world_l, world_w ,world_h);

	// add "universe box" of planes
	//makeUniverseBox(m_origin, m_origin + m_size );
}


void XCompleteSaExample::fillDeviceMap()
{
	fillDeviceMapByAxis(Y_AXIS);
}
