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

#include "Spheric2LJ.h"
#include "Cube.h"
#include "Point.h"
#include "Vector.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

#define CENTER_DOMAIN 1
// set to coords (x,y,z) if more accuracy is needed in such point
// (waiting for relative coordinates)
#if CENTER_DOMAIN
#define OFFSET_X (-lx/2)
#define OFFSET_Y (-ly/2)
#define OFFSET_Z (-lz/2)
#else
#define OFFSET_X 0
#define OFFSET_Y 0
#define OFFSET_Z 0
#endif

Spheric2LJ::Spheric2LJ(GlobalData *_gdata) : Problem(_gdata)
{
	// Size and origin of the simulation domain
	lx = 3.22;
	ly = 1.0;
	lz = 1.0;
	H = 0.55;
	wet = false;
	m_usePlanes = get_option("use-planes", false);

	m_size = make_double3(lx, ly, lz);
	m_origin = make_double3(OFFSET_X, OFFSET_Y, OFFSET_Z);

	SETUP_FRAMEWORK(
		kernel<WENDLAND>,
		viscosity<ARTVISC>,
		//viscosity<SPSVISC>,
		//viscosity<DYNAMICVISC>,
		boundary<LJ_BOUNDARY>,
		densitydiffusion<FERRARI>
	).select_options(
		m_usePlanes, add_flags<ENABLE_PLANES>()
	);

	// SPH parameters
	// ratio h / deltap (needs to be defined before calling set_deltap)
	simparams()->sfactor = 1.3;
	// set deltap (automatically computes h based on sfactor * deltap)
	set_deltap(0.02); //0.008
	simparams()->dtadaptfactor = 0.3;
	simparams()->buildneibsfreq = 10;
	simparams()->densityDiffCoeff = 0.1;
	simparams()->tend = 1.0f;

	// Free surface detection
	addPostProcess(SURFACE_DETECTION);

	// Test points
	addPostProcess(TESTPOINTS);

	// Physical parameters
	set_gravity(-9.81f);
	setMaxFall(H);

	add_fluid(1000.0);
	set_equation_of_state(0,  7.0f, 20.f);

	set_kinematic_visc(0, 1.0e-2f);

	// Drawing and saving times
	add_writer(VTKWRITER, 0.05);

	// Name of problem used for directory creation
	m_name = "Spheric2LJ";

	// Building the geometry
	setPositioning(PP_CORNER);
	float r0 = m_deltap;

	if (!m_usePlanes) {
		GeometryID experiment_box = addBox(GT_FIXED_BOUNDARY, FT_BORDER,
			Point(m_origin), lx, ly, lz);
		disableCollisions(experiment_box);
	}

	GeometryID obstacle = addBox(GT_FIXED_BOUNDARY, FT_BORDER,
		Point(m_origin + make_double3(2.3955, 0.295, r0)), 0.161, 0.403, 0.161-r0);
	GeometryID unfill_obstacle = addBox(GT_FIXED_BOUNDARY, FT_NOFILL,
		Point(m_origin + make_double3(2.3955+r0, 0.295+r0, r0)), 0.161-2*r0, 0.403-2*r0, 0.161-r0);
	setEraseOperation(unfill_obstacle, ET_ERASE_BOUNDARY);

	addBox(GT_FLUID, FT_SOLID, Point(m_origin + r0), 0.4, ly - 2*r0, H - r0);

	if (wet) {
		addBox(GT_FLUID, FT_SOLID, Point(m_origin + r0 + make_double3(H + m_deltap, 0, 0)),
			lx - H - m_deltap - 2*r0, 0.67 - 2*r0, 0.1);
		// this is done here because in the dry case we don't need to do the unfill
		// (there's no water around the obstacle anyway)
		setEraseOperation(obstacle, ET_ERASE_FLUID);
	}

	// Setting probes for Spheric2 test case
	//*******************************************************************
	// Wave gages
	add_gage(m_origin + make_double3(2.724, 0.5, 0.0));
	add_gage(m_origin + make_double3(2.228, 0.5, 0.0));
	add_gage(m_origin + make_double3(1.732, 0.5, 0.0));
	add_gage(m_origin + make_double3(0.582, 0.5, 0.0));

	// Pressure probes
	addTestPoint(m_origin + make_double3(2.3955, 0.529, 0.021));
	addTestPoint(m_origin + make_double3(2.3955, 0.529, 0.061));
	addTestPoint(m_origin + make_double3(2.3955, 0.529, 0.101));
	addTestPoint(m_origin + make_double3(2.3955, 0.529, 0.141));
	addTestPoint(m_origin + make_double3(2.4165, 0.471, 0.161));
	addTestPoint(m_origin + make_double3(2.4565, 0.471, 0.161));
	addTestPoint(m_origin + make_double3(2.4965, 0.471, 0.161));
	addTestPoint(m_origin + make_double3(2.5365, 0.471, 0.161));
	//*******************************************************************

}

void Spheric2LJ::copy_planes(PlaneList& planes)
{
	if (!m_usePlanes) return;

	// bottom
	planes.push_back( implicit_plane(0, 0, 1.0, -m_origin.z) );
	// back
	planes.push_back( implicit_plane(1.0, 0, 0, -m_origin.x) );
	// front
	planes.push_back( implicit_plane(-1.0, 0, 0, m_origin.x + lx) );
	// side with smaller Y ("left")
	planes.push_back( implicit_plane(0, 1.0, 0, -m_origin.y) );
	// side with greater Y ("right")
	planes.push_back( implicit_plane(0, -1.0, 0, m_origin.y + ly) );
}

void Spheric2LJ::fillDeviceMap()
{
	// TODO: test which split performs better, if Y (not many particles passing) or X (smaller section)
	fillDeviceMapByAxis(Y_AXIS);
	//fillDeviceMapByEquation();
}

