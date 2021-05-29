/*  Copyright (c) 2014-2019 INGV, EDF, UniCT, JHU

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
#include "BuoyancyTest.h"
#include <iostream>

#include "GlobalData.h"
#include "cudasimframework.cu"
#include "Cube.h"
#include "Sphere.h"
#include "Point.h"
#include "Vector.h"


BuoyancyTest::BuoyancyTest(GlobalData *_gdata) : Problem(_gdata)
{
	// Size and origin of the simulation domain
	double lx = 1.0;
	double ly = 1.0;
	double lz = 1.0;
	double H = 0.6;

	//m_size = make_double3(lx, ly, lz);
	//m_origin = make_double3(0.0, 0.0, 0.0);

	SETUP_FRAMEWORK(
		kernel<WENDLAND>,
		viscosity<ARTVISC>,
		//viscosity<SPSVISC>,
		//viscosity<KINEMATICVISC>,
		boundary<DYN_BOUNDARY>
	);

	// reduce autocomputed number of layers
	setDynamicBoundariesLayers(3);

	// SPH parameters
	set_deltap(0.04); //0.008
	set_timestep(0.0003f);
	simparams()->dtadaptfactor = 0.3;
	simparams()->buildneibsfreq = 1;
	simparams()->tend = 10.0f; //0.00036f

	// Physical parameters
	set_gravity(-9.81f);
	double g = get_gravity_magnitude();
	setMaxFall(H);

	add_fluid(1000.0);
	set_equation_of_state(0,  7.0f, 20.f);


	set_kinematic_visc(0, 1.0e-6f);

	add_writer(VTKWRITER, 0.1);

	const double offs = m_deltap * getDynamicBoundariesLayers();
	//addExtraWorldMargin(2*offs);

	setPositioning(PP_CORNER);

	// DYN fill is inwards, so when using a cube as a container must take care of offsets
	GeometryID cube = addBox(GT_FIXED_BOUNDARY, FT_BORDER, Point(0,0,0),
		lx + 2 * offs, ly + 2 * offs, lz + 2 * offs);
	disableCollisions(cube);

	GeometryID fluid = addBox(GT_FLUID, FT_SOLID, Point(offs, offs, offs), lx, ly, H);

	// TODO
	/*
	switch (object_type) {
		case 0: {
			olx, oly, olz = 10.0*m_deltap;
			cube  = Cube(Point(lx/2.0 - olx/2.0, ly/2.0 - oly/2.0, H/2.0 - olz/2.0), olx, oly, olz);
		case 1: {
			double r = 6.0*m_deltap;
			sphere = Sphere(Point(lx/2.0, ly/2.0, H/2.0 - r/4.0), r);
		case 2: // TORUS
	*/
	/*
	double R = lx * 0.2;
	double r = 4.0 * m_deltap;
	GeometryID torus = addTorus(GT_FLOATING_BODY, FT_BORDER, Point(lx/2.0, ly/2.0, H/2.0), R, r);
	setMassByDensity(torus, physparams()->rho0[0]*0.5);
	*/
	setPositioning(PP_CENTER);
	const double SIDE = lx * 0.4;
	GeometryID floating_cube = addCube(GT_FLOATING_BODY, FT_BORDER,
		Point(offs + lx/2.0, offs + ly/2.0, offs + H/2.0), SIDE);
	setMassByDensity(floating_cube, physparams()->rho0[0]*0.5);

	// Name of problem used for directory creation
	m_name = "BuoyancyTest";
}
