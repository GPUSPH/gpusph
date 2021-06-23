/*  Copyright (c) 2019 INGV, EDF, UniCT, JHU

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
#include <string>
#include <iostream>

#include "StillWaterRepackSA.h"
#include "GlobalData.h"
#include "cudasimframework.cu"
#include "utils.h"

StillWaterRepackSA::StillWaterRepackSA(GlobalData *_gdata) : Problem(_gdata)
{
	SETUP_FRAMEWORK(
		kernel<WENDLAND>,
		viscosity<DYNAMICVISC>,
		boundary<SA_BOUNDARY>,
		periodicity<PERIODIC_NONE>,
		add_flags<ENABLE_DTADAPT | ENABLE_REPACKING | ENABLE_GAMMA_QUADRATURE>
	);

	// *** Initialization of minimal physical parameters
	set_deltap(0.01);
	set_smoothing(1.3);
	set_gravity(-9.81);

	// *** Initialization of minimal simulation parameters
	resize_neiblist(128+128, 64);

	// *** ferrari correction
	simparams()->ferrariLengthScale = 0.05; //low values help stabilize the problem

	// *** buildneibs at every iteration
	simparams()->buildneibsfreq = 1;

	// *** Other parameters and settings
	simparams()-> tend = 10.f;
	add_writer(VTKWRITER, 0.001f);
	m_name = "StillWaterRepackSA";

	// Repacking options
	simparams()->repack_maxiter = 10;
	simparams()->repack_a = 0.1;
	simparams()->repack_alpha = 0.1;

	m_origin = make_double3(0., 0., 0.);
	m_size = make_double3(1., 1., 1.);

	//*** Fluid and Thermodynamic Properties
	add_fluid(1000.0);
	set_kinematic_visc(0, 1.e-6); //check the parameters
	set_equation_of_state(0, 7.0f,20.f); //check the parameters

	//*** Add the Fluid
	addHDF5File(GT_FLUID, Point(0,0,0), "./data_files/StillWaterRepackSA/0.stillWaterRepackSA.fluid.h5sph", NULL);

	//*** Add the Main Container
	addHDF5File(GT_FIXED_BOUNDARY, Point(0,0,0), "./data_files/StillWaterRepackSA/0.stillWaterRepackSA.kent0.h5sph", NULL);

	//*** Add the free-surface
	addHDF5File(GT_FREE_SURFACE, Point(0,0,0), "./data_files/StillWaterRepackSA/0.stillWaterRepackSA.kent1.h5sph", NULL);

}

void StillWaterRepackSA::fillDeviceMap()
{
	fillDeviceMapByAxis(X_AXIS);
}
