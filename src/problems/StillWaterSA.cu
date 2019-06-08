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

#include <iostream>

#include "StillWaterSA.h"
#include "GlobalData.h"
#include "cudasimframework.cu"
#include "textures.cuh"
#include "utils.h"
#include <string>

StillWaterSA::StillWaterSA(GlobalData *_gdata) : Problem(_gdata)
{
	SETUP_FRAMEWORK(
		kernel<WENDLAND>,
		formulation<SPH_F1>,
		viscosity<DYNAMICVISC>,
		boundary<SA_BOUNDARY>,
		periodicity<PERIODIC_NONE>,
		densitydiffusion<BREZZI>,
		add_flags<ENABLE_DTADAPT | ENABLE_DENSITY_SUM>
	);

	// *** Initialization of minimal physical parameters
	set_deltap(0.05);
	set_smoothing(1.3);
	set_gravity(-9.81);

	// *** Initialization of minimal simulation parameters
	resize_neiblist(128+128, 64);

	// *** ferrari correction
	//simparams()->ferrariLengthScale = 0.05; //low values help stabilize the problem
	simparams()->densityDiffCoeff = 0.05;

	// *** buildneibs at every iteration
	//simparams()->buildneibsfreq = 1;

	// *** Other parameters and settings
	simparams()->tend = 10.0f;
	add_writer(VTKWRITER, 0.01f);
	add_writer(COMMONWRITER, 0.f);
	m_name = "StillWaterSA";

	m_origin = make_double3(-0.5,-0.5,0.0);
	m_size = make_double3(1, 1, 1.1);

	//*** Fluid and Thermodynamic Properties
	add_fluid(1000.0);
	set_kinematic_visc(0, 0.00001f);  	//check the parameters
	set_equation_of_state(0, 7.0f,12.f); //check the parameters

	//*** Add the Fluid
	addHDF5File(GT_FLUID, Point(0,0,0), "./data_files/StillWaterSA/0.Hydrostatic_20.fluid.h5sph", NULL);

	//*** Add the Main Container
	GeometryID container = addHDF5File(GT_FIXED_BOUNDARY, Point(0,0,0), "./data_files/StillWaterSA/0.Hydrostatic_20.boundary.kent0.h5sph", NULL);
	disableCollisions(container);

}

void StillWaterSA::fillDeviceMap()
{
	fillDeviceMapByAxis(X_AXIS);
}









