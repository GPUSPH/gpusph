/*  Copyright (c) 2015-2019 INGV, EDF, UniCT, JHU

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

#include "Spheric2SA.h"
#include "GlobalData.h"
#include "cudasimframework.cu"
#include "textures.cuh"
#include "utils.h"

#define USE_PLANES 0

Spheric2SA::Spheric2SA(GlobalData *_gdata) : Problem(_gdata)
{
	SETUP_FRAMEWORK(
		viscosity<KEPSVISC>,
		//viscosity<DYNAMICVISC>,
		boundary<SA_BOUNDARY>,
		periodicity<PERIODIC_NONE>,
		kernel<WENDLAND>,
		densitydiffusion<FERRARI>,
		add_flags<ENABLE_GAMMA_QUADRATURE>
	);

	set_deltap(0.02715f);

	size_t water = add_fluid(1000.0);
	set_equation_of_state(water,  7.0f, 130.f);
	set_kinematic_visc(water, 1.0e-6f);
	set_gravity(-9.81f);

	simparams()->tend = 1.2;
	addPostProcess(SURFACE_DETECTION);
	addPostProcess(TESTPOINTS);
	H = 0.55;
	l = 3.5+0.02; w = 1.0+0.02; h = 2.0;
	m_origin = make_double3(-0.01, -0.01, -0.01);
	simparams()->ferrariLengthScale = 0.161f;
	resize_neiblist(128+128, 64);
	setMaxFall(H);

	// SPH parameters
	simparams()->dtadaptfactor = 0.3;
	simparams()->buildneibsfreq = 1;
	simparams()->nlexpansionfactor = 1.1;

	// Size and origin of the simulation domain
	m_size = make_double3(l, w ,h);

	// Physical parameters
	float g = get_gravity_magnitude();

	// Drawing and saving times
	add_writer(VTKWRITER, 1e-2f);

	// Name of problem used for directory creation
	m_name = "Spheric2SA";

	// Building the geometry
	addHDF5File(GT_FLUID, Point(0,0,0), "./data_files/Spheric2/0.spheric2.fluid.h5sph", NULL);

	GeometryID container =
		addHDF5File(GT_FIXED_BOUNDARY, Point(0,0,0), "./data_files/Spheric2/0.spheric2.boundary.kent0.h5sph", NULL);
	disableCollisions(container);

	// Add water level gages
	add_gage(m_origin + make_double3(2.724, 0.5, 0.0) + make_double3(0.01, 0.01, 0.01));
	add_gage(m_origin + make_double3(2.228, 0.5, 0.0) + make_double3(0.01, 0.01, 0.01));
	add_gage(m_origin + make_double3(1.732, 0.5, 0.0) + make_double3(0.01, 0.01, 0.01));
	add_gage(m_origin + make_double3(0.582, 0.5, 0.0) + make_double3(0.01, 0.01, 0.01));

	// Pressure probes
	addTestPoint(m_origin + make_double3(2.3955, 0.5, 0.021) + make_double3(0.01, 0.01, 0.01)); // the (0.01,0.01,0.01) vector accounts for the slightly shifted origin
	addTestPoint(m_origin + make_double3(2.3955, 0.5, 0.061) + make_double3(0.01, 0.01, 0.01));
	addTestPoint(m_origin + make_double3(2.3955, 0.5, 0.101) + make_double3(0.01, 0.01, 0.01));
	addTestPoint(m_origin + make_double3(2.3955, 0.5, 0.141) + make_double3(0.01, 0.01, 0.01));
	addTestPoint(m_origin + make_double3(2.4165, 0.5, 0.161) + make_double3(0.01, 0.01, 0.01));
	addTestPoint(m_origin + make_double3(2.4565, 0.5, 0.161) + make_double3(0.01, 0.01, 0.01));
	addTestPoint(m_origin + make_double3(2.4965, 0.5, 0.161) + make_double3(0.01, 0.01, 0.01));
	addTestPoint(m_origin + make_double3(2.5365, 0.5, 0.161) + make_double3(0.01, 0.01, 0.01));

}

void
Spheric2SA::initializeParticles(BufferList &buffers, const uint numParticles)
{
	printf("k and epsilon initialization...\n");

	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();
	double4 *pos = buffers.getData<BUFFER_POS_GLOBAL>();
	float *k = buffers.getData<BUFFER_TKE>();
	float *epsilon = buffers.getData<BUFFER_EPSILON>();

	for (uint i = 0; i < numParticles; i++) {
		const float Ti = 0.01f;
		const float u = 1.0f; // TODO set according to initial velocity
		const float L = 1.0f; // TODO set according to geometry
		if (k && epsilon) {
			k[i] = fmaxf(1e-5f, 3.0f/2.0f*(u*Ti)*(u*Ti));
			epsilon[i] = fmaxf(1e-5f, 2.874944542f*k[i]*u*Ti/L);
			//k[i] = k0;
			//e[i] = 1.0f/0.41f/fmax(1.0f-fabs(z),0.5f*(float)m_deltap);
		}
	}
}

uint
Spheric2SA::max_parts(uint numpart)
{
	// gives an estimate for the maximum number of particles
	return numpart;
}

void Spheric2SA::fillDeviceMap()
{
	fillDeviceMapByAxis(X_AXIS);
}

bool Spheric2SA::need_write(double t) const
{
 	return 0;
}




