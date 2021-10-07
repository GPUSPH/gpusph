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

#include "StandingWave2D.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

StandingWave2D::StandingWave2D(GlobalData *_gdata) : Problem(_gdata)
{
	// *** user parameters from command line
	const int mlsIters = get_option("mls", 0); // --mls N to enable MLS filter every N iterations
	const DensityDiffusionType RHODIFF = get_option("density-diffusion", DELTA_SPH);
	const uint ppH = get_option("ppH", 64);
	const bool USE_CCSPH = get_option("use_ccsph", true);

	// *** Framework setup
	SETUP_FRAMEWORK(
		space_dimensions<R2>,
		periodicity<PERIODIC_X>,
		viscosity<KINEMATICVISC>,
		boundary<DUMMY_BOUNDARY>,
		add_flags<ENABLE_INTERNAL_ENERGY>
	).select_options(
		RHODIFF,
		USE_CCSPH, add_flags<ENABLE_CCSPH>()
	);

	if (mlsIters > 0)
		addFilter(MLS_FILTER, mlsIters);

	addPostProcess(SURFACE_DETECTION);
	H = 1.0f; // Water level

	set_deltap(H/ppH);

	setMaxFall(H);

	l = H;
	h = 1.1*H;

	// SPH parameters
	simparams()->dtadaptfactor = 0.3;
	simparams()->tend = 20.0;
	simparams()->buildneibsfreq = 10;
	simparams()->ferrariLengthScale = H;

	dyn_thickness = 3.0*m_deltap;

	// Physical parameters
	set_gravity(-9.81f);
	g = get_gravity_magnitude();
	const float maxvel = sqrt(2*g*H);
	// purely for cosmetic reason, let's round the soundspeed to the next
	// integer
	const float c0 = ceil(20*maxvel);
	add_fluid(1000.0);
	set_equation_of_state(0, 1.0f, c0);

	set_kinematic_visc(0, 1.0e-6f);
	physparams()->artvisccoeff = 1e-6*10.0/(physparams()->sscoeff[0]*simparams()->slength);

	// Setting the standing wave from eq. 8 in Antuono et al 2011
	L = 1.0f; // Wavelenghth
//	A = 0.04f; // Wave amplitude
	//A = 2.0*m_deltap; // Wave amplitude
	A = 2.0/128.0; // Wave amplitude

	// derived variables
	k = 2*M_PI/L; // Wave number
	omega = sqrt(g*k*tanh(k*H)); //circular frequency
	printf("Omega = %g\n", omega);

	const double T = 2*M_PI/omega;

	add_gage (l/2.0);

	// Drawing and saving times
	add_writer(VTKWRITER, T/4.0);
	add_writer(COMMONWRITER, 0.01);

	// Name of problem used for directory creation
	m_name = "StandingWave2D";

	// Building the geometry
	setPositioning(PP_CORNER);

	const double half_dp = 0.5*m_deltap;
	GeometryID domain_box = addRect(GT_FIXED_BOUNDARY, FT_SOLID,
			Point(half_dp, -3.5*m_deltap, 0), l-m_deltap, 3*m_deltap);

	m_fluidOrigin = make_double3(half_dp, half_dp, 0.0);

	GeometryID fluid = addRect(GT_FLUID, FT_SOLID,
		m_fluidOrigin, l - m_deltap, H);
}

void StandingWave2D::initializeParticles(BufferList &buffer, const uint numParticle)
	{


		double4 *gpos = buffer.getData<BUFFER_POS_GLOBAL>();
		float4 *pos = buffer.getData<BUFFER_POS>();
		float4 *vel = buffer.getData<BUFFER_VEL>();
		const ushort4 *pinfo = buffer.getData<BUFFER_INFO>();


		const float epsilon = 2*A/H;
		const float c = -epsilon*H*g*k/(2*omega*cosh(k*H));

		for (uint i = 0 ; i < numParticle ; i++) {
			if (FLUID(pinfo[i])){

				double4 pg = gpos[i];
				vel[i].x = -c*cosh(k*(pg.y))*sin(k*pg.x);
				vel[i].y = c*sinh(k*(pg.y))*cos(k*pg.x);
				pos[i].w = physical_density(vel[i].w, 0)*m_deltap*m_deltap;
			}
		}
	}

void StandingWave2D::fillDeviceMap()
	{
		fillDeviceMapByAxis(X_AXIS);
	}


