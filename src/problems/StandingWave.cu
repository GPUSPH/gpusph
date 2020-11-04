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

#include "StandingWave.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

#define CENTER_DOMAIN 0
// set to coords (x,y,z) if more accuracy is needed in such point
// (waiting for relative coordinates)
#if CENTER_DOMAIN
#define OFFSET_X (-l/2)
#define OFFSET_Y (-w/2)
#define OFFSET_Z (-h/2)
#else
#define OFFSET_X 0
#define OFFSET_Y 0
#define OFFSET_Z 0
#endif

StandingWave::StandingWave(GlobalData *_gdata) : Problem(_gdata)
{
	m_usePlanes = get_option("use-planes", false); // --use-planes true to enable use of planes for boundaries
	const int mlsIters = get_option("mls", 0); // --mls N to enable MLS filter every N iterations
	const int ppH = get_option("ppH", 128); // --ppH N to change deltap to H/N
	const bool USE_CSPM = get_option("use_cspm", true);

	// density diffusion terms, see DensityDiffusionType
	const DensityDiffusionType rhodiff = get_option("density-diffusion", COLAGROSSI);

	SETUP_FRAMEWORK(
		periodicity<PERIODIC_XY>,
		//kernel<GAUSSIAN>,
		boundary<DYN_BOUNDARY>,
		add_flags<ENABLE_INTERNAL_ENERGY/* | ENABLE_XSPH*/>
	).select_options(
		rhodiff,
		USE_CSPM, add_flags<ENABLE_CSPM>()
		//m_usePlanes, add_flags<ENABLE_PLANES>()
	);

	if (mlsIters > 0)
		addFilter(MLS_FILTER, mlsIters);

	addPostProcess(SURFACE_DETECTION);
	H = 1.0f; // Water level

	set_deltap(H/ppH);

	setMaxFall(H);

	l = H;
	w = round_up(3*simparams()->influenceRadius, m_deltap);
	h = 1.1*H;

//	resize_neiblist(300);// FIXME temp

	// Size and origin of the simulation domain
	m_size = make_double3(l, w ,h);
	m_origin = make_double3(OFFSET_X, OFFSET_Y, OFFSET_Z);

	// SPH parameters
	simparams()->dtadaptfactor = 0.3;
	simparams()->tend = 20.0;
	simparams()->buildneibsfreq = 10;
	simparams()->ferrariLengthScale = H;
	//physparams()->artvisccoeff = 0.05;

	// enlarge the domain to take into account the extra layers of particles
	// of the boundary
	if ((simparams()->boundarytype == DUMMY_BOUNDARY || simparams()->boundarytype == DYN_BOUNDARY) && !m_usePlanes) {
		// number of layers
		dyn_layers = ceil(simparams()->kernelradius*simparams()->sfactor);
		dyn_thickness = (dyn_layers - 1)*m_deltap;
		// extra layers are one less (since other boundary types still have
		// one layer)
		double3 extra_offset = make_double3(0.0f, 0.0f, dyn_thickness);
		m_origin -= extra_offset;
		m_size += 2*extra_offset;
	} else {
		dyn_layers = 0;
	}


	if (simparams()->boundarytype == SA_BOUNDARY) {
		resize_neiblist(128, 128);
	};

	// Physical parameters
	set_gravity(-9.81f);
	g = get_gravity_magnitude();
	const float maxvel = sqrt(2*g*H);
	// purely for cosmetic reason, let's round the soundspeed to the next
	// integer
	const float c0 = ceil(20*maxvel);
	add_fluid(1000.0);
	set_equation_of_state(0, 7.0f, c0);

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



	add_gage (l/2.0, w/2.0);
	// Drawing and saving times
	add_writer(VTKWRITER, T/4.0);
	add_writer(COMMONWRITER, 0.01);

	// Name of problem used for directory creation
	m_name = "StandingWave";

	// Building the geometry
	setPositioning(PP_CORNER);

	GeometryID experiment_box = addBox(GT_FIXED_BOUNDARY, FT_BORDER,
		Point(m_origin) - Point(0.0, 0.0, dyn_thickness), m_size.x - m_deltap, m_size.y - m_deltap, dyn_thickness);
	disableCollisions(experiment_box);

	m_fluidOrigin = m_origin;
	// shift by the extra offset of the experiment box
	//m_fluidOrigin += make_double3(0.5*m_deltap, 0.5*m_deltap, 0.69*m_deltap); // tuned for LJ to avoid initial collapse
	m_fluidOrigin += make_double3(0.5*m_deltap, 0.5*m_deltap, m_deltap); // tuned for LJ to avoid initial collapse

	GeometryID fluid = addBox(GT_FLUID, FT_SOLID,
		m_fluidOrigin, l - m_deltap, w - m_deltap, H);

}

void StandingWave::copy_planes(PlaneList& planes)
{
	if (!m_usePlanes) return;

	planes.push_back( implicit_plane(0, 0, 1.0, -m_origin.z) );
	planes.push_back( implicit_plane(0, 1.0, 0, -m_origin.x) );
	planes.push_back( implicit_plane(0, -1.0, 0, m_origin.x + w) );
	planes.push_back( implicit_plane(1.0, 0, 0, -m_origin.y) );
	planes.push_back( implicit_plane(-1.0, 0, 0, m_origin.y + l) );
}


void StandingWave::initializeParticles(BufferList &buffer, const uint numParticle)
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
			vel[i].x = -c*cosh(k*(pg.z))*sin(k*pg.x);
			vel[i].z = c*sinh(k*(pg.z))*cos(k*pg.x);
			pos[i].w = physical_density(vel[i].w, 0)*m_deltap*m_deltap*m_deltap;
		}
	}
}

void StandingWave::fillDeviceMap()
{
	fillDeviceMapByAxis(X_AXIS);
}
