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

/*
   Example use of dynamic boundaries. Implement a 2D Poiseuille flow
   with a double-periodic domain (X for flow direction, Y to eliminate side
   boundary effects). Top and bottom layers are implemented as dynamic boundary
   particles.
 */



#include "DynBoundsExample.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

#include "Cube.h"

DynBoundsExample::DynBoundsExample(GlobalData *_gdata) : Problem(_gdata)
{
	W = 1; // 2D cell side
	H = 2*W; // still water height

	SETUP_FRAMEWORK(
		boundary<DYN_BOUNDARY>,
		viscosity<DYNAMICVISC>,
		periodicity<PERIODIC_XY>,
		add_flags<ENABLE_REPACKING>
	);

	set_deltap(W/64);
	resize_neiblist(128);

	w = m_deltap*4;

	m_size = make_double3(W, W, H + 2*w);
	m_origin = -m_size/2;

	simparams()->tend = 2;

	simparams()->repack_maxiter = 1000;

	/* slope */
	float degs = 60; /* degrees */
	alpha = M_PI*degs/180; /* radians */

	float g = 9.81f;
	set_gravity(g*sin(alpha), 0, -g*cos(alpha));

	float maxvel = sqrt(g*H);

	add_fluid(1);
	set_equation_of_state(0,  7, 10*maxvel);
	set_kinematic_visc(0, 120);

	add_writer(VTKWRITER, 0.01);

	m_name = "DynBoundsExample";

	// Building the geometry
	setPositioning(PP_CORNER);
	GeometryID fluid = addBox(GT_FLUID, FT_SOLID,
		m_origin + make_double3(m_deltap/2., m_deltap/2., w+m_deltap), W-m_deltap, W-m_deltap, H-2*m_deltap);

	GeometryID bp1 = addBox(GT_FIXED_BOUNDARY, FT_BORDER,
		m_origin, W, W, w);
	disableCollisions(bp1);

	GeometryID bp2 = addBox(GT_FIXED_BOUNDARY, FT_BORDER,
		m_origin + make_double3(0, 0, H + w), W, W, w);
	disableCollisions(bp2);

	// Print information
	float flowvel = H*H*fabs(get_gravity().x)/(8*physparams()->kinematicvisc[0]);
	printf("Expected maximum flow velocity: %f\n", flowvel);
}

// Density initialization
	void
DynBoundsExample::initializeParticles(BufferList &buffers, const uint numParticles)
{
	// 1. warn the user if this is expected to take much time
	printf("Initializing particles density...\n");

	// 2. grab the particle arrays from the buffer list
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();
	double4 *pos = buffers.getData<BUFFER_POS_GLOBAL>();

	// 3. iterate on the particles
	for (uint i = 0; i < numParticles; i++) {
		// 5. set in loco the desired values
		float ht = m_origin.z + H+2*w - pos[i].z;
		ht *= cos(alpha);
		vel[i].w = hydrostatic_density(ht, 0);
	}
}

