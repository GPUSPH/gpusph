/*  Copyright 2015 Giuseppe Bilotta, Alexis Herault, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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


#include <iostream>

#include "Bubble.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

#define USE_PLANES 0

#if USE_PLANES
#define USE_GHOST 1 // set to 0 for standard planes
#else
#define USE_GHOST 0 // never use ghosts unless using planes
#endif



Bubble::Bubble(GlobalData *_gdata) : XProblem(_gdata),
	dyn_layers(0)
{
	// Size and origin of the simulation domain
	R = 0.025;
	H = 10*R;
	lx = ly = 6*R;
	lz = H;

	// TODO GAUSSIAN kernel of radius 3
	SETUP_FRAMEWORK(
		formulation<SPH_GRENIER>,
		viscosity<DYNAMICVISC>,
		boundary<DYN_BOUNDARY>,
		flags<ENABLE_DTADAPT | (USE_PLANES ? ENABLE_PLANES : ENABLE_NONE)>
	);

	// SPH parameters
	// Grenier sets h/R = 0.128
	//set_deltap(6.72e-4/1.3);
	set_deltap(0.128*R/1.3);

	if (simparams()->boundarytype == DYN_BOUNDARY) {
		dyn_layers = simparams()->get_influence_layers() + 1;
		extra_offset = make_double3(dyn_layers*m_deltap);
	} else {
		dyn_layers = 0;
		extra_offset = make_double3(0.0);
	}
	m_size = make_double3(lx, ly, lz) + 2*extra_offset;
	m_origin = -m_size/2;

	simparams()->buildneibsfreq = 10;

	simparams()->tend = 1.0;

	physparams()->epsinterface = 0.08;

	// Physical parameters
	physparams()->gravity = make_float3(0.0, 0.0, -9.81f);
	float g = length(physparams()->gravity);

	//set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5
	physparams()->dcoeff = 5.0f*g*H;

	physparams()->r0 = m_deltap;

	float maxvel = sqrt(g*H);
	float rho0 = 1;
	float rho1 = 1000;

	size_t air = add_fluid(rho0);
	size_t water = add_fluid(rho1);

	set_equation_of_state(air,  1.4, 198*maxvel);
	set_equation_of_state(water,  7.0f, 14*maxvel);

	set_kinematic_visc(air, 4.5e-3f);
	set_kinematic_visc(water, 3.5e-5f);

	physparams()->artvisccoeff = 0.3f;
	physparams()->epsartvisc = 0.01*simparams()->slength*simparams()->slength;

	// Drawing and saving times
	add_writer(VTKWRITER, 0.01);

	// Name of problem used for directory creation
	m_name = "Bubble";

	// Building the geometry
	float r0 = physparams()->r0;

	setPositioning(PP_CORNER);
	GeometryID experiment_box = addBox(GT_FIXED_BOUNDARY, FT_BORDER,
		Point(m_origin),
		m_size.x,	m_size.y, m_size.z);
	disableCollisions(experiment_box);
	setMassByDensity(experiment_box, physparams()->rho0[1]);

	GeometryID fluid = addBox(GT_FLUID, FT_SOLID,
		Point(m_origin + extra_offset),
		lx, ly, H);
	// the actual particle mass will be set during the
	// initializeParticles routine
	setMassByDensity(fluid, physparams()->rho0[0]);

}

void Bubble::copy_planes(PlaneList &planes)
{
#if USE_PLANES
	// z = m_origin.z
	planes.push_back( implicit_plane(0, 0, 1.0, -m_origin.z) );
	// z = m_origin.z+lz
	planes.push_back( implicit_plane(0, 0, -1.0, m_origin.z+lz) );
	// y = m_origin.y
	planes.push_back( implicit_plane(0, 1.0, 0, -m_origin.y) );
	// y = m_origin.y+ly
	planes.push_back( implicit_plane(0, -1.0, 0, m_origin.y+ly) );
	// x = m_origin.x
	planes.push_back( implicit_plane(1.0, 0, 0, -m_origin.x) );
	// x = m_origin.x+lx
	planes.push_back( implicit_plane(-1.0, 0, 0, m_origin.x+lx) );
#endif
}


// the bubble is initially located centered at 2R from the bottom.
bool is_inside(double3 const& origin, float R, double4 const& pt)
{
	return
		(pt.x*pt.x) +
		(pt.y*pt.y) +
		(pt.z - (origin.z+2*R))*(pt.z - (origin.z+2*R)) < R*R;
}

// Mass and density initialization
	void
Bubble::initializeParticles(BufferList &buffers, const uint numParticles)
{
	// Example usage

	// 1. warn the user if this is expected to take much time
	printf("Initializing particles density and mass...\n");

	// 2. grab the particle arrays from the buffer list
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();
	double4 *pos_global = buffers.getData<BUFFER_POS_GLOBAL>();
	float4 *pos = buffers.getData<BUFFER_POS>();

	// 3. iterate on the particles
	for (uint i = 0; i < numParticles; i++) {
		float rho = 1;
		double depth = H - pos_global[i].z + m_origin.z;
		if (FLUID(info[i])) {
			int fluid_idx = is_inside(m_origin, R, pos_global[i]) ? 0 : 1;
			// hydrostatic density: for the heavy fluid, this is simply computed
			// as the density that gives pressure rho g h, with h depth
			rho = density(depth, fluid_idx);
			// for the bubble, the hydrostatic density must be computed in a slightly
			// more complex way:
			if (fluid_idx == 0) {
				// interface: depth of center of the bubble corrected by
				// R^2 - horizontal offset squared
				// note: no correction by m_origin.z because we are only
				// interested in deltas
				float z_intf = 2*R + sqrtf(R*R
						- (pos_global[i].x)*(pos_global[i].x)
						- (pos_global[i].y)*(pos_global[i].y)
						);
				// pressure at interface, from heavy fluid
				float g = length(physparams()->gravity);
				float P = physparams()->rho0[1]*(H - z_intf)*g;
				// plus hydrostatic pressure from _our_ fluid
				P += physparams()->rho0[0]*(z_intf - pos_global[i].z + m_origin.z)*g;
				rho = density_for_pressure(P, 0);
			}
			info[i]= make_particleinfo(PT_FLUID, fluid_idx, i);
		} else if (BOUNDARY(info[i])) {
			rho = density(depth, 1);
			info[i]= make_particleinfo(PT_BOUNDARY, 1, i);
		}
		pos[i].w *= rho;
		vel[i].w = rho;
	}
}

