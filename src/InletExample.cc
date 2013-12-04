/*  Copyright 2011 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

	Istituto de Nazionale di Geofisica e Vulcanologia
          Sezione di Catania, Catania, Italy

    Universita di Catania, Catania, Italy

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

#include "InletExample.h"

#define PERIODIC 0		// set to 1 for periodic boundaries, 0 for non-periodic
#define OUTFLOW_ONLY 0	// set to 1 to have outlet only, 0 to have inlet and outlet
#define HYDROSTATIC 1	// set to 1 for hydrostatic density, 0 otherwise

#define VEL_X 1.0
#define VEL_Y 0.0
#define VEL_Z 0.0

InletExample::InletExample(const Options &options) : Problem(options)
{
	// Size and origin of the simulation domain
	l = 8.0;
	w = 1.0;
	h = 0.7;
	H = 0.5;

	m_size = make_float3(l, w ,h);
	m_origin = make_float3(-l/2, -w/2, -h/2);

	m_writerType = VTKWRITER;

	// SPH parameters
	set_deltap(0.05f);
	m_simparams.slength = 1.3f*m_deltap;
	m_simparams.kernelradius = 2.0f;
	m_simparams.kerneltype = WENDLAND;
	m_simparams.dt = 0.00004f;
	m_simparams.xsph = false;
	m_simparams.dtadapt = true;
	m_simparams.dtadaptfactor = 0.3;
	m_simparams.buildneibsfreq = 10;
	m_simparams.shepardfreq = 0;
	m_simparams.mlsfreq = 10;
#if 0
	m_simparams.visctype = ARTVISC;
#else
	m_simparams.visctype = KINEMATICVISC;
#endif
	m_simparams.mbcallback = false;
	m_simparams.tend = 20;

	// Physical parameters
	m_physparams.gravity = make_float3(0, 0, -9.81);
	float g = length(m_physparams.gravity);
	float maxvel = sqrt(g*H);

	m_physparams.set_density(0, 1000, 7.0f, 20.f*maxvel);
	m_physparams.dcoeff = 5.0f*g*H;

	m_physparams.r0 = m_deltap;

	m_physparams.kinematicvisc = 1.0e-6f;

	m_physparams.epsartvisc = 0.01*m_simparams.slength*m_simparams.slength;
	//set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5

#if PERIODIC
	m_simparams.periodicbound = true;
	m_physparams.dispvect = make_float3(l, 0.0, 0.0);
	m_physparams.minlimit = make_float3(0.0f, 0.0f, 0.0f);
	m_physparams.maxlimit = make_float3(l, 0.0f, 0.0f);
#endif

	// create an outlet area ending on x = l
	// with an outflow displacement vector in the direction
	// normal to the x = l plane
	float disp = 2*m_simparams.slength*m_simparams.kernelradius;
	add_outlet(
		m_origin + make_float3(l - disp, 0, 0),
		m_origin + m_size,
		make_float3(1, 0, 0));

	add_inlet(
		m_origin,
		m_origin + make_float3(2*disp, w, h),
		make_float4(VEL_X, VEL_Y, VEL_Z, NAN));

	// Free surface detection
	m_simparams.surfaceparticle = false;
	m_simparams.savenormals = false;

	// Drawing and saving times
	m_displayinterval = 0.001f;
	m_writefreq = 10;
	m_screenshotfreq = 0;

	// Name of problem used for directory creation
	m_name = "InletExample";
	create_problem_dir();
}


InletExample::~InletExample(void)
{
	release_memory();
}


void InletExample::release_memory(void)
{
	parts.clear();
}


int InletExample::fill_parts()
{
	float r0 = m_physparams.r0;

	experiment_box = Cube(Point(m_origin), Vector(l, 0, 0), Vector(0, w, 0), Vector(0, 0, h));
#if OUTFLOW_ONLY
	Cube fluid = Cube(Point(m_origin + r0), Vector(l/2 - m_deltap, 0, 0), Vector(0, w - 2*r0, 0), Vector(0, 0, H - r0));
#else
	Cube fluid = Cube(Point(m_origin + make_float3(m_deltap/2.0, r0, r0)), Vector(l - m_deltap, 0, 0), Vector(0, w - 2*r0, 0), Vector(0, 0, H - r0));
#endif

	parts.reserve(14000);

	fluid.SetPartMass(m_deltap, m_physparams.rho0[0]);
	fluid.Fill(parts, m_deltap, true);

	return parts.size();
}

uint InletExample::fill_planes()
{
	// without inlet there's an extra plane
	return 3 + OUTFLOW_ONLY;
}

void InletExample::copy_planes(float4 *planes, float *planediv)
{
	uint pnum = 0;

#if OUTFLOW_ONLY
	// x = m_origin.x
	planes[pnum] = make_float4(1.0, 0.0, 0, -m_origin.x);
	planediv[pnum] = 1.0;
	++pnum;
#endif

	// z = m_origin.z
	planes[pnum] = make_float4(0, 0, 1.0, -m_origin.z);
	planediv[pnum] = 1.0;
	++pnum;
	// y = m_origin.y
	planes[pnum] = make_float4(0, 1.0, 0, -m_origin.y);
	planediv[pnum] = 1.0;
	++pnum;
	// y = m_origin.y + m_size.y
	planes[pnum] = make_float4(0, -1.0, 0, m_origin.y + m_size.y);
	planediv[pnum] = 1.0;
	++pnum;
}


void InletExample::copy_to_array(float4 *pos, float4 *vel, particleinfo *info)
{
	std::cout << "Fluid parts: " << parts.size() << "\n";
	for (uint i = 0; i < parts.size(); ++i) {
		pos[i] = make_float4(parts[i]);
#if HYDROSTATIC
		vel[i] = make_float4(VEL_X, VEL_Y, VEL_Z, density(H-(pos[i].z - m_origin.z), 0));
#else
		vel[i] = make_float4(VEL_X, VEL_Y, VEL_Z, m_physparams.rho0[0]);
#endif
		info[i]= make_particleinfo(FLUIDPART, 0, i);
	}
	std::cout << "Fluid part mass:" << pos[0].w << "\n";
}
