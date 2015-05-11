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


#include <cmath>
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



Bubble::Bubble(GlobalData *_gdata) : Problem(_gdata),
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
		boundary<DYN_BOUNDARY>
	);

	// SPH parameters
	// Grenier sets h/R = 0.128
	//set_deltap(6.72e-4/1.3);
	set_deltap(0.128*R/1.3);

	if (m_simparams->boundarytype == DYN_BOUNDARY) {
		dyn_layers = ceil(m_simparams->kerneltype*m_simparams->sfactor);
		extra_offset = make_double3(dyn_layers*m_deltap);
	} else {
		extra_offset = make_double3(0.0);
	}
	m_size = make_double3(lx, ly, lz) + 2*extra_offset;
	m_origin = -m_size/2;

	m_simparams->buildneibsfreq = 10;

	m_simparams->tend = 1.0;

	m_physparams->epsinterface = 0.08;

	// Physical parameters
	m_physparams->gravity = make_float3(0.0, 0.0, -9.81f);
	float g = length(m_physparams->gravity);

	float maxvel = sqrt(g*H);
	float rho0 = 1;
	float rho1 = 1000;

	size_t air = add_fluid(rho0, 1.4, 198*maxvel);
	size_t water = add_fluid(rho1, 7.0f, 14*maxvel);

	//set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5
	m_physparams->dcoeff = 5.0f*g*H;

	m_physparams->r0 = m_deltap;

	set_kinematic_visc(air, 4.5e-3f);
	set_kinematic_visc(water, 3.5e-5f);

	m_physparams->artvisccoeff = 0.3f;
	m_physparams->epsartvisc = 0.01*m_simparams->slength*m_simparams->slength;

	// Drawing and saving times
	add_writer(VTKWRITER, 0.01);

	// Name of problem used for directory creation
	m_name = "Bubble";
}


Bubble::~Bubble(void)
{
	release_memory();
}


void Bubble::release_memory(void)
{
	fluid_parts.clear();
	boundary_parts.clear();
}


int Bubble::fill_parts()
{
	float r0 = m_physparams->r0;

	experiment_box = Cube(Point(m_origin), m_size.x,
		m_size.y, m_size.z);

	fluid = Cube(Point(m_origin + extra_offset),
		lx, ly, H);

	experiment_box.SetPartMass(r0, m_physparams->rho0[1]);

#if !USE_PLANES
	switch (m_simparams->boundarytype) {
	case LJ_BOUNDARY:
	case MK_BOUNDARY:
		experiment_box.FillBorder(boundary_parts, r0, false);
		break;
	case DYN_BOUNDARY:
		experiment_box.FillIn(boundary_parts, m_deltap, dyn_layers);
		break;
	default:
		throw std::runtime_error("unhandled boundary type in fill_parts");
	}
#endif

	// the actual particle mass will be set during the copy_array
	// routine
	fluid.SetPartMass(m_deltap, 1);
	fluid.Fill(fluid_parts, m_deltap, true);

	return fluid_parts.size() + boundary_parts.size();
}

uint Bubble::fill_planes()
{
#if USE_PLANES
	return 6;
#else
	return 0;
#endif
}

void Bubble::copy_planes(float4 *planes, float *planediv)
{
	uint pnum = 0;
	// z = m_origin.z
	planes[pnum] = make_float4(0, 0, 1.0, -m_origin.z);
	planediv[pnum] = 1.0;
	++pnum;
	// z = m_origin.z+lz
	planes[pnum] = make_float4(0, 0, -1.0, m_origin.z+lz);
	planediv[pnum] = 1.0;
	++pnum;
	// y = m_origin.y
	planes[pnum] = make_float4(0, 1.0, 0, -m_origin.y);
	planediv[pnum] = 1.0;
	++pnum;
	// y = m_origin.y+ly
	planes[pnum] = make_float4(0, -1.0, 0, m_origin.y+ly);
	planediv[pnum] = 1.0;
	++pnum;
	// x = m_origin.x
	planes[pnum] = make_float4(1.0, 0, 0, -m_origin.x);
	planediv[pnum] = 1.0;
	++pnum;
	// x = m_origin.x+lx
	planes[pnum] = make_float4(-1.0, 0, 0, m_origin.x+lx);
	planediv[pnum] = 1.0;
	++pnum;
}


// the bubble is initially located centered at 2R from the bottom.
bool is_inside(double3 const& origin, float R, const Point &pt)
{
	return
		(pt(0)*pt(0)) +
		(pt(1)*pt(1)) +
		(pt(2) - (origin.z+2*R))*(pt(2) - (origin.z+2*R)) < R*R;
}

void Bubble::copy_to_array(BufferList &buffers)
{
	float4 *pos = buffers.getData<BUFFER_POS>();
	hashKey *hash = buffers.getData<BUFFER_HASH>();
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();

	std::cout << "Boundary parts: " << boundary_parts.size() << "\n";
	for (uint i = 0; i < boundary_parts.size(); i++) {
		info[i]= make_particleinfo(PT_BOUNDARY, 1, i);
		double depth = H - boundary_parts[i](2) + m_origin.z;
		vel[i] = make_float4(0, 0, 0, density(depth, 1));
		calc_localpos_and_hash(boundary_parts[i], info[i], pos[i], hash[i]);
	}

	int j = boundary_parts.size();
	std::cout << "Boundary part mass:" << pos[j-1].w << "\n";

	std::cout << "Fluid parts: " << fluid_parts.size() << "\n";
	int count[2] = {0, 0};
	for (uint i = j; i < j + fluid_parts.size(); i++) {

		Point &pt(fluid_parts[i-j]);
		int fluid_idx = is_inside(m_origin, R, pt) ? 0 : 1;
		double depth = H - pt(2) + m_origin.z;

		// hydrostatic density: for the heavy fluid, this is simply computed
		// as the density that gives pressure rho g h, with h depth
		float rho = density(depth, fluid_idx);
		// for the bubble, the hydrostatic density must be computed in a slightly
		// more complex way:
		if (fluid_idx == 0) {
			// interface: depth of center of the bubble corrected by
			// R^2 - horizontal offset squared
			// note: no correction by m_origin.z because we are only
			// interested in deltas
			float z_intf = 2*R + sqrtf(R*R
					- (pt(0))*(pt(0))
					- (pt(1))*(pt(1))
					);
			// pressure at interface, from heavy fluid
			float g = length(m_physparams->gravity);
			float P = m_physparams->rho0[1]*(H - z_intf)*g;
			// plus hydrostatic pressure from _our_ fluid
			P += m_physparams->rho0[0]*(z_intf - pt(2) + m_origin.z)*g;
			rho = density_for_pressure(P, 0);
		}
		info[i]= make_particleinfo(PT_FLUID, fluid_idx, i);
		vel[i] = make_float4(0, 0, 0, rho);
		calc_localpos_and_hash(fluid_parts[i-j], info[i], pos[i], hash[i]);
		pos[i].w *= rho;

		++count[fluid_idx];
		if (count[fluid_idx] == 1)
			std::cout << "Fluid #" << fluid_idx << " part mass: " << pos[i].w << "\n";
	}
}

