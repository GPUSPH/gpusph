/*  Copyright 2011-2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

#include "OpenChannel.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

OpenChannel::OpenChannel(GlobalData *_gdata) : Problem(_gdata)
{
	use_side_walls = get_option("sidewalls", true);

	SETUP_FRAMEWORK(
		//viscosity<ARTVISC>,
		viscosity<KINEMATICVISC>,
		boundary<DYN_BOUNDARY>,
		periodicity<PERIODIC_X>
	).select_options(
		use_side_walls, periodicity<PERIODIC_XY>()
	);

	// SPH parameters
	set_deltap(0.02f);
	simparams()->dt = 0.00004f;
	simparams()->dtadaptfactor = 0.3;
	simparams()->buildneibsfreq = 10;
	simparams()->tend = 20;

	H = 0.5; // water level

	if (simparams()->boundarytype == DYN_BOUNDARY) {
		dyn_layers = ceil(simparams()->influenceRadius/m_deltap) + 1;
		// no extra offset in the X direction, since we have periodicity there
		// no extra offset in the Y direction either if we do NOT have side walls
		dyn_offset = dyn_layers*make_double3(0,
			use_side_walls ? m_deltap : 0,
			m_deltap);
	} else {
		dyn_layers = 0;
		dyn_offset = make_double3(0.0);
	}

	// Size and origin of the simulation domain
	a = 1.0;
	h = H*1.4;
	l = 15*simparams()->influenceRadius;

	m_size = make_double3(l, a, h) + 2*dyn_offset;
	m_origin = make_double3(0.0, 0.0, 0.0) - dyn_offset;

	// Physical parameters
	const double angle = 4.5; // angle in degrees
	const float g = 9.81f;
	physparams()->gravity = make_float3(g*sin(M_PI*angle/180), 0.0, -g*cos(M_PI*angle/180));

	add_fluid(2650.0f);
	set_equation_of_state(0,  2.0f, 20.f);
	set_dynamic_visc(0, 110.f);

	physparams()->dcoeff = 5.0f*g*H;

	physparams()->r0 = m_deltap;

	physparams()->epsartvisc = 0.01*simparams()->slength*simparams()->slength;
	//set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5

	// Drawing and saving times
	add_writer(VTKWRITER, 0.5);

	// Name of problem used for directory creation
	m_name = "OpenChannel";
}


OpenChannel::~OpenChannel(void)
{
	release_memory();
}


void OpenChannel::release_memory(void)
{
	parts.clear();
	boundary_parts.clear();
}


int OpenChannel::fill_parts()
{
	const float r0 = physparams()->r0;
	// gap due to periodicity
	const double3 periodicity_gap = make_double3(m_deltap/2,
		use_side_walls ? 0 : m_deltap/2, 0);

	experiment_box = Cube(m_origin, l, a, h);

	// bottom: it must cover the whole bottom floor, including under the walls,
	// hence it must not be shifted by dyn_offset in the y direction. In the
	// Y-periodic case (no side walls), the Y length must be decreased by
	// a deltap to account for periodicity (start at deltap/2, end deltap/2 before the end)
	rect1 = Rect(m_origin + make_double3(dyn_offset.x, 0, dyn_offset.z) + periodicity_gap,
		Vector(0, m_size.y - (use_side_walls ? 0 : m_deltap), 0),
		Vector(m_size.x - m_deltap, 0, 0));

	if (use_side_walls) {
		// side walls: shifted by dyn_offset, and with opposite orientation so that
		// they "fill in" towards the outside
		rect2 = Rect(m_origin + dyn_offset + periodicity_gap + make_double3(0, 0, r0),
			Vector(l - m_deltap, 0, 0), Vector(0, 0, h - r0));
		rect3 = Rect(m_origin + dyn_offset + periodicity_gap + make_double3(0, a, r0),
			Vector(0, 0, h - r0), Vector(l - m_deltap, 0, 0));
	}

	Cube fluid = use_side_walls ?
		Cube(m_origin + dyn_offset + periodicity_gap + make_double3(0, r0, r0),
		l - m_deltap, a - 2*r0, H - r0) :
		Cube(m_origin + dyn_offset + periodicity_gap + make_double3(0, 0, r0),
		l - m_deltap, a - m_deltap, H - r0) ;

	boundary_parts.reserve(2000);
	parts.reserve(14000);

	rect1.SetPartMass(r0, physparams()->rho0[0]);
	if (use_side_walls) {
		rect2.SetPartMass(r0, physparams()->rho0[0]);
		rect3.SetPartMass(r0, physparams()->rho0[0]);
	}

	if (simparams()->boundarytype == DYN_BOUNDARY) {
		rect1.FillIn(boundary_parts, m_deltap, dyn_layers);
		if (use_side_walls) {
			rect2.FillIn(boundary_parts, m_deltap, dyn_layers);
			rect3.FillIn(boundary_parts, m_deltap, dyn_layers);
		}
	} else {
		rect1.Fill(boundary_parts, r0, true);
		if (use_side_walls) {
			rect2.Fill(boundary_parts, r0, true);
			rect3.Fill(boundary_parts, r0, true);
		}
	}

	fluid.SetPartMass(m_deltap, physparams()->rho0[0]);
	fluid.Fill(parts, m_deltap, true);

	return parts.size() + boundary_parts.size();
}

void OpenChannel::copy_to_array(BufferList &buffers)
{
	float4 *pos = buffers.getData<BUFFER_POS>();
	hashKey *hash = buffers.getData<BUFFER_HASH>();
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();

	std::cout << "Boundary parts: " << boundary_parts.size() << "\n";
	for (uint i = 0; i < boundary_parts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, physparams()->rho0[0]);
		info[i]= make_particleinfo(PT_BOUNDARY,0,i);
		calc_localpos_and_hash(boundary_parts[i], info[i], pos[i], hash[i]);
	}
	int j = boundary_parts.size();

	std::cout << "Fluid parts: " << parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, physparams()->rho0[0]);
		info[i]= make_particleinfo(PT_FLUID,0,i);
		calc_localpos_and_hash(parts[i-j], info[i], pos[i], hash[i]);
	}
	j += parts.size();
	std::cout << "Fluid part mass:" << pos[j-1].w << "\n";
	std::flush(std::cout);
}
