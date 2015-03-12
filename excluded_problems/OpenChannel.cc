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


OpenChannel::OpenChannel(const GlobalData *_gdata) : Problem(_gdata)
{
	SETUP_FRAMEWORK(
		//viscosity<ARTVISC>,
		viscosity<KINEMATICVISC>
	);

	// SPH parameters
	set_deltap(0.05f);
	m_simparams.dt = 0.00004f;
	m_simparams.dtadaptfactor = 0.3;
	m_simparams.buildneibsfreq = 10;
	m_simparams.mbcallback = false;
	m_simparams.tend = 20;

	// Size and origin of the simulation domain
	a = 1.0;
	h = 0.7;
	H = 0.5;

	m_simparams.periodicbound = PERIODIC_X;
	m_gridsize.x = 15;
	l = m_gridsize.x*m_simparams.kernelradius*m_simparams.slength;
	m_size = make_double3(l, a, h);
	m_origin = make_double3(0.0, 0.0, 0.0);

	// Physical parameters
	m_physparams.gravity = make_float3(9.81f*sin(3.14159/40.0), 0.0, -9.81f*cos(3.14159/40.0));
	float g = length(m_physparams.gravity);

	m_physparams.set_density(0, 2650.0f, 2.0f, 20.f);
	m_physparams.dcoeff = 5.0f*g*H;

	m_physparams.r0 = m_deltap;
	m_physparams.kinematicvisc = 110.f/m_physparams.rho0[0];

	m_physparams.epsartvisc = 0.01*m_simparams.slength*m_simparams.slength;
	//set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5

	// Free surface detection
	m_simparams.surfaceparticle = false;
	m_simparams.savenormals = false;

	// Drawing and saving times
	add_writer(VTKWRITER, 0.1);

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
	float r0 = m_physparams.r0;

	rect1 = Rect(Point(m_deltap/2.0, 0, 0), Vector(l - m_deltap, 0, 0), Vector(0, a, 0));
	rect2 = Rect(Point(m_deltap/2., 0, r0), Vector(l - m_deltap, 0, 0), Vector(0, 0, h - r0));
	rect3 = Rect(Point(m_deltap/2., a, r0), Vector(l - m_deltap, 0, 0), Vector(0, 0, h - r0));

	experiment_box = Cube(Point(0, 0, 0), l, a, h + r0);
	Cube fluid = Cube(Point(m_deltap/2.0, r0, r0), l - m_deltap, a - 2*r0, H - r0);

	boundary_parts.reserve(2000);
	parts.reserve(14000);

	rect1.SetPartMass(r0, m_physparams.rho0[0]);
	rect1.Fill(boundary_parts, r0, true);
	rect2.SetPartMass(r0, m_physparams.rho0[0]);
	rect2.Fill(boundary_parts, r0, true);
	rect3.SetPartMass(r0, m_physparams.rho0[0]);
	rect3.Fill(boundary_parts, r0, true);

	fluid.SetPartMass(m_deltap, m_physparams.rho0[0]);
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
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(BOUNDPART,0,i);
		calc_localpos_and_hash(boundary_parts[i], info[i], pos[i], hash[i]);
	}
	int j = boundary_parts.size();

	std::cout << "Fluid parts: " << parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(FLUIDPART,0,i);
		calc_localpos_and_hash(parts[i-j], info[i], pos[i], hash[i]);
	}
	j += parts.size();
	std::cout << "Fluid part mass:" << pos[j-1].w << "\n";
	std::flush(std::cout);
}
