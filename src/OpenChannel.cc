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

#include <math.h>
#include <iostream>
#ifdef __APPLE__
#include <OpenGl/gl.h>
#else
#include <GL/gl.h>
#endif

#include "OpenChannel.h"


OpenChannel::OpenChannel(const Options &options) : Problem(options)
{
	// Size and origin of the simulation domain
	l = 2.0;
	a = 1.0;
	h = 0.7;
	H = 0.5;

	m_size = make_float3(l, a ,h);
	m_origin = make_float3(0.0f, 0.0f, 0.0f);

	m_writerType = VTKWRITER;

	// SPH parameters
	set_deltap(0.05f);
	m_simparams.slength = 1.3f*m_deltap;
	m_simparams.kernelradius = 2.0f;
	m_simparams.kerneltype = QUADRATIC;
	m_simparams.dt = 0.00004f;
	m_simparams.xsph = false;
	m_simparams.dtadapt = true;
	m_simparams.dtadaptfactor = 0.3;
	m_simparams.buildneibsfreq = 10;
	m_simparams.shepardfreq = 0;
	m_simparams.mlsfreq = 15;
	//m_simparams.visctype = ARTVISC;
	m_simparams.visctype = KINEMATICVISC;
	m_simparams.mbcallback = false;
	m_simparams.tend = 20;

	// Physical parameters
	m_physparams.gravity = make_float3(9.81f*sin(3.14159/20.0), 0.0, -9.81f*cos(3.14159/20.0));
	float g = length(m_physparams.gravity);

	m_physparams.set_density(0, 2650.0f, 2.0f, 20.f);
	m_physparams.dcoeff = 5.0f*g*H;

	m_physparams.r0 = m_deltap;
	m_physparams.kinematicvisc = 1.1e4f/m_physparams.rho0[0];

	m_physparams.epsartvisc = 0.01*m_simparams.slength*m_simparams.slength;
	//set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5
	m_simparams.periodicbound = true;
	m_physparams.dispvect = make_float3(l, 0.0, 0.0);
	m_physparams.minlimit = make_float3(0.0f, 0.0f, 0.0f);
	m_physparams.maxlimit = make_float3(l, 0.0f, 0.0f);

	// Free surface detection
	m_simparams.surfaceparticle = true;
	m_simparams.savenormals =true;

	// Scales for drawing
	m_maxrho = density(h,0);
	m_minrho = m_physparams.rho0[0];
	m_minvel = 0.0f;
	m_maxvel = 0.03f;

	// Drawing and saving times
	m_displayinterval = 0.001f;
	m_writefreq = 100;
	m_screenshotfreq = 10;

	// Name of problem used for directory creation
	m_name = "OpenChannel";
	create_problem_dir();
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

	experiment_box = Cube(Point(0, 0, 0), Vector(l, 0, 0), Vector(0, a, 0), Vector(0, 0, h + r0));
	Cube fluid = Cube(Point(m_deltap/2.0, r0, r0), Vector(l - m_deltap, 0, 0), Vector(0, a - 2*r0, 0), Vector(0, 0, H - r0));

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


void OpenChannel::draw_boundary(float t)
{
	glColor3f(0.0, 1.0, 0.0);
	rect1.GLDraw();
	rect2.GLDraw();
	rect3.GLDraw();
	glColor3f(1.0, 0.0, 0.0);
	experiment_box.GLDraw();
}


void OpenChannel::copy_to_array(float4 *pos, float4 *vel, particleinfo *info)
{
	std::cout << "Boundary parts: " << boundary_parts.size() << "\n";
	for (uint i = 0; i < boundary_parts.size(); i++) {
		pos[i] = make_float4(boundary_parts[i]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(BOUNDPART,0,i);
	}
	int j = boundary_parts.size();
	std::cout << "Boundary part mass:" << pos[j-1].w << "\n";

	std::cout << "Fluid parts: " << parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		pos[i] = make_float4(parts[i-j]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(FLUIDPART,0,i);
	}
	j += parts.size();
	std::cout << "Fluid part mass:" << pos[j-1].w << "\n";
}
