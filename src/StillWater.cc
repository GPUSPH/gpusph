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

#ifdef __APPLE__
#include <OpenGl/gl.h>
#else
#include <GL/gl.h>
#endif
#include <math.h>
#include <iostream>

#include "StillWater.h"


#define USE_PLANES 0

StillWater::StillWater(const Options &options) : Problem(options)
{
	H = 1;

	set_deltap(0.0625f);

	l = sqrt(2)*H; w = l; h = 1.1*H;

	// SPH parameters
	m_simparams.slength = 1.3f*m_deltap;
	m_simparams.kernelradius = 2.0f;
	m_simparams.kerneltype = WENDLAND;
	m_simparams.dt = 0.00004f;
	m_simparams.xsph = false;
	m_simparams.dtadapt = true;
	m_simparams.dtadaptfactor = 0.3;
	m_simparams.buildneibsfreq = 20;
	m_simparams.shepardfreq = 0;
	m_simparams.mlsfreq = 0;
	m_simparams.ferrari = 0.1;
	//m_simparams.visctype = KINEMATICVISC;
	m_simparams.visctype = DYNAMICVISC;
	//m_simparams.visctype = ARTVISC;
	m_simparams.mbcallback = false;
	m_simparams.boundarytype = MF_BOUNDARY;

	// Size and origin of the simulation domain
	m_size = make_float3(l, w ,h);
	m_origin = make_float3(0.0f, 0.0f, 0.0f);

	m_writerType = VTKWRITER;

	// Physical parameters
	m_physparams.gravity = make_float3(0.0, 0.0, -9.81f);
	float g = length(m_physparams.gravity);
	m_physparams.set_density(0, 1000.0, 7.0f, 32.0f);

	m_physparams.dcoeff = 5.0f*g*H;

	m_physparams.r0 = m_deltap;
	//m_physparams.visccoeff = 0.05f;
	m_physparams.kinematicvisc = 3.0e-2f;
	//m_physparams.kinematicvisc = 1.0e-6f;
	m_physparams.artvisccoeff = 0.3f;
	m_physparams.epsartvisc = 0.01*m_simparams.slength*m_simparams.slength;
	m_physparams.epsxsph = 0.5f;

	m_simparams.periodicbound = false;

	// Scales for drawing
	m_maxrho = density(H, 0);
	m_minrho = m_physparams.rho0[0];
	m_minvel = 0.0f;
	m_maxvel = 0.1f;

	// Drawing and saving times
	m_displayinterval = 1.0e-4;
	m_writefreq = 1000;
	m_screenshotfreq = 0;

	// Name of problem used for directory creation
	m_name = "StillWater";
	create_problem_dir();
}


StillWater::~StillWater(void)
{
	release_memory();
}


void StillWater::release_memory(void)
{
	parts.clear();
	boundary_parts.clear();
}


int StillWater::fill_parts()
{
	// distance between fluid box and wall
	float wd = m_physparams.r0;

	parts.reserve(14000);

	experiment_box = Cube(Point(0, 0, 0), Vector(l, 0, 0), Vector(0, w, 0), Vector(0, 0, h));

	experiment_box.SetPartMass(wd, m_physparams.rho0[0]);

#if !USE_PLANES
	if(m_simparams.boundarytype == MF_BOUNDARY) {
		experiment_box.FillBorder(boundary_parts, boundary_elems, vertex_parts, vertex_indexes, wd, false);
	}
	else {
		experiment_box.FillBorder(boundary_parts, wd, false);
	}
#endif

	Cube fluid = Cube(Point(wd, wd, wd), Vector(l-2*wd, 0, 0), Vector(0, w-2*wd, 0), Vector(0, 0, H-2*wd));
	fluid.SetPartMass(m_deltap, m_physparams.rho0[0]);
	// InnerFill puts particle in the center of boxes of step m_deltap, hence at
	// m_deltap/2 from the sides, so the total distance between particles and walls
	// is m_deltap = r0
	fluid.Fill(parts, m_deltap);
	
	//DEBUG: set only one fluid particle
//	parts.clear();
//	parts.push_back(Point(0.0, w/2.f, 0.0));
//	for(int i=0; i < vertex_parts.size(); i++)
//		if(	vertex_parts[i](2) == 0 &&
//			vertex_parts[i](0) > 0.5*w && vertex_parts[i](0) < 0.5*w+2*m_deltap &&
//			vertex_parts[i](1) > 0.5*w && vertex_parts[i](1) < 0.5*w+2*m_deltap)
//			parts.push_back(Point(vertex_parts[i](0) + 0.5*m_deltap, vertex_parts[i](1) + 0.5*m_deltap, 0.0));

	return parts.size() + boundary_parts.size() + vertex_parts.size();
}

uint StillWater::fill_planes()
{
#if USE_PLANES
	return 5;
#else
	return 0;
#endif

}

void StillWater::copy_planes(float4 *planes, float *planediv)
{
	planes[0] = make_float4(0, 0, 1.0, 0);
	planediv[0] = 1.0;
	planes[1] = make_float4(0, 1.0, 0, 0);
	planediv[1] = 1.0;
	planes[2] = make_float4(0, -1.0, 0, w);
	planediv[2] = 1.0;
	planes[3] = make_float4(1.0, 0, 0, 0);
	planediv[3] = 1.0;
	planes[4] = make_float4(-1.0, 0, 0, l);
	planediv[4] = 1.0;
}


void StillWater::draw_boundary(float t)
{
	glColor3f(1.0, 0.0, 0.0);
	experiment_box.GLDraw();
}


void StillWater::copy_to_array(float4 *pos, float4 *vel, particleinfo *info)
{
	std::cout << "Boundary parts: " << boundary_parts.size() << "\n";
	for (uint i = 0; i < boundary_parts.size(); i++) {
		pos[i] = make_float4(boundary_parts[i]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i] = make_particleinfo(BOUNDPART, 0, i);
	}
	int j = boundary_parts.size();
	std::cout << "Boundary part mass: " << pos[j-1].w << "\n";

	std::cout << "Fluid parts: " << parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		pos[i] = make_float4(parts[i-j]);
		float rho = density(H - pos[i].z, 0);
		vel[i] = make_float4(0, 0, 0, rho);
		info[i] = make_particleinfo(FLUIDPART, 0, i);
	}
	j += parts.size();
	std::cout << "Fluid part mass: " << pos[j-1].w << "\n";
}

void StillWater::copy_to_array(float4 *pos, float4 *vel, particleinfo *info, vertexinfo *vertices, float4 *boundelm)
{
	copy_to_array(pos, vel, info);

	uint j = parts.size() + boundary_parts.size();

	std::cout << "Vertex parts: " << vertex_parts.size() << "\n";
	for (uint i = j; i < j + vertex_parts.size(); i++) {
		pos[i] = make_float4(vertex_parts[i-j]);
		float rho = density(H - pos[i].z, 0);
		vel[i] = make_float4(0, 0, 0, rho);
		info[i] = make_particleinfo(VERTEXPART, 0, i);
	}
	j += vertex_parts.size();
	std::cout << "Vertex part mass: " << pos[j-1].w << "\n";

	if(vertex_indexes.size() != boundary_parts.size()) {
		std::cout << "ERROR! Incorrect connectivity array!\n";
		exit(1);
	}
	if(boundary_elems.size() != boundary_parts.size()) {
		std::cout << "ERROR! Incorrect boundary elements array!\n";
		exit(1);
	}

	uint offset = parts.size() + boundary_parts.size();
	for (uint i = 0; i < boundary_parts.size(); i++) {
		vertex_indexes[i].x += offset;
		vertex_indexes[i].y += offset;
		vertex_indexes[i].z += offset;

		vertices[i] = vertex_indexes[i];
		
		boundelm[i] = make_float4(boundary_elems[i]);
	}
}
