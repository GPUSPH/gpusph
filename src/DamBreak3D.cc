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
#include <cmath>
#include <iostream>

#include "DamBreak3D.h"
#include "Cube.h"
#include "Point.h"
#include "Vector.h"

// set to coords (x,y,z) if more accuracy is needed in such point
// (waiting for relative coordinates)
#define OFFSET_X (-lx/2)
#define OFFSET_Y (-ly/2)
#define OFFSET_Z (-lz/2)


DamBreak3D::DamBreak3D(const Options &options) : Problem(options)
{
	// Size and origin of the simulation domain
	lx = 1.6;
	ly = 0.67;
	lz = 0.6;	
	H = 0.4;
	wet = false;
	m_usePlanes = true;
	
	m_size = make_float3(lx, ly, lz);
	//m_origin = make_float3(0.0, 0.0, 0.0);
	m_origin = make_float3(OFFSET_X, OFFSET_Y, OFFSET_Z);

	m_writerType = VTKWRITER;
	//m_writerType = UDPWRITER;

	// SPH parameters
	set_deltap(0.02f);
	m_simparams.slength = 1.3f*m_deltap;
	m_simparams.kernelradius = 2.0f;
	m_simparams.kerneltype = WENDLAND;
	m_simparams.dt = 0.0001f;
	m_simparams.xsph = false;
	m_simparams.dtadapt = true;
	m_simparams.dtadaptfactor = 0.3;
	m_simparams.buildneibsfreq = 10;
	m_simparams.shepardfreq = 0;
	m_simparams.mlsfreq = 0;
	m_simparams.visctype = ARTVISC;
	//m_simparams.visctype = DYNAMICVISC;
	m_simparams.boundarytype= LJ_BOUNDARY;
	m_simparams.tend = 1.5f;

	// Free surface detection
	m_simparams.surfaceparticle = false;
	m_simparams.savenormals = false;

	// We have no moving boundary
	m_simparams.mbcallback = false;

	// Physical parameters
	H = 0.4f;
	m_physparams.gravity = make_float3(0.0, 0.0, -9.81f);
	float g = length(m_physparams.gravity);
	m_physparams.set_density(0,1000.0, 7.0f, 20.f);
	
	//set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5
	m_physparams.dcoeff = 5.0f*g*H;
	m_physparams.r0 = m_deltap;
	
	// BC when using MK boundary condition: Coupled with m_simsparams.boundarytype=MK_BOUNDARY
	#define MK_par 2
	m_physparams.MK_K = g*H;
	m_physparams.MK_d = 1.1*m_deltap/MK_par;
	m_physparams.MK_beta = MK_par;
	#undef MK_par
	
	m_physparams.kinematicvisc = 1.0e-6f;
	m_physparams.artvisccoeff = 0.3f;
	m_physparams.epsartvisc = 0.01*m_simparams.slength*m_simparams.slength;
	
	// Scales for drawing
	m_maxrho = density(H,0);
	m_minrho = m_physparams.rho0[0];
	m_minvel = 0.0f;
	//m_maxvel = sqrt(m_physparams.gravity*H);
	m_maxvel = 3.0f;
	
	// Drawing and saving times
	m_displayinterval = 0.001f;
	m_writefreq = 20;
	m_screenshotfreq = 20;
	
	// Name of problem used for directory creation
	m_name = "DamBreak3D";
	create_problem_dir();
}


DamBreak3D::~DamBreak3D(void)
{
	release_memory();
}


void DamBreak3D::release_memory(void)
{
	parts.clear();
	obstacle_parts.clear();
	boundary_parts.clear();
}


int DamBreak3D::fill_parts()
{
	float r0 = m_physparams.r0;

	Cube fluid, fluid1;

	experiment_box = Cube(Point(0 + OFFSET_X, 0 + OFFSET_Y, 0 + OFFSET_Z), Vector(lx, 0, 0),
						Vector(0, ly, 0), Vector(0, 0, lz));

	obstacle = Cube(Point(0.9 + OFFSET_X, 0.24  + OFFSET_Y, r0 + OFFSET_Z), Vector(0.12, 0, 0),
					Vector(0, 0.12, 0), Vector(0, 0, lz - r0));

	fluid = Cube(Point(r0 + OFFSET_X, r0  + OFFSET_Y, r0 + OFFSET_Z), Vector(0.4, 0, 0),
				Vector(0, ly - 2*r0, 0), Vector(0, 0, H - r0));
	
	if (wet) {
		fluid1 = Cube(Point(H + m_deltap + r0 + OFFSET_X , r0 + OFFSET_Y, r0 + OFFSET_Z), Vector(lx - H - m_deltap - 2*r0, 0, 0),
					Vector(0, 0.67 - 2*r0, 0), Vector(0, 0, 0.1));
	}

	boundary_parts.reserve(2000);
	parts.reserve(14000);

	if (!m_usePlanes) {
		experiment_box.SetPartMass(r0, m_physparams.rho0[0]);
		experiment_box.FillBorder(boundary_parts, r0, false);
	}

	obstacle.SetPartMass(r0, m_physparams.rho0[0]);
	obstacle.FillBorder(obstacle_parts, r0, true);

	fluid.SetPartMass(m_deltap, m_physparams.rho0[0]);
	fluid.Fill(parts, m_deltap, true);
	if (wet) {
		fluid1.SetPartMass(m_deltap, m_physparams.rho0[0]);
		fluid1.Fill(parts, m_deltap, true);
		obstacle.Unfill(parts, r0);
	}

	return parts.size() + boundary_parts.size() + obstacle_parts.size();
}

uint DamBreak3D::fill_planes()
{
	return (m_usePlanes ? 5 : 0);
}

void DamBreak3D::copy_planes(float4 *planes, float *planediv)
{
	if (!m_usePlanes) return;

	// bottom
	planes[0] = make_float4(0, 0, 1.0, -OFFSET_Z);
	planediv[0] = 1.0;
	// back
	planes[1] = make_float4(1.0, 0, 0, -OFFSET_X);
	planediv[1] = 1.0;
	// front
	planes[2] = make_float4(-1.0, 0, 0, lx + OFFSET_X);
	planediv[2] = 1.0;
	// side with smaller Y ("left")
	planes[3] = make_float4(0, 1.0, 0, -OFFSET_Y);
	planediv[3] = 1.0;
	// side with greater Y ("right")
	planes[4] = make_float4(0, -1.0, 0, ly + OFFSET_Y);
	planediv[4] = 1.0;
}


void DamBreak3D::draw_boundary(float t)
{
	glColor3f(0.0, 1.0, 0.0);
	experiment_box.GLDraw();
	glColor3f(1.0, 0.0, 0.0);
	obstacle.GLDraw();
}

void DamBreak3D::fillDeviceMap(GlobalData* gdata)
{
	// TODO: test which split performs better, if Y (not many particles passing) or X (smaller section)
	fillDeviceMapByAxis(gdata, Y_AXIS);
	//fillDeviceMapByEquation(gdata);
}

void DamBreak3D::copy_to_array(float4 *pos, float4 *vel, particleinfo *info)
{
	std::cout << "Boundary parts: " << boundary_parts.size() << "\n";
	for (uint i = 0; i < boundary_parts.size(); i++) {
		pos[i] = make_float4(boundary_parts[i]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(BOUNDPART,0,i);
	}
	int j = boundary_parts.size();
	std::cout << "Boundary part mass:" << pos[j-1].w << "\n";

	std::cout << "Obstacle parts: " << obstacle_parts.size() << "\n";
	for (uint i = j; i < j + obstacle_parts.size(); i++) {
		pos[i] = make_float4(obstacle_parts[i-j]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(BOUNDPART,1,i);
	}
	j += obstacle_parts.size();
	std::cout << "Obstacle part mass:" << pos[j-1].w << "\n";

	std::cout << "Fluid parts: " << parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		pos[i] = make_float4(parts[i-j]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(FLUIDPART,0,i);
	}
	j += parts.size();
	std::cout << "Fluid part mass:" << pos[j-1].w << "\n";
}
