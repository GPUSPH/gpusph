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

#include <math.h>
#include <iostream>

#include "StillWater.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

#define CENTER_DOMAIN 1
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

StillWater::StillWater(GlobalData *_gdata) : Problem(_gdata)
{
	m_usePlanes = get_option("use-planes", false); // --use-planes true to enable use of planes for boundaries
	const int mlsIters = get_option("mls", 0); // --mls N to enable MLS filter every N iterations
	const int ppH = get_option("ppH", 16); // --ppH N to change deltap to H/N

	// density diffusion terms: 0 none, 1 Molteni & Colagrossi, 2 Ferrari
	const int rhodiff = get_option("density-diffusion", 1);

	SETUP_FRAMEWORK(
		//viscosity<KINEMATICVISC>,
		viscosity<DYNAMICVISC>,
		//viscosity<ARTVISC>,
		boundary<DYN_BOUNDARY>
		//boundary<SA_BOUNDARY>
		//boundary<LJ_BOUNDARY>
	).select_options(
		rhodiff, FlagSwitch<ENABLE_NONE, ENABLE_DENSITY_DIFFUSION, ENABLE_FERRARI>(),
		m_usePlanes, add_flags<ENABLE_PLANES>()
	);

	if (mlsIters > 0)
		addFilter(MLS_FILTER, mlsIters);

	H = 1;

	set_deltap(H/ppH);

	l = w = sqrt(2)*H; h = 1.1*H;

	// Size and origin of the simulation domain
	m_size = make_double3(l, w ,h);
	m_origin = make_double3(OFFSET_X, OFFSET_Y, OFFSET_Z);

	// SPH parameters
	simparams()->dt = 0.00004f;
	simparams()->dtadaptfactor = 0.3;
	simparams()->buildneibsfreq = 20;
	simparams()->ferrariLengthScale = H;

	// enlarge the domain to take into account the extra layers of particles
	// of the boundary
	if (simparams()->boundarytype == DYN_BOUNDARY && !m_usePlanes) {
		// number of layers
		dyn_layers = ceil(simparams()->kernelradius*simparams()->sfactor);
		// extra layers are one less (since other boundary types still have
		// one layer)
		double3 extra_offset = make_double3((dyn_layers-1)*m_deltap);
		m_origin -= extra_offset;
		m_size += 2*extra_offset;
	} else {
		dyn_layers = 1;
	}

	simparams()->tend = 100.0;
	if (simparams()->boundarytype == SA_BOUNDARY) {
		simparams()->maxneibsnum = 256; // needed during gamma initialization phase
	};

	// Physical parameters
	physparams()->gravity = make_float3(0.0, 0.0, -9.81f);
	const float g = length(physparams()->gravity);
	const float maxvel = sqrt(2*g*H);
	// purely for cosmetic reason, let's round the soundspeed to the next
	// integer
	const float c0 = ceil(10*maxvel);
	add_fluid(1000.0);
	set_equation_of_state(0,  7.0f, c0);

	physparams()->dcoeff = 5.0f*g*H;

	physparams()->r0 = m_deltap;
	//physparams()->visccoeff = 0.05f;
	set_kinematic_visc(0, 3.0e-2f);
	//set_kinematic_visc(0, 1.0e-6f);
	physparams()->artvisccoeff = 0.3f;
	physparams()->epsartvisc = 0.01*simparams()->slength*simparams()->slength;
	physparams()->epsxsph = 0.5f;

	// Drawing and saving times
	add_writer(VTKWRITER, 1.0);

	// Name of problem used for directory creation
	m_name = "StillWater";
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
	float wd = physparams()->r0;

	parts.reserve(14000);

	experiment_box = Cube(Point(m_origin), m_size.x, m_size.y, m_size.z);

	experiment_box.SetPartMass(wd, physparams()->rho0[0]);

	if (!m_usePlanes) {
		switch (simparams()->boundarytype) {
		case SA_BOUNDARY:
			experiment_box.FillBorder(boundary_parts, boundary_elems, vertex_parts, vertex_indexes, wd, false);
			break;
		case DYN_BOUNDARY:
			experiment_box.FillIn(boundary_parts, m_deltap, dyn_layers, false);
			break;
		default:
			experiment_box.FillBorder(boundary_parts, wd, false);
			break;
		}
	}

	m_fluidOrigin = m_origin;
	if (dyn_layers > 1) // shift by the extra offset of the experiment box
		m_fluidOrigin += make_double3((dyn_layers-1)*m_deltap);
	m_fluidOrigin += make_double3(wd); // one wd space from the boundary
	Cube fluid = Cube(m_fluidOrigin, l-2*wd, w-2*wd, H-2*wd);
	fluid.SetPartMass(m_deltap, physparams()->rho0[0]);
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

void StillWater::copy_planes(PlaneList& planes)
{
	if (!m_usePlanes) return;

	planes.push_back( implicit_plane(0, 0, 1.0, -m_origin.z) );
	planes.push_back( implicit_plane(0, 1.0, 0, -m_origin.x) );
	planes.push_back( implicit_plane(0, -1.0, 0, m_origin.x + w) );
	planes.push_back( implicit_plane(1.0, 0, 0, -m_origin.y) );
	planes.push_back( implicit_plane(-1.0, 0, 0, m_origin.y + l) );
}


void StillWater::copy_to_array(BufferList &buffers)
{
	float4 *pos = buffers.getData<BUFFER_POS>();
	hashKey *hash = buffers.getData<BUFFER_HASH>();
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();
	vertexinfo *vertices = buffers.getData<BUFFER_VERTICES>();
	float4 *boundelm = buffers.getData<BUFFER_BOUNDELEMENTS>();

	std::cout << "Boundary parts: " << boundary_parts.size() << "\n";
	for (uint i = 0; i < boundary_parts.size(); i++) {
#if 1
		double water_column = m_fluidOrigin.z + H - boundary_parts[i](2);
		if (water_column < 0)
			water_column = 0;
		float rho = density(water_column, 0);
#else
		float rho = physparams()->rho0[0];
#endif
		vel[i] = make_float4(0, 0, 0, rho);
		info[i] = make_particleinfo(PT_BOUNDARY, 0, i);
		calc_localpos_and_hash(boundary_parts[i], info[i], pos[i], hash[i]);
	}
	int j = boundary_parts.size();
	std::cout << "Boundary part mass: " << pos[j-1].w << "\n";

	std::cout << "Fluid parts: " << parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		double water_column = m_fluidOrigin.z + H - parts[i - j](2);
		if (water_column < 0)
			water_column = 0;
		float rho = density(water_column, 0);
		vel[i] = make_float4(0, 0, 0, rho);
		info[i] = make_particleinfo(PT_FLUID, 0, i);
		calc_localpos_and_hash(parts[i-j], info[i], pos[i], hash[i]);
	}
	j += parts.size();
	std::cout << "Fluid part mass: " << pos[j-1].w << "\n";

	if (simparams()->boundarytype == SA_BOUNDARY) {
			uint j = parts.size() + boundary_parts.size();

			std::cout << "Vertex parts: " << vertex_parts.size() << "\n";
		for (uint i = j; i < j + vertex_parts.size(); i++) {
			float rho = density(H - vertex_parts[i-j](2), 0);
			vel[i] = make_float4(0, 0, 0, rho);
			info[i] = make_particleinfo(PT_VERTEX, 0, i);
			calc_localpos_and_hash(vertex_parts[i-j], info[i], pos[i], hash[i]);
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
}
