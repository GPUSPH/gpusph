/*  Copyright 2014 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

/*
   Example use of dynamic boundaries. Implement a 2D Poiseuille flow
   with a double-periodic domain (X for flow direction, Y to eliminate side
   boundary effects). Top and bottom layers are implemented as dynamic boundary
   particles.
 */


#include <cmath>

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
		periodicity<PERIODIC_XY>
	);

	set_deltap(W/64);

	w = m_deltap*4;

	m_size = make_double3(W, W, H + 2*w);
	m_origin = -m_size/2;

	simparams()->tend = 2;

	/* slope */
	float degs = 60; /* degrees */
	alpha = M_PI*degs/180; /* radians */

	float g = 9.81f;
	physparams()->gravity = make_float3(g*sin(alpha), 0, -g*cos(alpha));

	float maxvel = sqrt(g*H);

	add_fluid(1);
	set_equation_of_state(0,  7, 10*maxvel);
	set_kinematic_visc(0, 120);

	physparams()->r0 = m_deltap/2;

	add_writer(VTKWRITER, 0.01);

	m_name = "DynBoundsExample";
}

DynBoundsExample::~DynBoundsExample(void)
{
	release_memory();
}

void
DynBoundsExample::release_memory(void)
{
	parts.clear();
	boundary_parts.clear();
}

int
DynBoundsExample::fill_parts(void)
{
	Cube fluid = Cube(m_origin + make_double3(0, 0, w), W, W, H);
	fluid.InnerFill(parts, m_deltap);

	Cube *bp = new Cube(m_origin, W, W, w);
	bp->InnerFill(boundary_parts, m_deltap);
	delete bp;
	bp = new Cube(m_origin + make_double3(0, 0, H + w), W, W, w);
	bp->InnerFill(boundary_parts, m_deltap);
	delete bp;

	return parts.size() + boundary_parts.size();
}

void
DynBoundsExample::copy_to_array(BufferList &buffers)
{
	float4 *pos = buffers.getData<BUFFER_POS>();
	hashKey *hash = buffers.getData<BUFFER_HASH>();
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();


	std::cout << "Boundary parts: " << boundary_parts.size() << std::endl;
	for (uint i = 0; i < boundary_parts.size(); ++i) {
		float ht = m_origin.z + H+2*w - boundary_parts[i](2);
		ht *= cos(alpha);
		float rho = density(ht, 0);
		vel[i] = make_float4(0, 0, 0, rho);
		info[i] = make_particleinfo(PT_BOUNDARY, 0, i);
		calc_localpos_and_hash(boundary_parts[i], info[i], pos[i], hash[i]);
		pos[i].w = m_deltap*m_deltap*m_deltap*rho;
	}
	uint j = boundary_parts.size();

	std::cout << "Fluid parts: " << parts.size() << std::endl;
	for (uint i = 0; i < parts.size(); ++i) {
		uint ij = i+j;
		float ht = m_origin.z + H+2*w - parts[i](2);
		ht *= cos(alpha);
		float rho = density(ht, 0);
		vel[ij] = make_float4(0, 0, 0, rho);
		info[ij] = make_particleinfo(PT_FLUID, 0, ij);
		calc_localpos_and_hash(parts[i], info[ij], pos[ij], hash[ij]);
		pos[ij].w = m_deltap*m_deltap*m_deltap*rho;
	}
	j += parts.size();

	std::cout << "Fluid part mass: " << pos[j-1].w << std::endl;

	float flowvel = H*H*fabs(physparams()->gravity.x)/(8*physparams()->kinematicvisc[0]);
	cout << "Expected maximum flow velocity: " << flowvel << endl;

	std::flush(std::cout);
}

