/*  Copyright (c) 2019 INGV, EDF, UniCT, JHU, NU

    Istituto Nazionale di Geofisica e Vulcanologia, Sezione di Catania, Italy
    Électricité de France, Paris, France
    Università di Catania, Catania, Italy
    Johns Hopkins University, Baltimore (MD), USA
    Northwestern University, Evanston (IL), USA

    This file is part of GPUSPH. Project founders:
        Alexis Hérault, Giuseppe Bilotta, Robert A. Dalrymple,
        Eugenio Rustico, Ciro Del Negro
    For a full list of authors and project partners, consult the logs
    and the project website <https://www.gpusph.org>

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

#include "Gate.h"
#include "Cube.h"
#include "Point.h"
#include "Vector.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

Gate::Gate(GlobalData *_gdata) : XProblem(_gdata)
{
	// *** user parameters from command line
	// density diffusion terms: 0 none, 1 Ferrari, 2 Molteni & Colagrossi, 3 Brezzi
	const DensityDiffusionType RHODIFF = get_option("density-diffusion", COLAGROSSI);

	// ** framework setup
	// viscosities: KINEMATICVISC*, DYNAMICVISC*
	// turbulence models: ARTVISC*, SPSVISC, KEPSVISC
	// boundary types: LJ_BOUNDARY*, MK_BOUNDARY, SA_BOUNDARY, DYN_BOUNDARY*
	// * = tested in this problem
	SETUP_FRAMEWORK(
		kernel<WENDLAND>,
		boundary<DYN_BOUNDARY>,
		periodicity<PERIODIC_Y>,
		add_flags<ENABLE_FEA>
	).select_options(
		RHODIFF
	);

	// Explicitly set number of layers. Also, prevent having undefined number of layers before the constructor ends.
	setDynamicBoundariesLayers(3);

	// *** Initialization of minimal physical parameters
	set_deltap(0.001);
	physparams()->r0 = m_deltap;
	physparams()->gravity = make_float3(0.0, 0.0, -9.81);
	const float g = length(physparams()->gravity);
	H = 0.14; //water height
	const double GateL = 0.08; //Gate height
	physparams()->dcoeff = 10.0f*g*H;
	water = add_fluid(1000.0);

	set_equation_of_state(water, 7.0f, 20.0*sqrt(2*9.81*H));
	set_kinematic_visc(0, 1.0e-6f);
	physparams()->artvisccoeff = 0.025;
	simparams()->densityDiffCoeff = 0.1f;

	simparams()->tend = 0.4f;

	// Drawing and saving times
	add_writer(VTKWRITER, 0.005f);
	// *** Other parameters and settings
	m_name = "Gate";

	// *** Geometrical parameters, starting from the size of the domain
	const double dimX = 0.3; // inner dimension of the chamber
	const double dimY = 0.012;
	const double dimZ = 0.145;
	const double water_length = 0.1;
	const double water_height = H;

	double BOUNDARY_DISTANCE = m_deltap;
	double BOUNDARY_THICKNESS = m_deltap;
	if (simparams()->boundarytype == DYN_BOUNDARY || simparams()->boundarytype == DUMMY_BOUNDARY) {
		BOUNDARY_DISTANCE *= getDynamicBoundariesLayers(); // FIXME gives issues if number of layers not settled manually
		BOUNDARY_THICKNESS *= (getDynamicBoundariesLayers() - 1);
	}

	m_origin = make_double3(0, m_deltap/2.0, 0);
	m_size = make_double3(dimX, dimY - m_deltap, dimZ);

	// set positioning policy to PP_CORNER: given point will be the corner of the geometry
	setPositioning(PP_CORNER);
	// main container

	GeometryID box = addBox(GT_FIXED_BOUNDARY, FT_BORDER, Point(-BOUNDARY_THICKNESS, -BOUNDARY_THICKNESS, -BOUNDARY_THICKNESS),
		dimX + 2*BOUNDARY_THICKNESS, dimY + 2*BOUNDARY_THICKNESS, dimZ + 2*BOUNDARY_THICKNESS, 2, 5); // the last two integers are the number of fea shell in the two directions
	setEraseOperation(box, ET_ERASE_NOTHING);

	// Add the main water part
	addBox(GT_FLUID, FT_SOLID, Point(m_deltap, m_deltap, m_deltap),
		water_length - 2*m_deltap, dimY - 2*m_deltap, water_height - m_deltap); // check BC on the free surface

	// add wall above the gate
	GeometryID wall = addRect(GT_FIXED_BOUNDARY, FT_BORDER, Point(water_length, m_deltap, 2*m_deltap + GateL), dimZ - GateL - 3*m_deltap, dimY - 2*m_deltap);

	rotate(wall, 0, M_PI/2, 0);
	setEraseOperation(wall, ET_ERASE_NOTHING);


	// erase side walls in case of periodicity
	GeometryID wall1 = addBox(GT_FIXED_BOUNDARY, FT_UNFILL, Point(-BOUNDARY_THICKNESS, dimY + m_deltap/2.0, -BOUNDARY_THICKNESS), dimX + 2*BOUNDARY_THICKNESS, BOUNDARY_THICKNESS, dimZ + 2*BOUNDARY_THICKNESS);
	setEraseOperation(wall1, ET_ERASE_BOUNDARY);
	GeometryID wall2 = addBox(GT_FIXED_BOUNDARY, FT_UNFILL, Point(-BOUNDARY_THICKNESS, -BOUNDARY_THICKNESS, -BOUNDARY_THICKNESS), dimX + 2*BOUNDARY_THICKNESS, BOUNDARY_THICKNESS, dimZ + 2*BOUNDARY_THICKNESS);
	setEraseOperation(wall2, ET_ERASE_BOUNDARY);

	GeometryID erase_ceil = addBox(GT_FIXED_BOUNDARY, FT_UNFILL, Point(-BOUNDARY_THICKNESS, -BOUNDARY_THICKNESS, dimZ), dimX + 2*BOUNDARY_THICKNESS, dimY + 2*BOUNDARY_THICKNESS, dimZ);
	setEraseOperation(wall2, ET_ERASE_BOUNDARY);


	// Add the flexible gate as a mesh
	GeometryID gate = addBox(GT_DEFORMABLE_BODY, FT_BORDER, Point(water_length + 0.005, m_deltap, m_deltap), GateL, dimY - 2*m_deltap, round_up(0.005, m_deltap), 10, 1);
	setEraseOperation(gate, ET_ERASE_NOTHING);
	setYoungModulus(gate, 1e7);
	setPoissonRatio(gate, 0.3);
	setAlphaDamping(gate, 0.001);
	setDensity(gate, 1100);
	rotate(gate, 0, M_PI/2, 0);
	set_fea_ground(0, 0, -1, -(GateL + 0.5*m_deltap)); // a, b, c and d parameters of a plane equation. Grounding nodes in the negative side of the plane
}

// since the fluid topology is roughly symmetric along Y through the whole simulation, prefer Y split
/*void Gate::fillDeviceMap()
{
	fillDeviceMapByAxis(Y_AXIS);
}
*/

void Gate::initializeParticles(BufferList &buffer, const uint numParticle)
{
	float4 *pos = buffer.getData<BUFFER_POS>();
	float4 *vel = buffer.getData<BUFFER_VEL>();
	ushort4 *info= buffer.getData<BUFFER_INFO>();
	double4 *pos_global = buffer.getData<BUFFER_POS_GLOBAL>();

	for (uint i = 0; i < numParticle; i++) {

		double depth = 3*m_deltap +  H - pos_global[i].z + m_origin.z;

		pos[i].w = physparams()->rho0[0]*m_deltap*m_deltap*m_deltap;
		vel[i].w = hydrostatic_density(depth, water);
	}
}

bool Gate::need_write(double t) const
{
	return false;
}



