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
		viscosity<DYNAMICVISC>,
		boundary<DYN_BOUNDARY>,
		periodicity<PERIODIC_Y>,
		add_flags<ENABLE_FEA>
	).select_options(
		RHODIFF
	);


	// Allow user to set the MLS frequency at runtime. Default to 0 if density
	// diffusion is enabled, 10 otherwise
	const int mlsIters = get_option("mls",
		(simparams()->densitydiffusiontype != DENSITY_DIFFUSION_NONE) ? 0 : 10);

	if (mlsIters > 0)
		addFilter(MLS_FILTER, mlsIters);

	// Explicitly set number of layers. Also, prevent having undefined number of layers before the constructor ends.
	setDynamicBoundariesLayers(3);

	//resize_neiblist(128);

	// *** Initialization of minimal physical parameters
	set_deltap(1/512.0f);
	physparams()->r0 = m_deltap;
	physparams()->gravity = make_float3(0.0, 0.0, -9.81);
	const float g = length(physparams()->gravity);
	H = 0.14; //water height
	const double GateL = 0.08; //Gate height
	physparams()->dcoeff = 10.0f * g * H;
	water = add_fluid(1000.0);

	set_equation_of_state(water,  7.0f, 100.0f);
	set_kinematic_visc(0, 1.0e-6f);

	simparams()->tend=0.4f;
	simparams()->densityDiffCoeff = 0.1f;

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

	// Planes unfill automatically but the box won't, to void deleting all the water. Thus,
	// we define the water at already the right distance from the walls.
	double BOUNDARY_DISTANCE = m_deltap;
	double BOUNDARY_THICKNESS = m_deltap;
	if (simparams()->boundarytype == DYN_BOUNDARY) {
			BOUNDARY_DISTANCE *= getDynamicBoundariesLayers(); // FIXME gives issues if number of layers not settled manually
			BOUNDARY_THICKNESS *= (getDynamicBoundariesLayers() - 1);

		/*BOUNDARY_DISTANCE *= 4;
		BOUNDARY_THICKNESS *= 3;*/
	}

	// If we used only makeUniverseBox(), origin and size would be computed automatically

	/* // no periodicity
	m_origin = make_double3(-BOUNDARY_DISTANCE, -BOUNDARY_DISTANCE, -BOUNDARY_DISTANCE);
	m_size = make_double3(dimX + 2*BOUNDARY_DISTANCE, dimY + 2*BOUNDARY_DISTANCE, dimZ + 2*BOUNDARY_DISTANCE);
	*/
	m_origin = make_double3(0, m_deltap/2.0, 0);
	m_size = make_double3(dimX, dimY - m_deltap, dimZ);

	// set positioning policy to PP_CORNER: given point will be the corner of the geometry
	setPositioning(PP_CORNER);
	// main container

	GeometryID box = addBox(GT_FIXED_BOUNDARY, FT_BORDER, Point(-BOUNDARY_THICKNESS, -BOUNDARY_THICKNESS, -BOUNDARY_THICKNESS),
		dimX + 2*BOUNDARY_THICKNESS, dimY + 2*BOUNDARY_THICKNESS, dimZ + 2*BOUNDARY_THICKNESS, 2, 5); // the last three integers are the number of fea elements in the three directions
	// we simulate inside the box, so do not erase anything
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
//	addTetFile(GT_DEFORMABLE_BODY, FT_BORDER, Point(water_length, m_deltap, m_deltap), "gate_001.node", "gate_001.ele", GateL + m_deltap*0.5) ;
	GeometryID gate = addBox(GT_DEFORMABLE_BODY, FT_BORDER, Point(water_length, m_deltap, m_deltap), 0.005, dimY - 2*m_deltap, GateL, 1, 10);
	setEraseOperation(gate, ET_ERASE_NOTHING);
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

	// TODO FIXME the particle mass should be assigned from the mesh. We should 
	// understand why GetMass on the fea mesh gives 0

	for (uint i = 0; i < numParticle; i++) {

		double depth = 3*m_deltap +  H - pos_global[i].z + m_origin.z;

		pos[i].w = 1000*m_deltap*m_deltap*m_deltap;
		vel[i].w = hydrostatic_density(depth, water);
	}
}

bool Gate::need_write(double t) const
{
	return false;
}



