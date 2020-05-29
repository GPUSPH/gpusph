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
#include <stdexcept>
#include <math.h>

#include "TestFeaPile.h"
#include "particledefine.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

/*0 load from TetFile, 1: define single Cylinder, 2: Four legged Platform square pillers, 3:Four legged platform Cylindrical pillers*/

#define USE_WATER 0


#define MK_par 2

TestFeaPile::TestFeaPile(GlobalData *_gdata) : XProblem(_gdata)
{
	// Size and origin of the simulation domain
	lx = 1.0;
	ly = 1.0;
	lz = 4.0;

	SETUP_FRAMEWORK(
		add_flags<ENABLE_FEA>
	);

	m_size = make_double3(lx, ly, lz);
	m_origin = make_double3(-lx/2.0, -ly/2.0, -0.5);


	// SPH parameters
	set_deltap(1/64.0);
	simparams()->dtadaptfactor = 0.2;
	simparams()->buildneibsfreq = 10;

	simparams()->t_fea_start= 0.0f; //seconds

	simparams()->fcallback = true;

	// Physical parameters
	H = 0.0;
	physparams()->gravity = make_float3(0.0f, 0.0f, -9.81f);
	float g = length(physparams()->gravity);

	float r0 = m_deltap;
	physparams()->r0 = r0;

	add_fluid( 1000.0f);
	set_equation_of_state(0,  7.0f, 80.f);
	set_kinematic_visc(0,1.0e-6);

	physparams()->artvisccoeff =  0.2;
	physparams()->smagfactor = 0.12*0.12*m_deltap*m_deltap;
	physparams()->kspsfactor = (2.0/3.0)*0.0066*m_deltap*m_deltap;
	physparams()->epsartvisc = 0.01*simparams()->slength*simparams()->slength;

	// BC when using LJ
	physparams()->dcoeff = 0.5*g*H;
	//set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5

	add_writer(VTKWRITER, 0.005);  //second argument is saving time in seconds

	// Name of problem used for directory creation
	m_name = "TestFeaPile";

	// Building the geometry
	setPositioning(PP_CORNER);
/*
	if (H > 0) {
		GeometryID fluid = addBox(GT_FLUID, FT_SOLID,
			Point(-lx/2.0, -ly/2.0, 0), lx, ly, H);
	}
*/
	setPositioning(PP_CORNER);

	set_fea_ground(0, 0, 1, -0.5); // a, b, c and d parameters of a plane equation. Grounding nodes in the negative side of the plane

	double inner_radius = 0.065;
	double outer_radius = 0.1;
	double height = 3;
	int num_els = 1;

	GeometryID pile = addCylinder(GT_DEFORMABLE_BODY, FT_BORDER, Point(-outer_radius, -outer_radius, 0), outer_radius, inner_radius, height, num_els);
	setEraseOperation(pile, ET_ERASE_FLUID);

	setYoungModulus(pile, 1e8);
	setPoissonRatio(pile, 0.3);
	setDensity(pile, 1522.0);

	GeometryID force_box = addBox(GT_FEA_FORCE, FT_BORDER, Point(-0.05, -0.05, height - 0.05), 0.1, 0.1, 0.1);
	GeometryID write_box = addBox(GT_FEA_WRITE, FT_BORDER, Point(-0.05, -0.05, height - 0.05), 0.1, 0.1, 0.1);
	setPositioning(PP_CENTER);
	GeometryID joint_box = addBox(GT_FEA_RIGID_JOINT, FT_BORDER, Point(-0.0, -0.0, - 0.05), 0.1, 0.1, 0.2);
	simparams()->fea_write_every = 0.01f;
}

float3 TestFeaPile::ext_force_callback(const double t)
{
	/* NOTE: set simparams::fcallback to true */
	float forcex;

	if (t < 6.0)
		forcex = 50/(1 + exp(2*(2.5 - t))); // logistic function with L = 100, k = 3; x0 = 2 
	else
		forcex = 0;

	return make_float3(forcex, 0.0, 0.0);
}

void TestFeaPile::initializeParticles(BufferList &buffer, const uint numParticle)
{
	float4 *pos = buffer.getData<BUFFER_POS>();
	const float4 *vel = buffer.getData<BUFFER_VEL>();
	const ushort4 *info= buffer.getData<BUFFER_INFO>();

	// TODO FIXME the particle mass should be assigned from the mesh. We should 
	// understand why GetMass on the fea mesh gives 0

	for (uint i = 0; i < numParticle; i++) {
		if (DEFORMABLE(info[i]))
			pos[i].w = physical_density(vel[i].w, 0)*m_deltap*m_deltap*m_deltap;
	}
}

bool TestFeaPile::need_write(double t) const
{
	return false;
}
#undef MK_par
