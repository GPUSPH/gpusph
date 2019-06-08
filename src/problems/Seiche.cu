/*  Copyright (c) 2011-2019 INGV, EDF, UniCT, JHU

    Istituto Nazionale di Geofisica e Vulcanologia, Sezione di Catania, Italy
    Électricité de France, Paris, France
    Università di Catania, Catania, Italy
    Johns Hopkins University, Baltimore (MD), USA

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

#include "cudasimframework.cu"

#include "Seiche.h"
#include "particledefine.h"
#include "GlobalData.h"

Seiche::Seiche(GlobalData *_gdata) : Problem(_gdata)
{
	SETUP_FRAMEWORK(
		viscosity<SPSVISC>,
		add_flags<ENABLE_PLANES>
	);

	addFilter(MLS_FILTER, 20);

	set_deltap(0.015f);
	H = .5f;
	l = sqrt(2)*H; w = l/2; h = 1.5*H;
	cout << "length= " << l<<"\n";
	cout << "width= " << w <<"\n";
	cout << "h = " << h <<"\n";

	// Size and origin of the simulation domain
	m_size = make_double3(l, w ,h);
	m_origin = make_double3(0.0, 0.0, 0.0);

	// SPH parameters
	set_timestep(0.00004f);
	simparams()->dtadaptfactor = 0.2;
	simparams()->buildneibsfreq = 10;
	simparams()->tend=10.0f;
	simparams()->gcallback=true;

	// Physical parameters
	set_gravity(-9.81f); //must be set first
	add_fluid(1000.0);
	set_equation_of_state(0,  7.0f, 20.f);
	setMaxFall(H);

	set_kinematic_visc(0, 5.0e-6f);

	// Variable gravity terms:  starting with physparams()->gravity as defined above
	m_gtstart=0.3;
	m_gtend=3.0;

	// Drawing and saving times
	add_writer(VTKWRITER, 0.1);

	// Name of problem used for directory creation
	m_name = "Seiche";

	// Building the geometry
	setPositioning(PP_CORNER);
	// distance between fluid box and wall
	float wd = m_deltap; //Used to be divided by 2

	GeometryID experiment_box = addBox(GT_FIXED_BOUNDARY, FT_BORDER, Point(0, 0, 0), l, w, h);
	disableCollisions(experiment_box);
	GeometryID fluid = addBox(GT_FLUID, FT_SOLID, Point(wd, wd, wd), l-2*wd, w-2*wd, H-2*wd);
}

float3 Seiche::g_callback(const double t)
{
	if(t > m_gtstart && t < m_gtend)
		set_gravity(2.*sin(9.8*(t-m_gtstart)), 0.0, -9.81f);
	else
		set_gravity(0.,0.,-9.81f);
	return get_gravity();
}

void Seiche::copy_planes(PlaneList& planes)
{
	planes.push_back( implicit_plane(0, 0, 1, 0) );
	planes.push_back( implicit_plane(0, 1, 0, 0) );
	planes.push_back( implicit_plane(0, -1, 0, w) );
	planes.push_back( implicit_plane(1, 0, 0, 0) );
	planes.push_back( implicit_plane(-1, 0, 0, l) );
}
