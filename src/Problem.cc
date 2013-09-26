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

#include <sstream>
#include <stdexcept>
#include <string>
#include <cmath>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifdef __APPLE__
#include <OpenGl/gl.h>
#else
#include <GL/gl.h>
#endif

#include "Problem.h"
#include "vector_math.h"


Problem::Problem(const Options &options)
{
	m_options = options;
	m_last_display_time = 0.0;
	m_last_write_time = -1.0;
	m_last_screenshot_time = 0.0;
	m_mbnumber = 0;
	m_rbdatafile = NULL;
	m_rbdata_writeinterval = 0;
	memset(m_mbcallbackdata, 0, MAXMOVINGBOUND*sizeof(float4));
	m_bodies = NULL;
	if (options.custom_dir.length()>0)
		m_problem_dir = options.custom_dir;
	m_gdata= NULL;
}


Problem::~Problem(void)
{
	if (m_simparams.numbodies)
		delete [] m_bodies;
	if (m_rbdatafile != NULL) {
        fclose(m_rbdatafile);
    }
}

		
float
Problem::density(float h, int i)
{
	float density = m_physparams.rho0[i];

	if (h > 0) {
		float g = length(m_physparams.gravity);
		density = m_physparams.rho0[i]*pow(g*m_physparams.rho0[i]*h/m_physparams.bcoeff[i] + 1,
				1/m_physparams.gammacoeff[i]);
		}
	return density;
}


float
Problem::soundspeed(float rho, int i)
{
	return m_physparams.sscoeff[i]*pow(rho/m_physparams.rho0[i], m_physparams.sspowercoeff[i]);
}


float
Problem::pressure(float rho, int i) const
{
	return m_physparams.bcoeff[i]*(pow(rho/m_physparams.rho0[i], m_physparams.gammacoeff[i]) - 1);
}

bool
Problem::add_outlet(
			float min_x, float min_y, float min_z,
			float max_x, float max_y, float max_z,
			float dir_x, float dir_y, float dir_z)
{
	if (m_physparams.outlets == MAXOUTLETS)
		return false;
	uint outlet = m_physparams.outlets++;

	// minimum outlet coordinates
	m_physparams.outlet_min[outlet].x = min(min_x, max_x);
	m_physparams.outlet_min[outlet].y = min(min_y, max_y);
	m_physparams.outlet_min[outlet].z = min(min_z, max_z);

	// maximum outlet coordinates
	m_physparams.outlet_max[outlet].x = max(min_x, max_x);
	m_physparams.outlet_max[outlet].y = max(min_y, max_y);
	m_physparams.outlet_max[outlet].z = max(min_z, max_z);

	// outflow direction: check that only one component is zero
	if ( ( dir_x && (dir_y || dir_z) ) || (dir_y && dir_z) ) {
		// TODO introduce consistency error exception and throw that
		fprintf(stderr, "ERROR: outlet %u direction (%g, %g, %g) has two non-zero components\n",
			outlet, dir_x, dir_y, dir_z);
		return false;
	}

	float4 size = m_physparams.outlet_max[outlet] - m_physparams.outlet_min[outlet];
	float4 dir = make_float4(dir_x, dir_y, dir_z, 0);

	// outlet displacement vector: vector as long as the outlet in the direction of the flow
	// note that this is a component-by-component product, not a dot product
	m_physparams.outlet_disp[outlet] = size*dir;

	// outlet reflection plane: normal to the outflow direction, passes through a point
	// which is `offset` further than the outflow limit in that direction
	m_physparams.outlet_plane[outlet] = dir;
	float offset = 0; // TODO find the correct value

	// since the direction is normal to one of the planes, just do a search rather
	// than thinking up a creative formula
	if (dir_x > 0) {
		m_physparams.outlet_plane[outlet].w = -(m_physparams.outlet_max[outlet].x + offset);
	} else if (dir_x < 0) {
		m_physparams.outlet_plane[outlet].w = -(m_physparams.outlet_min[outlet].x - offset);
	} else if (dir_y > 0) {
		m_physparams.outlet_plane[outlet].w = -(m_physparams.outlet_max[outlet].y + offset);
	} else if (dir_y < 0) {
		m_physparams.outlet_plane[outlet].w = -(m_physparams.outlet_min[outlet].y - offset);
	} else if (dir_z > 0) {
		m_physparams.outlet_plane[outlet].w = -(m_physparams.outlet_max[outlet].z + offset);
	} else if (dir_z < 0) {
		m_physparams.outlet_plane[outlet].w = -(m_physparams.outlet_min[outlet].z - offset);
	} else {
		// TODO introduce consistency error exception and throw that
		fprintf(stderr, "ERROR: outlet %u direction (%g, %g, %g) is null\n",
			outlet, dir_x, dir_y, dir_z);
		return false;
	}

	printf("Outlet %u plane (%g %g %g %g)\n", outlet,
		m_physparams.outlet_plane[outlet].x,
		m_physparams.outlet_plane[outlet].y,
		m_physparams.outlet_plane[outlet].z,
		m_physparams.outlet_plane[outlet].w);

	return true;

}

bool
Problem::add_inlet(
			float min_x, float min_y, float min_z,
			float max_x, float max_y, float max_z,
			float vel_x, float vel_y, float vel_z, float vel_w)
{
	if (m_physparams.inlets == MAXOUTLETS)
		return false;
	uint inlet = m_physparams.inlets++;

	m_physparams.inlet_min[inlet].x = min(min_x, max_x);
	m_physparams.inlet_min[inlet].y = min(min_y, max_y);
	m_physparams.inlet_min[inlet].z = min(min_z, max_z);

	m_physparams.inlet_max[inlet].x = max(min_x, max_x);
	m_physparams.inlet_max[inlet].y = max(min_y, max_y);
	m_physparams.inlet_max[inlet].z = max(min_z, max_z);

	m_physparams.inlet_vel[inlet].x = vel_x;
	m_physparams.inlet_vel[inlet].y = vel_y;
	m_physparams.inlet_vel[inlet].z = vel_z;
	m_physparams.inlet_vel[inlet].w = vel_w;

	// the inlet displacement vector is computed automatically from the
	// min, max and vel

	float4 diff = m_physparams.inlet_max[inlet] - m_physparams.inlet_min[inlet];
	diff.w = 0;
	float4 vdir = make_float4(
		isfinite(vel_x) ? vel_x : 0,
		isfinite(vel_y) ? vel_y : 0,
		isfinite(vel_z) ? vel_z : 0,
		isfinite(vel_w) ? vel_w : 0);

	// opposite of projection of the box sizes on the velocity direction
	float4 disp = -dot(diff, vdir)*vdir/dot(vdir,vdir);
	disp.w = 0;

	// particles are regenerated when they pass _half_ the inlet
	m_physparams.inlet_disp[inlet] = disp/2;

	// TODO: check that inlet is long enough

	return true;

}

bool 
Problem::need_display(float t)
{
	if (t - m_last_display_time >= m_displayinterval) {
		m_last_display_time = t;
		return true;
	}

	return false;
}


std::string 
Problem::create_problem_dir(void)
{
	// if no custom dir was set, create one based on the name of the problem plus the time
	if (m_problem_dir.length()==0) {
		time_t  rawtime;
		char	time_str[17];

		time(&rawtime);
		strftime(time_str, 17, "%Y-%m-%d %Hh%M", localtime(&rawtime));
		time_str[16] = '\0';
		m_problem_dir = "./tests/" + m_name + ' ' + std::string(time_str);

		// create "./tests/" if it doesn't exist yet. Assuming this yield no error...
		mkdir("./tests/", S_IRWXU | S_IRWXG | S_IRWXO);
	}

	// customize the directory name according to the process rank
	if (m_gdata && m_gdata->mpi_nodes > 1)
		m_problem_dir += ".MPI_" + m_gdata->to_string(m_gdata->mpi_rank) + "." + m_gdata->to_string(m_gdata->mpi_nodes);

	// create the directory
	if (mkdir(m_problem_dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO)) {
		fprintf(stderr, " * WARNING: couldn't create directory %s\n",
			m_problem_dir.c_str());
		fprintf(stderr, "   Possible causes: no permessions, parent directory doesn't exist, etc.\n");
	}

	if (m_rbdata_writeinterval) {
		string rbdata_filename = m_problem_dir + "/rbdata.txt";
		m_rbdatafile = fopen(rbdata_filename.c_str(), "w");

		if (m_rbdatafile == NULL) {
			stringstream ss;
			ss << "Cannot open rigid bodies data file " << rbdata_filename;
			throw runtime_error(ss.str());
			}
	}
	return m_problem_dir;
}


bool 
Problem::need_write(float t)
{
	if (m_writefreq == 0)
		return false;

	if (t - m_last_write_time >= m_displayinterval*m_writefreq || (t == 0.0 && m_last_write_time != 0.0)) {
		return true;
	}

	return false;
}


bool 
Problem::need_write_rbdata(float t)
{
	if (m_rbdata_writeinterval == 0)
		return false;

	if (t - m_last_rbdata_write_time >= m_rbdata_writeinterval) {
		m_last_rbdata_write_time = t;
		return true;
	}

	return false;
}


void
Problem::write_rbdata(float t)
{
	if (m_simparams.numbodies) {
		if (need_write_rbdata(t)) {
			for (int i = 0; i < m_simparams.numbodies; i++) {
				m_bodies[i].Write(t, m_rbdatafile);
			}
		}
	}
}

bool 
Problem::need_screenshot(float t)
{
	if (m_screenshotfreq == 0)
		return false;

	if (t - m_last_screenshot_time >= m_displayinterval*m_screenshotfreq) {
		m_last_screenshot_time = t;
		return true;
	}

	return false;
}


// is the simulation finished at the given time?
bool 
Problem::finished(float t)
{
	float tend(m_simparams.tend);
	return tend && (t > tend);
}


MbCallBack& 
Problem::mb_callback(const float t, const float dt, const int i)
{
	return m_mbcallbackdata[i];
};


float3 
Problem::g_callback(const float t)
{
	return make_float3(0.0);
}

// Fill the device map with "devnums" (*global* device ids) in range [0..numDevices[.
// Default algorithm: split along the longest axis
void Problem::fillDeviceMap(GlobalData* gdata)
{
	fillDeviceMapByAxis(gdata, LONGEST_AXIS);
}

// partition by splitting the cells according to their linearized hash.
void Problem::fillDeviceMapByCellHash(GlobalData* gdata)
{
	uint cells_per_device = gdata->nGridCells / gdata->totDevices;
	for (uint i=0; i < gdata->nGridCells; i++)
		gdata->s_hDeviceMap[i] = min( i/cells_per_device, gdata->totDevices-1);
}

// partition by splitting along the specified axis
void Problem::fillDeviceMapByAxis(GlobalData* gdata, SplitAxis preferred_split_axis)
{
	// select the longest axis
	if (preferred_split_axis == LONGEST_AXIS) {
		if (	gdata->worldSize.x >= gdata->worldSize.y &&
				gdata->worldSize.x >= gdata->worldSize.z)
			preferred_split_axis = X_AXIS;
		else
		if (	gdata->worldSize.y >= gdata->worldSize.z)
			preferred_split_axis = Y_AXIS;
		else
			preferred_split_axis = Z_AXIS;
	}
	uint cells_per_longest_axis;
	switch (preferred_split_axis) {
		case X_AXIS:
			cells_per_longest_axis = gdata->gridSize.x;
			break;
		case Y_AXIS:
			cells_per_longest_axis = gdata->gridSize.y;
			break;
		case Z_AXIS:
			cells_per_longest_axis = gdata->gridSize.z;
			break;
	}
	uint cells_per_device_per_longest_axis = cells_per_longest_axis / gdata->totDevices;
	for (uint cx = 0; cx < gdata->gridSize.x; cx++)
		for (uint cy = 0; cy < gdata->gridSize.y; cy++)
			for (uint cz = 0; cz < gdata->gridSize.z; cz++) {
				uint axis_coordinate;
				switch (preferred_split_axis) {
					case X_AXIS: axis_coordinate = cx; break;
					case Y_AXIS: axis_coordinate = cy; break;
					case Z_AXIS: axis_coordinate = cz; break;
				}
				// everything is just a preparation for the following line
				uchar dstDevice = axis_coordinate / cells_per_device_per_longest_axis;
				// handle the case when cells_per_longest_axis multiplies cells_per_longest_axis
				dstDevice = min(dstDevice, gdata->totDevices - 1);
				// compute cell address
				uint cellLinearHash = gdata->calcGridHashHost(cx, cy, cz);
				// assign it
				gdata->s_hDeviceMap[cellLinearHash] = dstDevice;
			}
}

void Problem::fillDeviceMapByEquation(GlobalData* gdata)
{
	// 1st equation: (x+y+z / #devices)
	uint longest_grid_size = max ( max( gdata->gridSize.x, gdata->gridSize.y), gdata->gridSize.z );
	uint coeff = longest_grid_size /  (gdata->totDevices + 1);
	// 2nd equation: spheres
	uint diagonal = (uint) sqrt(	gdata->gridSize.x * gdata->gridSize.x +
									gdata->gridSize.y * gdata->gridSize.y +
									gdata->gridSize.z * gdata->gridSize.z) / 2;
	uint radius_part = diagonal /  gdata->totDevices;
	for (uint cx = 0; cx < gdata->gridSize.x; cx++)
		for (uint cy = 0; cy < gdata->gridSize.y; cy++)
			for (uint cz = 0; cz < gdata->gridSize.z; cz++) {
				uint dstDevice;
				// 1st equation: rough oblique plane split --
				dstDevice = (cx + cy + cz) / longest_grid_size;
				// -- end of 1st eq.
				// 2nd equation: spheres --
				//uint distance_from_origin = (uint) sqrt( cx * cx + cy * cy + cz * cz);
				// comparing directly the square would be more efficient but could require long uints
				//dstDevice = distance_from_origin / radius_part;
				// -- end of 2nd eq.
				// handle the case when cells_per_device multiplies cells_per_longest_axis
				dstDevice = min(dstDevice, gdata->totDevices - 1);
				// compute cell address
				uint cellLinearHash = gdata->calcGridHashHost(cx, cy, cz);
				// assign it
				gdata->s_hDeviceMap[cellLinearHash] = (uchar)dstDevice;
			}
}

void 
Problem::allocate_bodies(const int i)
{
	m_simparams.numbodies = i;
	m_bodies = new RigidBody[i];
}


RigidBody* 
Problem::get_body(const int i)
{
	if (i >= m_simparams.numbodies) {
		stringstream ss;
		ss << "get_body: body number " << i << " >= numbodies";
		throw runtime_error(ss.str());
	}
	return &m_bodies[i];
}


int 
Problem::get_body_numparts(const int i)
{
	if (!m_simparams.numbodies)
		return 0;

	return m_bodies[i].GetParts().size();
}


int 
Problem::get_bodies_numparts(void)
{
	int total_parts = 0;
	for (int i = 0; i < m_simparams.numbodies; i++) {
		total_parts += m_bodies[i].GetParts().size();
	}

	return total_parts;
}


void 
Problem::get_rigidbodies_data(float3 * & cg, float * & steprot)
{
	cg = m_bodies_cg;
	steprot = m_bodies_steprot;
}


float3* 
Problem::get_rigidbodies_cg(void)
{
	for (int i = 0; i < m_simparams.numbodies; i++)  {
		m_bodies[i].GetCG(m_bodies_cg[i]);
	}
	
	return m_bodies_cg;
}


float* 
Problem::get_rigidbodies_steprot(void)
{
	return m_bodies_steprot;
}


void 
Problem::rigidbodies_timestep(const float3 *force, const float3 *torque, const int step,
									const double dt, float3 * & cg, float3 * & trans, float * & steprot)
{
	for (int i = 0; i < m_simparams.numbodies; i++)  {
		m_bodies[i].TimeStep(force[i], m_physparams.gravity, torque[i], step, dt,
				m_bodies_cg + i, m_bodies_trans + i, m_bodies_steprot + 9*i);
		}
	cg = m_bodies_cg;
	steprot = m_bodies_steprot;
	trans = m_bodies_trans;
}

// Maximum number of particles that may be generated
uint
Problem::max_parts(uint numParts)
{
	if (m_physparams.inlets == 0)
		return numParts;

	// if we have inlets, we (over)estimate the number of particles in all the inlets,
	// and add a multiple of it to the number of particles
	uint inletParts = 0;
	for (uint inlet = 0; inlet < m_physparams.inlets; ++inlet) {
		float3 range = as_float3(m_physparams.inlet_max[inlet]) - as_float3(m_physparams.inlet_min[inlet]);
		range /= m_deltap; // regular fill
		uint iparts = max(range.x,1)*max(range.y,1)*max(range.z,1);
		printf("  estimating %u particles in inlet %u\n", iparts, inlet);

		// number of 'fills', computed (if possible) from (inlet vel)*tend/(inlet disp)
		float3 vel = as_float3(m_physparams.inlet_vel[inlet]);
		if (!isfinite(vel.x)) vel.x = 0;
		if (!isfinite(vel.y)) vel.y = 0;
		if (!isfinite(vel.z)) vel.z = 0;
		uint fills = length(vel)*m_simparams.tend/length(m_physparams.inlet_disp[inlet]);
		if (fills > 0) {
			printf("  estimating %u fills for inlet %u (%gs at %gm/s over %gm)\n", fills, inlet,
				m_simparams.tend, length(vel), length(m_physparams.inlet_disp[inlet]));
		} else {
			fills = 2;
			printf("Could not estimate fills for inlet %u, defaulting to %u\n", inlet, fills);
		}

		inletParts += iparts*fills;

	}

	// however, we assume that we can't have more particles than by filling the whole domain:
	float3 range = get_worldsize();
	range /= m_deltap; // regular fill
	uint wparts = max(range.x,1)*max(range.y,1)*max(range.z,1);
	printf("  estimating %u particles to fill the world\n", wparts);

	uint maxparts = min(wparts, numParts + inletParts);

	return maxparts;

}


// Number of planes
uint 
Problem::fill_planes(void)
{
	return 0;
}


// Copy planes for upload
void 
Problem::copy_planes(float4*, float*)
{
	return;
}


float4* 
Problem::get_mbdata(const float t, const float dt, const bool forceupdate)
{
	bool needupdate = false;

	for (int i=0; i < m_mbnumber; i++) {
		MbCallBack& mbcallbackdata = mb_callback(t, dt, i);
		float4 data = make_float4(0.0f);

		switch(mbcallbackdata.type) {
			case PISTONPART:
				data.x = mbcallbackdata.origin.x + mbcallbackdata.disp.x;
				break;

			case PADDLEPART:
				data.x = mbcallbackdata.origin.x;
				data.y = mbcallbackdata.origin.z;
				data.z = mbcallbackdata.sintheta;
				data.w = mbcallbackdata.costheta;
				break;

			case GATEPART:
				data.x = mbcallbackdata.vel.x;
				data.y = mbcallbackdata.vel.y;
				data.z = mbcallbackdata.vel.z;
				break;
		}
		if (m_mbdata[i].x != data.x || m_mbdata[i].y != data.y ||
			m_mbdata[i].z != data.z ||m_mbdata[i].w != data.w) {
			m_mbdata[i] = data;
			needupdate = true;
			}
	}

	if (needupdate || forceupdate)
		return m_mbdata;

	return NULL;
}


void
Problem::draw_axis()
{	
	float3 axis_center = m_origin + 0.5*m_size;
	float axis_length = std::max(std::max(m_size.x, m_size.y), m_size.z)/4.0;
	
	/* X axis in green */
	glColor3f(0.0f, 0.8f, 0.0f);
	glBegin(GL_LINES);
	glVertex3f(axis_center.x, axis_center.y, axis_center.z);
	glVertex3f(axis_center.x + axis_length, axis_center.y, axis_center.z);
	glEnd();
	
	/* Y axis in red */
	glColor3f(0.8f, 0.0f, 0.0f);
	glBegin(GL_LINES);
	glVertex3f(axis_center.x, axis_center.y, axis_center.z);
	glVertex3f(axis_center.x, axis_center.y  + axis_length, axis_center.z);
	glEnd();
	
	/* Z axis in blu */
	glColor3f(0.0f, 0.0f, 0.8f);
	glBegin(GL_LINES);
	glVertex3f(axis_center.x, axis_center.y, axis_center.z);
	glVertex3f(axis_center.x, axis_center.y, axis_center.z + axis_length);
	glEnd();
}

/* void
Problem::draw_inlets()
{
	glColor3f(1, 0, 1);
	for (uint inlet = 0; inlet < m_physparams.inlets; ++inlet) {
		float4 *lo = m_physparams.inlet_min + inlet;
		float4 *hi = m_physparams.inlet_max + inlet;

		glBegin(GL_QUADS);
		// bottom quad
		glVertex3f(lo->x, lo->y, lo->z);
		glVertex3f(hi->x, lo->y, lo->z);
		glVertex3f(hi->x, hi->y, lo->z);
		glVertex3f(lo->x, hi->y, lo->z);

		// top quad
		glVertex3f(hi->x, hi->y, hi->z);
		glVertex3f(lo->x, hi->y, hi->z);
		glVertex3f(lo->x, lo->y, hi->z);
		glVertex3f(hi->x, lo->y, hi->z);
		glEnd();

		// other four edges
		glBegin(GL_LINES);
		glVertex3f(lo->x, lo->y, lo->z);
		glVertex3f(lo->x, lo->y, hi->z);

		glVertex3f(hi->x, lo->y, lo->z);
		glVertex3f(hi->x, lo->y, hi->z);

		glVertex3f(hi->x, hi->y, lo->z);
		glVertex3f(hi->x, hi->y, hi->z);

		glVertex3f(lo->x, hi->y, lo->z);
		glVertex3f(lo->x, hi->y, hi->z);
		glEnd();
	}
}

void
Problem::draw_outlets()
{
	glColor3f(1, 0, 1);
	for (uint outlet = 0; outlet < m_physparams.outlets; ++outlet) {
		float4 *lo = m_physparams.outlet_min + outlet;
		float4 *hi = m_physparams.outlet_max + outlet;

		glBegin(GL_QUADS);
		// bottom quad
		glVertex3f(lo->x, lo->y, lo->z);
		glVertex3f(hi->x, lo->y, lo->z);
		glVertex3f(hi->x, hi->y, lo->z);
		glVertex3f(lo->x, hi->y, lo->z);

		// top quad
		glVertex3f(hi->x, hi->y, hi->z);
		glVertex3f(lo->x, hi->y, hi->z);
		glVertex3f(lo->x, lo->y, hi->z);
		glVertex3f(hi->x, lo->y, hi->z);
		glEnd();

		// other four edges
		glBegin(GL_LINES);
		glVertex3f(lo->x, lo->y, lo->z);
		glVertex3f(lo->x, lo->y, hi->z);

		glVertex3f(hi->x, lo->y, lo->z);
		glVertex3f(hi->x, lo->y, hi->z);

		glVertex3f(hi->x, hi->y, lo->z);
		glVertex3f(hi->x, hi->y, hi->z);

		glVertex3f(lo->x, hi->y, lo->z);
		glVertex3f(lo->x, hi->y, hi->z);
		glEnd();
	}
}
*/

