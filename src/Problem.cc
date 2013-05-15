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

int Problem::m_total_ODE_bodies = 0;

Problem::Problem(const Options &options)
{
	m_options = options;
	m_last_display_time = 0.0;
	m_last_write_time = -1.0;
	m_last_screenshot_time = 0.0;
	m_mbnumber = 0;
	m_rbdatafile = NULL;
	memset(m_mbcallbackdata, 0, MAXMOVINGBOUND*sizeof(float4));
	m_ODE_bodies = NULL;
	m_problem_dir = options.dir;
}


Problem::~Problem(void)
{
	if (m_ODE_bodies)
		delete [] m_ODE_bodies;
	if (m_rbdatafile != NULL) {
        fclose(m_rbdatafile);
    }
}

void
Problem::check_dt(void)
{
	float dt_from_sspeed = INFINITY;
	for (uint f = 0 ; f < m_physparams.numFluids; ++f) {
		float sspeed = m_physparams.sscoeff[f];
		dt_from_sspeed = fmin(dt_from_sspeed, m_simparams.slength/sspeed);
	}
	dt_from_sspeed *= m_simparams.dtadaptfactor;

	float dt_from_gravity = sqrt(m_simparams.slength/length(m_physparams.gravity));
	dt_from_gravity *= m_simparams.dtadaptfactor;

	float dt_from_visc = NAN;
	if (m_simparams.visctype != ARTVISC) {
		dt_from_visc = m_simparams.slength*m_simparams.slength/m_physparams.kinematicvisc;
		dt_from_visc *= 0.125; // TODO this should be configurable
	}

	float cfl_dt = fmin(dt_from_sspeed, fmin(dt_from_gravity, dt_from_visc));

	if (m_simparams.dt > cfl_dt) {
		fprintf(stderr, "WARNING: dt %g bigger than %g imposed by CFL conditions (sspeed: %g, gravity: %g, viscosity: %g)\n",
			m_simparams.dt, cfl_dt,
			dt_from_sspeed, dt_from_gravity, dt_from_visc);
	} else if (!m_simparams.dt) { // dt wasn't set
			m_simparams.dt = cfl_dt;
			printf("setting dt = %g from CFL conditions (soundspeed: %g, gravity: %g, viscosity: %g)\n",
				m_simparams.dt,
				dt_from_sspeed, dt_from_gravity, dt_from_visc);
	} else {
			printf("dt = %g (CFL conditions from soundspeed: %g, from gravity %g, from viscosity %g)\n",
				m_simparams.dt,
				dt_from_sspeed, dt_from_gravity, dt_from_visc);
	}

}


float
Problem::density(float h, int i) const
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
Problem::soundspeed(float rho, int i) const
{
	return m_physparams.sscoeff[i]*pow(rho/m_physparams.rho0[i], m_physparams.sspowercoeff[i]);
}


float
Problem::pressure(float rho, int i) const
{
	return m_physparams.bcoeff[i]*(pow(rho/m_physparams.rho0[i], m_physparams.gammacoeff[i]) - 1);
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

void
Problem::add_gage(double3 const& pt)
{
	m_simparams.gage.push_back(pt);
}

std::string const&
Problem::create_problem_dir(void)
{
	// if no data save directory was specified, default to a name
	// composed of problem name followed by date and time
	if (m_problem_dir.empty()) {
		time_t  rawtime;
		char	time_str[18];

		time(&rawtime);
		strftime(time_str, 18, "_%Y-%m-%dT%Hh%M", localtime(&rawtime));
		time_str[17] = '\0';
		// if "./tests/" doesn't exist yet...
		mkdir("./tests/", S_IRWXU | S_IRWXG | S_IRWXO);
		m_problem_dir = "./tests/" + m_name + std::string(time_str);
	}

	// TODO it should be possible to specify a directory with %-like
	// replaceable strings, such as %{problem} => problem name,
	// %{time} => launch time, etc.

	mkdir(m_problem_dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);

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
	if (m_simparams.numODEbodies) {
		if (need_write_rbdata(t)) {
			for (int i = 0; i < m_simparams.numODEbodies; i++) {
				const dReal* quat = dBodyGetQuaternion(m_ODE_bodies[i]->m_ODEBody);
				const dReal* cg = dBodyGetPosition(m_ODE_bodies[i]->m_ODEBody);
				fprintf(m_rbdatafile, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", i, t, cg[0],
						cg[1], cg[2], quat[0], quat[1], quat[2], quat[3]);
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


void
Problem::allocate_ODE_bodies(const int i)
{
	m_simparams.numODEbodies = i;
	m_ODE_bodies = new Object *[i];
}


Object*
Problem::get_ODE_body(const int i)
{
	if (i >= m_simparams.numODEbodies) {
		stringstream ss;
		ss << "get_ODE_body: body number " << i << " >= numbodies";
		throw runtime_error(ss.str());
	}
	return m_ODE_bodies[i];
}


void
Problem::add_ODE_body(Object* object)
{
	if (m_total_ODE_bodies >= m_simparams.numODEbodies) {
		stringstream ss;
		ss << "add_ODE_body: body number " << m_total_ODE_bodies << " >= numbodies";
		throw runtime_error(ss.str());
	}
	m_ODE_bodies[m_total_ODE_bodies] = object;
	m_total_ODE_bodies++;
}


int 
Problem::get_ODE_bodies_numparts(void)
{
	int total_parts = 0;
	for (int i = 0; i < m_simparams.numODEbodies; i++) {
		total_parts += m_ODE_bodies[i]->GetParts().size();
	}

	return total_parts;
}


int
Problem::get_ODE_body_numparts(const int i)
{
	if (!m_simparams.numODEbodies)
		return 0;

	return m_ODE_bodies[i]->GetParts().size();
}


void 
Problem::get_ODE_bodies_data(float3 * & cg, float * & steprot)
{
	cg = m_bodies_cg;
	steprot = m_bodies_steprot;
}


float3*
Problem::get_ODE_bodies_cg(void)
{
	for (int i = 0; i < m_simparams.numODEbodies; i++)  {
		m_bodies_cg[i] = make_float3(dBodyGetPosition(m_ODE_bodies[i]->m_ODEBody));
	}

	return m_bodies_cg;
}


float* 
Problem::get_ODE_bodies_steprot(void)
{
	return m_bodies_steprot;
}


void
Problem::ODE_bodies_timestep(const float3 *force, const float3 *torque, const int step,
		const double dt, float3 * & cg, float3 * & trans, float * & steprot)
{
	dReal prev_quat[MAXBODIES][4];
	for (int i = 0; i < m_total_ODE_bodies; i++)  {
		const dReal* quat = dBodyGetQuaternion(m_ODE_bodies[i]->m_ODEBody);
		prev_quat[i][0] = quat[0];
		prev_quat[i][1] = quat[1];
		prev_quat[i][2] = quat[2];
		prev_quat[i][3] = quat[3];
		dBodyAddForce(m_ODE_bodies[i]->m_ODEBody, force[i].x, force[i].y, force[i].z);
		dBodyAddTorque(m_ODE_bodies[i]->m_ODEBody, torque[i].x, torque[i].y, torque[i].z);
	}

	dSpaceCollide(m_ODESpace, (void *) this, &ODE_near_callback_wrapper);
	dWorldStep(m_ODEWorld, dt);
	dJointGroupEmpty(m_ODEJointGroup);

	for (int i = 0; i < m_simparams.numODEbodies; i++)  {
		float3 new_cg = make_float3(dBodyGetPosition(m_ODE_bodies[i]->m_ODEBody));
		m_bodies_trans[i] = new_cg - m_bodies_cg[i];
		m_bodies_cg[i] = new_cg;
		const dReal *new_quat = dBodyGetQuaternion(m_ODE_bodies[i]->m_ODEBody);
		dQuaternion step_quat;
		dMatrix3 R;
		dQMultiply2 (step_quat, new_quat, prev_quat[i]);
		dQtoR (step_quat, R);
		float *base_addr = m_bodies_steprot + 9*i;
		base_addr[0] = R[0];
		base_addr[1] = R[1];
		base_addr[2] = R[2]; // Skipp R[3]
		base_addr[3] = R[4];
		base_addr[4] = R[5];
		base_addr[5] = R[6]; // Skipp R[7]
		base_addr[6] = R[8];
		base_addr[7] = R[9];
		base_addr[8] = R[10];
	}
	cg = m_bodies_cg;
	steprot = m_bodies_steprot;
	trans = m_bodies_trans;
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
				data.x = mbcallbackdata.vel.x;
				break;

			case PADDLEPART:
				data.x = mbcallbackdata.origin.x;
				data.y = mbcallbackdata.origin.z;
				data.z = mbcallbackdata.dthetadt;
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
	float3 axis_center = make_float3(m_origin + 0.5*m_size);
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


/*! Compute grid and cell size from the kernel influence radius
 * The number of cell is obtained as the ratio between the domain size and the
 * influence radius, rounded down to the closest integer.
 * The reason for rounding down is that we want the cell size to be no smaller
 * than the influence radius, to guarantee that all neighbors of a particle are
 * found at most one cell away in each direction.
 */
void
Problem::set_grid_params(void)
{
	double influenceRadius = m_simparams.kernelradius*m_simparams.slength;

	m_gridsize.x = floor(m_size.x / influenceRadius);
	m_gridsize.y = floor(m_size.y / influenceRadius);
	m_gridsize.z = floor(m_size.z / influenceRadius);

	m_cellsize.x = m_size.x / m_gridsize.x;
	m_cellsize.y = m_size.y / m_gridsize.y;
	m_cellsize.z = m_size.z / m_gridsize.z;

	printf("set_grid_params\t:\n");
	printf("Domain size\t: (%f, %f, %f)\n", m_size.x, m_size.y, m_size.z);
	printf("Grid   size\t: (%d, %d, %d)\n", m_gridsize.x, m_gridsize.y, m_gridsize.z);
	printf("Cell   size\t: (%f, %f, %f)\n", m_cellsize.x, m_cellsize.y, m_cellsize.z);
	printf("       delta\t: (%.2f%%, %.2f%%, %.2f%%)\n",
		(m_cellsize.x - influenceRadius)*100/influenceRadius,
		(m_cellsize.y - influenceRadius)*100/influenceRadius,
		(m_cellsize.z - influenceRadius)*100/influenceRadius);
}


// Compute position in uniform grid (clamping to edges)
int3
Problem::calc_grid_pos(const Point&	pos)
{
	int3 gridPos;
	gridPos.x = floor((pos(0) - m_origin.x) / m_cellsize.x);
	gridPos.y = floor((pos(1) - m_origin.y) / m_cellsize.y);
	gridPos.z = floor((pos(2) - m_origin.z) / m_cellsize.z);
	gridPos.x = min(max(0, gridPos.x), m_gridsize.x-1);
	gridPos.y = min(max(0, gridPos.y), m_gridsize.y-1);
	gridPos.z = min(max(0, gridPos.z), m_gridsize.z-1);

	return gridPos;
}


// Compute address in grid from position
uint
Problem::calc_grid_hash(int3 gridPos)
{
	return gridPos.z*m_gridsize.y*m_gridsize.x + gridPos.y*m_gridsize.x + gridPos.x;
}


void
Problem::calc_localpos_and_hash(const Point& pos, float4& localpos, uint& hash)
{
	int3 gridPos = calc_grid_pos(pos);
	hash = calc_grid_hash(gridPos);
	localpos.x = float(pos(0) - m_origin.x - (gridPos.x + 0.5)*m_cellsize.x);
	localpos.y = float(pos(1) - m_origin.y - (gridPos.y + 0.5)*m_cellsize.y);
	localpos.z = float(pos(2) - m_origin.z - (gridPos.z + 0.5)*m_cellsize.z);
	localpos.w = float(pos(3));
}
