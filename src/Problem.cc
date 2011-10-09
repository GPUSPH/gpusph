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
#include <math.h>
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
	m_last_write_time = 0.0;
	m_last_screenshot_time = 0.0;
	m_mbnumber = 0;
	m_rbdatafile = NULL;
	memset(m_mbcallbackdata, 0, MAXMOVINGBOUND*sizeof(float4));
	m_bodies = NULL;
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
	time_t  rawtime;
	char	time_str[17];

	time(&rawtime);
	strftime(time_str, 17, "%Y-%m-%d %Hh%M", localtime(&rawtime));
	time_str[16] = '\0';
	// if "./tests/" doesn't exist yet...
	mkdir("./tests/", S_IRWXU | S_IRWXG | S_IRWXO);
	m_problem_dir = "./tests/" + m_name + ' ' + std::string(time_str);
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

	if (t - m_last_write_time >= m_displayinterval*m_writefreq) {
		m_last_write_time = t;
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