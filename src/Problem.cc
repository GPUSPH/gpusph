#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "Problem.h"
#include "vector_math.h"


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


bool Problem::need_display(float t)
{
	if (t - m_last_display_time >= m_displayinterval) {
		m_last_display_time = t;
		return true;
	}

	return false;
}


std::string Problem::create_problem_dir(void)
{
	time_t  rawtime;
	char	time_str[32];

	time(&rawtime);
	// timeinfo = localtime(&rawtime);
	ctime_r(&rawtime, time_str);
	time_str[13]='h';
	time_str[16] = '\0';
	// if "./tests/" doesn't exist yet...
	mkdir("./tests/", S_IRWXU | S_IRWXG | S_IRWXO);
	m_problem_dir = "./tests/" + m_name + ' ' + std::string(time_str);
	mkdir(m_problem_dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);

	return m_problem_dir;
}


bool Problem::need_write(float t)
{
	if (m_writefreq == 0)
		return false;

	if (t - m_last_write_time >= m_displayinterval*m_writefreq) {
		m_last_write_time = t;
		return true;
	}

	return false;
}


bool Problem::need_screenshot(float t)
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
bool Problem::finished(float t)
{
	float tend(m_simparams.tend);
	return tend && (t > tend);
}


MbCallBack& Problem::mb_callback(const float t, const float dt, const int i)
{
	return m_mbcallbackdata[i];
};

// Number of planes
uint Problem::fill_planes(void)
{
	return 0;
}


// Copy planes for upload
void Problem::copy_planes(float4*, float*)
{
	return;
}


float4* Problem::get_mbdata(const float t, const float dt, const bool forceupdate)
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