#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "Problem.h"
#include "vector_math.h"


/* NOT NEEDED NOW

void
Problem::skipComment(ifstream &fin) {
	while (!fin.eof() && (fin.get() != '\n'));
	return;
}


int
Problem::read_init(char* fname)
{
	ifstream finit(fname);
	if (!finit) {
			cout << "File not found\n";
			return -1;
			}

	string s;
	s.reserve(256);

	int nl = 0;
	float temp = 0;
	float H;
	bool err = false;
	for (int i = 1;i <= 31; i++) {
		finit >> s;
		if (s.find("originx:") != string::npos) { finit >> m_origin.x; skipComment(finit); nl++; }
		else if (s.find("originy:") != string::npos) { finit >> m_origin.y; skipComment(finit); nl++; }
		else if (s.find("sizex:") != string::npos) { finit >> m_size.x; skipComment(finit); nl++; }
		else if (s.find("sizey:") != string::npos) { finit >> m_size.y; skipComment(finit); nl++; }
		else if (s.find("deltap:") != string::npos) { finit >> m_deltap; skipComment(finit); nl++; }
		else if (s.find("slength_factor:") != string::npos) {
			finit >> temp;
			m_simparams.slength = temp*m_deltap;
			skipComment(finit);
			nl++;
			}

		// else if (s.find("kerneltype:") != string::npos) { finit >> m_simparams.kerneltype; skipComment(finit); nl++; }
		else if (s.find("xsph:") != string::npos) { finit >> m_simparams.xsph; skipComment(finit); nl++; }
		else if (s.find("epsxsph:") != string::npos) { finit >> m_physparams.epsxsph; skipComment(finit); nl++; }
		else if (s.find("dt:") != string::npos) { finit >> m_simparams.dt; skipComment(finit); nl++; }
		else if (s.find("dtadapt:") != string::npos) { finit >> m_simparams.dtadapt; skipComment(finit); nl++; }
		else if (s.find("buildneibsfreq:") != string::npos) { finit >> m_simparams.buildneibsfreq; skipComment(finit); nl++; }
		else if (s.find("shepardfreq:") != string::npos) { finit >> m_simparams.shepardfreq; skipComment(finit); nl++; }
	//	else if (s.find("rho:") != string::npos) { finit >> m_physparams.rho0; skipComment(finit); nl++; }
		//else if (s.find("gravity:") != string::npos) { finit >> m_physparams.gravity; skipComment(finit); nl++; }
		else if (s.find("H:") != string::npos) { finit >> H; skipComment(finit); nl++; }
		else if (s.find("gamma:") != string::npos) { finit >> m_physparams.gammacoeff; skipComment(finit); nl++; }
		else if (s.find("p1:") != string::npos) { finit >> m_physparams.p1coeff; skipComment(finit); nl++; }
		else if (s.find("p2:") != string::npos) { finit >> m_physparams.p2coeff; skipComment(finit); nl++; }

		// else if (s.find("visctype:") != string::npos) { finit >> m_simparams.visctype; skipComment(finit); nl++; }
		else if (s.find("alpha:") != string::npos) { finit >> m_physparams.visccoeff; skipComment(finit); nl++; }
		else if (s.find("mboriginx:") != string::npos) { finit >> m_physparams.mborigin.x; skipComment(finit); nl++; }
		else if (s.find("mboriginy:") != string::npos) { finit >> m_physparams.mborigin.y; skipComment(finit); nl++; }
		else if (s.find("mbvx:") != string::npos) { finit >> m_physparams.mbv.x; skipComment(finit); nl++; }
		else if (s.find("mbvy:") != string::npos) { finit >> m_physparams.mbv.y; skipComment(finit); nl++; }
		else if (s.find("mbamplitude:") != string::npos) { finit >> m_physparams.mbamplitude; skipComment(finit); nl++; }
		else if (s.find("mbomega:") != string::npos) { finit >> m_physparams.mbomega; skipComment(finit); nl++; }
		else if (s.find("minvel:") != string::npos) { finit >> m_minvel; skipComment(finit); nl++; }
		else if (s.find("maxvel:") != string::npos) { finit >> m_maxvel; skipComment(finit); nl++; }
		else if (s.find("minrho:") != string::npos) { finit >> m_minrho; skipComment(finit); nl++; }
		else if (s.find("maxrho:") != string::npos) { finit >> m_maxrho; skipComment(finit); nl++; }
		else err = true;
		}
	if (nl != 31) err = true;

	finit.close();

	if (err) {
		cout << "Bad file format\n";
		return -1;
		}

	if (m_simparams.kerneltype == QUINTICSPLINE)
		m_simparams.kernelradius = 3.0f;
	else
		m_simparams.kernelradius = 2.0f;

	if (m_simparams.visctype == KINEMATICVISC)
		m_physparams.visccoeff *= 4.0f;

   // m_physparams.bcoeff = 200.0f*m_physparams.rho0*m_physparams.gravity*H/m_physparams.gammacoeff;
//	m_physparams.sscoeff = sqrt(m_physparams.bcoeff*m_physparams.gammacoeff/m_physparams.rho0);
//	m_physparams.sspowercoeff = (m_physparams.gammacoeff - 1)/2.0f;
   // m_physparams.dcoeff = 5.0f*m_physparams.gravity*H;

	return 0;
}
*/

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
	// tm	  *timeinfo;
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


float4* Problem::get_mbdata(const float t, const float dt)
{
	bool needupdate = false;

	for (int i=0; i < m_mbnumber; i++) {
		MbCallBack& mbcallbackdata = mb_callback(t, dt, i);
		float4 data = make_float4(0.0f);
		if (mbcallbackdata.needupdate)
			needupdate = true;

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
		m_mbdata[i] = data;
	}

	if (needupdate)
		return m_mbdata;

	return NULL;
}