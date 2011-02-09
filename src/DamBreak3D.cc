#include <math.h>
#ifdef __APPLE__
#include <OpenGl/gl.h>
#else
#include <GL/gl.h>
#endif

#include "DamBreak3D.h"
#include "Cube.h"
#include "Point.h"
#include "Vector.h"


DamBreak3D::DamBreak3D(const Options &options) : Problem(options)
{
	// Size and origin of the simulation domain
	m_size = make_float3(1.6f, 0.67f, 0.4f);
	m_origin = make_float3(0.0f, 0.0f, 0.0f);

	m_writerType = VTKWRITER;

	// SPH parameters
	set_deltap(0.015f);
	m_simparams.slength = 1.3f*m_deltap;
	m_simparams.kernelradius = 2.0f;
	m_simparams.kerneltype = WENDLAND;
	m_simparams.dt = 0.0001f;
	m_simparams.xsph = false;
	m_simparams.dtadapt = true;
	m_simparams.dtadaptfactor = 0.3;
	m_simparams.buildneibsfreq = 10;
	m_simparams.shepardfreq = 0;
	m_simparams.mlsfreq = 0;
	m_simparams.visctype = ARTVISC;
	//m_simparams.visctype = DYNAMICVISC;
    m_simparams.boundarytype= LJ_BOUNDARY;
	m_simparams.tend = 1.5f;

	// We have no moving boundary
	m_simparams.mbcallback = false;

	// Physical parameters
	H = 0.4f;
	m_physparams.gravity = make_float3(0.0, 0.0, -9.81f);
	float g = length(m_physparams.gravity);
	m_physparams.set_density(0,1000.0, 7.0f, 300.0f*H);
	
    //set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5
	m_physparams.dcoeff = 5.0f*g*H;
	m_physparams.r0 = m_deltap;
	
	// BC when using MK boundary condition: Coupled with m_simsparams.boundarytype=MK_BOUNDARY
	#define MK_par 2
	m_physparams.MK_K = g*H;
	m_physparams.MK_d = 1.1*m_deltap/MK_par;
	m_physparams.MK_beta = MK_par;
	#undef MK_par
	
	m_physparams.kinematicvisc = 1.0e-6f;
	m_physparams.artvisccoeff = 0.3f;
	m_physparams.epsartvisc = 0.01*m_simparams.slength*m_simparams.slength;
	
	// Scales for drawing
	m_maxrho = density(H,0);
	m_minrho = m_physparams.rho0[0];
	m_minvel = 0.0f;
	//m_maxvel = sqrt(m_physparams.gravity*H);
	m_maxvel = 3.0f;
	
	// Drawing and saving times
	m_displayinterval = 0.05f;
	m_writefreq = 1;
	m_screenshotfreq = 0;
	
	// Name of problem used for directory creation
	m_name = "DamBreak3D";
	create_problem_dir();
}


DamBreak3D::~DamBreak3D(void)
{
	release_memory();
}


void DamBreak3D::release_memory(void)
{
	parts.clear();
	obstacle_parts.clear();
	boundary_parts.clear();
}


int DamBreak3D::fill_parts()
{
	float r0 = m_physparams.r0;

	Cube fluid, fluid1, fluid2, fluid3, fluid4;

	experiment_box = Cube(Point(0, 0, 0), Vector(1.6, 0, 0),
						Vector(0, 0.67, 0), Vector(0, 0, 0.4));

	obstacle = Cube(Point(0.9, 0.24, r0), Vector(0.12, 0, 0),
					Vector(0, 0.12, 0), Vector(0, 0, 0.4 - r0));

	fluid = Cube(Point(r0, r0, r0), Vector(0.4, 0, 0),
				Vector(0, 0.67 - 2*r0, 0), Vector(0, 0, 0.4 - r0));

	bool wet = false;	// set wet to true have a wet bed experiment
	if (wet) {
		fluid1 = Cube(Point(0.4 + m_deltap + r0 , r0, r0), Vector(0.5 - m_deltap - 2*r0, 0, 0),
					Vector(0, 0.67 - 2*r0, 0), Vector(0, 0, 0.03));

		fluid2 = Cube(Point(1.02 + r0 , r0, r0), Vector(0.58 - 2*r0, 0, 0),
					Vector(0, 0.67 - 2*r0, 0), Vector(0, 0, 0.03));

		fluid3 = Cube(Point(0.9 , m_deltap , r0), Vector(0.12, 0, 0),
					Vector(0, 0.24 - 2*r0, 0), Vector(0, 0, 0.03));

		fluid4 = Cube(Point(0.9 , 0.36 + m_deltap , r0), Vector(0.12, 0, 0),
					Vector(0, 0.31 - 2*r0, 0), Vector(0, 0, 0.03));
	}

	boundary_parts.reserve(2000);
	parts.reserve(14000);

	experiment_box.SetPartMass(r0, m_physparams.rho0[0]);
	experiment_box.FillBorder(boundary_parts, r0, false);

	obstacle.SetPartMass(r0, m_physparams.rho0[0]);
	obstacle.FillBorder(obstacle_parts, r0, true);

	fluid.SetPartMass(m_deltap, m_physparams.rho0[0]);
	fluid.Fill(parts, m_deltap, true);

	if (wet) {
		fluid1.SetPartMass(m_deltap, m_physparams.rho0[0]);
		fluid1.Fill(parts, m_deltap, true);
		fluid2.SetPartMass(m_deltap, m_physparams.rho0[0]);
		fluid2.Fill(parts, m_deltap, true);
		fluid3.SetPartMass(m_deltap, m_physparams.rho0[0]);
		fluid3.Fill(parts, m_deltap, true);
		fluid4.SetPartMass(m_deltap, m_physparams.rho0[0]);
		fluid4.Fill(parts, m_deltap, true);
	}

	return parts.size() + boundary_parts.size() + obstacle_parts.size();
}


void DamBreak3D::draw_boundary(float t)
{
	glColor3f(0.0, 1.0, 0.0);
	experiment_box.GLDraw();
	glColor3f(1.0, 0.0, 0.0);
	obstacle.GLDraw();
}


void DamBreak3D::copy_to_array(float4 *pos, float4 *vel, particleinfo *info)
{
	std::cout << "Boundary parts: " << boundary_parts.size() << "\n";
	for (uint i = 0; i < boundary_parts.size(); i++) {
		pos[i] = make_float4(boundary_parts[i]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(BOUNDPART,0,i);
	}
	int j = boundary_parts.size();
	std::cout << "Boundary part mass:" << pos[j-1].w << "\n";

	std::cout << "Obstacle parts: " << obstacle_parts.size() << "\n";
	for (uint i = j; i < j + obstacle_parts.size(); i++) {
		pos[i] = make_float4(obstacle_parts[i-j]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(BOUNDPART,1,i);
	}
	j += obstacle_parts.size();
	std::cout << "Obstacle part mass:" << pos[j-1].w << "\n";

	std::cout << "Fluid parts: " << parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		pos[i] = make_float4(parts[i-j]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(FLUIDPART,0,i);
	}
	j += parts.size();
	std::cout << "Fluid part mass:" << pos[j-1].w << "\n";
}
