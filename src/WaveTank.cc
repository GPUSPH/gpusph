#include <math.h>
#include <stdio.h>
#ifdef __APPLE__
#include <OpenGl/gl.h>
#else
#include <GL/gl.h>
#endif

#include "WaveTank.h"
#include "particledefine.h"

#define MK_par 2

WaveTank::WaveTank(const Options &options) : Problem(options)
{
	// Size and origin of the simulation domain
	m_size = make_float3(9.0f, 0.4f, 1.0f);
	m_origin = make_float3(0.0f, 0.0f,0.0f);

	m_writerType = VTKWRITER;

	// Data for problem setup
	slope_length = 8.5f;
	h_length = 0.5f;
	height = .63f;
	beta = 4.2364*3.14116/180.0;

    wmakertype = 1; // 0 for paddle, 1 for solitary wave
    icyl = 0; // icyl= 0 means no cylinders
	icone = 0; // icone = 0 means no cone
	i_use_bottom_plane = 0; // 1 for real plane instead of boundary parts

	// SPH parameters
	set_deltap(0.04f);  //0.005f;
	m_simparams.slength = 1.3f*m_deltap;
	m_simparams.kernelradius = 2.0f;
	m_simparams.kerneltype = WENDLAND;
	m_simparams.dt = 0.00013f;
	m_simparams.xsph = false;
	m_simparams.dtadapt = true;
	m_simparams.dtadaptfactor = 0.2;
	m_simparams.buildneibsfreq = 10;
	m_simparams.shepardfreq = 20;
	m_simparams.mlsfreq = 0;
	//m_simparams.visctype = ARTVISC;
	// m_simparams.visctype = KINEMATICVISC;
	m_simparams.visctype = SPSVISC;
	m_simparams.usedem = false;
	m_simparams.tend = 10.0;

	m_simparams.vorticity = true;
	m_simparams.boundarytype = LJ_BOUNDARY;  //LJ_BOUNDARY or MK_BOUNDARY

    // Physical parameters
	H = 0.45f;
	m_physparams.gravity = make_float3(0.0f, 0.0f, -9.81f);
	float g = length(m_physparams.gravity);

	m_physparams.set_density(0, 1000.0f, 7.0f, 300*H);
	m_physparams.numFluids = 1;
	float r0 = m_deltap;
	m_physparams.r0 = r0;

	m_physparams.kinematicvisc =  1.0e-6f;
	m_physparams.smagfactor = 0.12*0.12*m_deltap*m_deltap;
	m_physparams.kspsfactor = (2.0/3.0)*0.0066*m_deltap*m_deltap;
	m_physparams.epsartvisc = 0.01*m_simparams.slength*m_simparams.slength;

	// BC when using LJ
	m_physparams.dcoeff = 5.0f*g*H;
    //set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5

	// BC when using MK
	m_physparams.MK_K = g*H;
	m_physparams.MK_d = 1.1*m_deltap/MK_par;
	m_physparams.MK_beta = MK_par;

     if (wmakertype == 0 ){
	    m_physparams.mborigin = make_float3(0.13f, r0, -0.1344);
		m_mbtstart = 0;
		m_mbtend = m_simparams.tend;
        m_simparams.mbcallback = true;
		// The stroke value is given at free surface level H
		float stroke = 0.1;
		// m_mbamplitude is the maximal angular value par paddle angle
		// Paddle angle is in [-m_mbamplitude, m_mbamplitude]
		m_mbamplitude = atan(stroke/(2.0*(H - m_physparams.mborigin.z)));
		m_mbomega = 2.0*M_PI;
       }
	 else {
	    m_simparams.mbcallback = true;
	    m_physparams.mborigin = make_float3(r0, 0.0, 0.0);
	    // Parameter for solitary wave generator
		// TODO: make some comments
	    float amplitude = 0.2f;
	    m_Hoh = amplitude/H;
	    float kappa = sqrt((3*m_Hoh)/(4.0*H*H));
	    float cel = sqrt(g*(H + amplitude));
	    m_S = sqrt(16.0*amplitude*H/3.0);
	//  std::cout << "cel:  " << cel << "\n";
	//  std::cout << "kappa:  " << kappa << "\n";
	 // std::cout << "m_Hoh:  " << m_Hoh << "\n";
	//  std::cout << "m_S:  " << m_S << "\n";
	    m_tau = 2.0*(3.8 + m_Hoh)/(kappa*cel);
	    std::cout << "m_tau: " << m_tau << "\n";
	 // m_tau=2.5f;
		m_mbtstart = 0.0;
		m_mbtend = m_tau;
		m_mbnextimeupdate = true;
		m_mbposx = r0;

//		m_physparams.mbv = make_float3(0.0f, 0.0f, .5f);
//	    m_physparams.mbtstart = make_float3(0.0,0.0,0.0);  //piston, paddlepart, gatepart
//	    m_physparams.mbtend = make_float3(m_tau,0.0,fabs(2.5*H/m_physparams.mbv.z));
//			std::cout << "mbv.z = " << m_physparams.mbv.z <<endl;
//	    m_physparams.mbphase = 0.0f;
//	    m_physparams.mbomega = 0.0f;
//        m_physparams.mbamplitude = m_deltap; // In this example mbamplitude is position of the piston
      }

	// Scales for drawing
	m_maxrho = density(H,0);
	m_minrho = m_physparams.rho0[0];
	m_minvel = 0.0f;
	//m_maxvel = sqrt(m_physparams.gravity*H);
	m_maxvel = 0.4f;

	// Drawing and saving times
	m_displayinterval = 0.001f;
	m_writefreq = 10;
	m_screenshotfreq = 0;

	// Call the callback function with t = 0;
	mb_callback(0.0);

	// Name of problem used for directory creation
	m_name = "WaveTank";
	create_problem_dir();
}



WaveTank::~WaveTank(void)
{
	release_memory();
}


void WaveTank::release_memory(void)
{
	parts.clear();
	paddle_parts.clear();
	boundary_parts.clear();
	gate_parts.clear();
	piston_parts.clear();
}

MbCallBack& WaveTank::mb_callback(float t)
{
	// Paddle
	if (wmakertype == 0) {
		float theta = m_mbamplitude*cos(m_mbomega*t);
		m_mbcallback.mbsincostheta.x = sin(theta);
		m_mbcallback.mbsincostheta.y = cos(theta);
		m_mbcallback.needupdate = true;
		m_mbcallback.type = PADDLEPART;
		}
	// Solitary wave
	else {
		m_mbcallback.type = PISTONPART;
		if (t >= m_mbtstart && t < m_mbtend) {
			float arg = 2.0*((3.8 + m_Hoh)*(t/m_tau - 0.5) - 2.0*m_Hoh*((m_mbposx/m_S) - 0.5));
			m_mbposx = m_S*(1.0 + tanh(arg))/2.0;
			m_mbcallback.mbdisp = m_mbposx;
			m_mbcallback.needupdate = true;
			m_mbnextimeupdate = true;
			}
		else {
			m_mbcallback.needupdate = m_mbnextimeupdate;
			m_mbnextimeupdate = false;
			}
		}
	return m_mbcallback;
}


int WaveTank::fill_parts()
{
	float r0 = m_physparams.r0;
	float width = m_size.y;
//	float l = h_length + slope_length;
//    w = width;
//    h = height;
    //float wd = m_deltap/2;

	float br = (m_simparams.boundarytype == MK_BOUNDARY ? m_deltap/MK_par : r0);


    experiment_box = Cube(Point(0, 0, 0), Vector(h_length + slope_length, 0, 0),
						Vector(0, width, 0), Vector(0, 0, height));

    paddle_length = 1.0f;
	paddle_width = width - 2*r0;
	paddle_origin(0) = m_physparams.mborigin.x;
	paddle_origin(1) = m_physparams.mborigin.y;
	paddle_origin(2) = m_physparams.mborigin.z;
	paddle_width = width - 2*r0;
//	paddle = Rect(paddle_origin, Vector(0, paddle_width, 0),
//			Vector(0, 0, paddle_length));
	paddle = Rect(paddle_origin, Vector(0, paddle_width, 0),
				Vector(paddle_length*m_mbcallback.mbsincostheta.x, 0,
					paddle_length*m_mbcallback.mbsincostheta.y));

//	Cube fluid = Cube(Point(wd+paddle_origin(0), wd, wd), Vector(l-2*wd-paddle_origin(0), 0, 0), Vector(0, w-2*wd, 0), Vector(0, 0, H-2*wd));
//	fluid.SetPartMass(m_deltap, m_physparams.rho0);
//	fluid.InnerFill(parts, m_deltap);

	boundary_parts.reserve(100);
	paddle_parts.reserve(500);
	parts.reserve(34000);
	gate_parts.reserve(2000);
	piston_parts.reserve(500);

    if (wmakertype == 0) {
	   paddle.SetPartMass(m_deltap, m_physparams.rho0[0]);
	   paddle.Fill(paddle_parts, br, true);
	}
	else {
	   experiment_box6 = Rect(Point(paddle_origin(0), 0, 0), Vector(0, width, 0),
			Vector(0, 0, height)); //origin end wall and moving piston
	   experiment_box6.SetPartMass(m_deltap, m_physparams.rho0[0]);    //  moving boundary
	   experiment_box6.Fill(piston_parts, br, true);
     }

	if (i_use_bottom_plane == 0) {
	   experiment_box1 = Rect(Point(h_length,0,0  ), Vector(0,width,0 ), Vector(slope_length/cos(beta), 0.0, slope_length*tan(beta)));
	   experiment_box1.SetPartMass(m_deltap, m_physparams.rho0[0]);
	   experiment_box1.Fill(boundary_parts,br,true);
	   std::cout << "bottom rectangle defined" <<"\n";
	   }

	if (icyl == 1) {
		Point p1 = Point(h_length + slope_length/(cos(beta)*10), width/2, -height);
		Point p2 = Point(h_length + slope_length/(cos(beta)*10), width/6,  -height);
		Point p3 = Point(h_length + slope_length/(cos(beta)*10), 5*width/6, -height);
		Point p4 = Point(h_length + slope_length/(cos(beta)*5), 0, -height);
		Point p5 = Point(h_length + slope_length/(cos(beta)*5),  width/3, -height);
		Point p6 = Point(h_length + slope_length/(cos(beta)*5), 2*width/3, -height);
		Point p7 = Point(h_length + slope_length/(cos(beta)*5),  width, -height);
		Point p8 = Point(h_length + 3*slope_length/(cos(beta)*10),  width/6, -height);
		Point p9 = Point(h_length + 3*slope_length/(cos(beta)*10),  width/2, -height);
		Point p10 = Point(h_length+ 3*slope_length/(cos(beta)*10), 5*width/6, -height);
		Point p11 = Point(h_length+ 4*slope_length/(cos(beta)*10), width/2, -height*.75);

	    cyl1 = Cylinder(p1,Vector(.05, 0, 0),Vector(0,0,height));
	    cyl1.SetPartMass(m_deltap, m_physparams.rho0[0]);
	    cyl1.FillBorder(gate_parts, br, true, true);
		cyl2 = Cylinder(p2,Vector(.025, 0, 0),Vector(0,0,height));
		cyl2.SetPartMass(m_deltap, m_physparams.rho0[0]);
		cyl2.FillBorder(gate_parts, br, false, false);
		cyl3 = Cylinder(p3,Vector(.025, 0, 0),Vector(0,0,height));
		cyl3.SetPartMass(m_deltap, m_physparams.rho0[0]);
		cyl3.FillBorder(gate_parts, br, false, false);
		cyl4 = Cylinder(p4,Vector(.025, 0, 0),Vector(0,0,height));
		cyl4.SetPartMass(m_deltap, m_physparams.rho0[0]);
		cyl4.FillBorder(gate_parts, br, false, false);
		cyl5  = Cylinder(p5,Vector(.025, 0, 0),Vector(0,0,height));
		cyl5.SetPartMass(m_deltap, m_physparams.rho0[0]);
		cyl5.FillBorder(gate_parts, br, false, false);
		cyl6 = Cylinder(p6,Vector(.025, 0, 0),Vector(0,0,height));
		cyl6.SetPartMass(m_deltap, m_physparams.rho0[0]);
		cyl6.FillBorder(gate_parts, br, false, false);
		cyl7 = Cylinder(p7,Vector(.025, 0, 0),Vector(0,0,height));
		cyl7.SetPartMass(m_deltap, m_physparams.rho0[0]);
		cyl7.FillBorder(gate_parts, br, false, false);
		cyl8 = Cylinder(p8,Vector(.025,0,0),Vector(0,0,height));
		cyl8.SetPartMass(m_deltap, m_physparams.rho0[0]);
		cyl8.FillBorder(gate_parts, br, false, false);
		cyl9 = Cylinder(p9,Vector(.025, 0, 0),Vector(0,0,height));
		cyl9.SetPartMass(m_deltap, m_physparams.rho0[0]);
		cyl9.FillBorder(gate_parts, br, false, false);
		cyl10 = Cylinder(p10,Vector(.025, 0, 0),Vector(0,0,height));
		cyl10.SetPartMass(m_deltap, m_physparams.rho0[0]);
		cyl10.FillBorder(gate_parts, br, false, false);
		/*     cyl11= Cylinder(Point(h_length+4*slope_length/(cos(beta)*10), width/2, H+4*r0),
			   Vector(0.2,0,0), Vector(0,0,height));
			   cyl1.SetPartMass(m_deltap, m_physparams.rho0[0]);
			   cyl11.FillBorder(gate_parts, br, true,true);

		 */
	}
	if (icone == 1) {
		 Point p1 = Point(h_length + slope_length/(cos(beta)*10), width/2, -height);
	     cone = Cone(p1,Vector(width/4,0.0,0.0), Vector(width/10,0.,0.), Vector(0,0,height));
		 cone.SetPartMass(m_deltap, m_physparams.rho0[0]);
		 cone.FillBorder(gate_parts, br, false,true);
    }


	Rect fluid;
	float z = 0;
	int n = 0;
//	std::cout << "m_deltap: " << m_deltap << "\n";
//	std::cout << "m_physparams.rho0: " << m_physparams.rho0 << "\n";
	while (z < H) {
		z = n*m_deltap + 1.5*r0;    //z = n*m_deltap + 1.5*r0;
		float x = paddle_origin(0) + (z - paddle_origin(2))*tan(m_mbamplitude)  + 1.0*r0/cos(m_mbamplitude);
		//float x = paddle_origin(0) + (z - paddle_origin(2))*tan(m_physparams.mbamplitude) + 0.25*r0;
		float l = h_length + z/tan(beta) - 1.5*r0/sin(beta) - x;
		fluid = Rect(Point(x,  r0, z),
				Vector(0, width-2.0*r0, 0), Vector(l, 0, 0));
		//float l = h_length - x + (z-br)/tan(beta);
		//fluid = Rect(Point(x, 0, z),
		//	Vector(0, width, 0), Vector(l, 0, 0));
		fluid.SetPartMass(m_deltap, m_physparams.rho0[0]);
		fluid.Fill(parts, m_deltap, true);
		n++;
	 }

    return parts.size() + boundary_parts.size() + paddle_parts.size() +gate_parts.size() + piston_parts.size();

	}


uint WaveTank::fill_planes()
{

    if (i_use_bottom_plane == 0) {
		return 5;
		}
	else {
		return 6;
		} //corresponds to number of planes
}


void WaveTank::copy_planes(float4 *planes, float *planediv)
{
	float w = m_size.y;
	float l = h_length + slope_length;

	//  plane is defined as a x + by +c z + d= 0
	planes[0] = make_float4(0, 0, 1.0, 0);   //bottom, where the first three numbers are the normal, and the last is d.
	planediv[0] = 1.0;
	planes[1] = make_float4(0, 1.0, 0, 0);   //wall
	planediv[1] = 1.0;
	planes[2] = make_float4(0, -1.0, 0, w); //far wall
	planediv[2] = 1.0;
 	planes[3] = make_float4(1.0, 0, 0, 0);  //end
 	planediv[3] = 1.0;
 	planes[4] = make_float4(-1.0, 0, 0, l);  //one end
 	planediv[4] = 1.0;
 	if (i_use_bottom_plane == 1)  {
		planes[5] = make_float4(-sin(beta),0,cos(beta), h_length*sin(beta));  //sloping bottom starting at x=h_length
		planediv[5] = 1.0;
	}
}


void WaveTank::draw_boundary(float t)
{
	float displace;
	glColor3f(0.0, 1.0, 0.0);
	experiment_box.GLDraw();

	glColor3f(1.0, 0.0, 0.0);

	// Paddle
	if (wmakertype == 0) {;
		Rect actual_paddle = Rect(paddle_origin, Vector(0, paddle_width, 0),
				Vector(paddle_length*m_mbcallback.mbsincostheta.x, 0,
						paddle_length*m_mbcallback.mbsincostheta.y));
		actual_paddle.GLDraw();
		glColor3f(0.5, 0.5, 1.0);
	}
	// Solitary wave
	else {
		Rect actual_gate = Rect(Point(m_physparams.mborigin.x + m_mbposx, 0, 0),
				Vector(0, paddle_width, 0),Vector(0, 0, paddle_length));
		actual_gate.GLDraw();
   	}

//	if (t < m_physparams.mbtend.z) {
//		  displace = m_physparams.mbv.z*t;}
//	else {
//		  displace = m_physparams.mbv.z*m_physparams.mbtend.z;
//		  }

	float width = m_size.y;
	if (icyl ==1) {
		Point p1 = Point(h_length + slope_length/(cos(beta)*10), width/2,    -height + displace);
	    Point p2 = Point(h_length + slope_length/(cos(beta)*10), width/6,    -height + displace);
	    Point p3 = Point(h_length + slope_length/(cos(beta)*10), 5*width/6,  -height + displace);
	    Point p4 = Point(h_length + slope_length/(cos(beta)*5), 0,           -height + displace);
	    Point p5 = Point(h_length + slope_length/(cos(beta)*5),  width/3,    -height + displace);
	    Point p6 = Point(h_length + slope_length/(cos(beta)*5), 2*width/3,   -height + displace);
	    Point p7 = Point(h_length + slope_length/(cos(beta)*5),  width,      -height + displace);
	    Point p8 = Point(h_length + 3*slope_length/(cos(beta)*10),  width/6, -height + displace);
        Point p9 = Point(h_length + 3*slope_length/(cos(beta)*10),  width/2, -height + displace);
	    Point p10 = Point(h_length+ 3*slope_length/(cos(beta)*10), 5*width/6,-height + displace);
        Point p11 = Point(h_length+ 4*slope_length/(cos(beta)*10), width/2,  -height + displace);

	    cyl1 = Cylinder(p1,Vector(.025,0,0),Vector(0,0,height));
	    cyl1.GLDraw();
		cyl2 = Cylinder(p2,Vector(.025,0,0),Vector(0,0,height));
		cyl2.GLDraw();
		cyl3= Cylinder(p3,Vector(.025,0,0),Vector(0,0,height));
		cyl3.GLDraw();
		cyl4= Cylinder(p4,Vector(.025,0,0),Vector(0,0,height));
		cyl4.GLDraw();
		cyl5= Cylinder(p5,Vector(.025,0,0),Vector(0,0,height));
		cyl5.GLDraw();
		cyl6= Cylinder(p6,Vector(.025,0,0),Vector(0,0,height));
		cyl6.GLDraw();
		cyl7= Cylinder(p7,Vector(.025,0,0),Vector(0,0,height));
		cyl7.GLDraw();
		cyl8= Cylinder(p8,Vector(.025,0,0),Vector(0,0,height));
		cyl8.GLDraw();
		cyl9= Cylinder(p9,Vector(.025,0,0),Vector(0,0,height));
		cyl9.GLDraw();
		cyl10= Cylinder(p10,Vector(.025,0,0),Vector(0,0,height));
		cyl10.GLDraw();
		}

	if (icone == 1) {
	 	Point p1 = Point(h_length + slope_length/(cos(beta)*10), width/2, -height + displace);
	  	cone = Cone(p1,Vector(width/4,0.0,0.0), Vector(width/10,0.,0.), Vector(0,0,height));
		cone.GLDraw();
		}
}


void WaveTank::copy_to_array(float4 *pos, float4 *vel, particleinfo *info)
{
	std::cout << "\nBoundary parts: " << boundary_parts.size() << "\n";
		std::cout << "      "<< 0  <<"--"<< boundary_parts.size() << "\n";
	for (uint i = 0; i < boundary_parts.size(); i++) {
		pos[i] = make_float4(boundary_parts[i]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(BOUNDPART, 0, i);  // first is type, object, 3rd id
	}
	int j = boundary_parts.size();
	std::cout << "Boundary part mass:" << pos[j-1].w << "\n";

	std::cout << "\nPaddle parts: " << paddle_parts.size() << "\n";
		std::cout << "      "<< j  <<"--"<< j+ paddle_parts.size() << "\n";
	for (uint i = j; i < j + paddle_parts.size(); i++) {
		pos[i] = make_float4(paddle_parts[i-j]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(PADDLEPART, 0, i);
	}
	j += paddle_parts.size();
	std::cout << "Paddle part mass:" << pos[j-1].w << "\n";

	std::cout << "\nPiston parts: " << piston_parts.size() << "\n";
	std::cout << "     " << j << "--" << j + piston_parts.size() << "\n";
	for (uint i = j; i < j + piston_parts.size(); i++) {
		pos[i] = make_float4(piston_parts[i-j]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i] = make_particleinfo(PISTONPART, 0, i);
	}
	j += piston_parts.size();
	std::cout << "Piston part mass:" << pos[j-1].w << "\n";

	std::cout << "\nGate parts: " << gate_parts.size() << "\n";
	std::cout << "       " << j << "--" << j+gate_parts.size() <<"\n";
	for (uint i = j; i < j + gate_parts.size(); i++) {
		pos[i] = make_float4(gate_parts[i-j]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i] = make_particleinfo(GATEPART, 0, i);
	}
	j += gate_parts.size();
	std::cout << "Gate part mass:" << pos[j-1].w << "\n";


	float g = length(m_physparams.gravity);
	std::cout << "\nFluid parts: " << parts.size() << "\n";
	std::cout << "      "<< j  <<"--"<< j+ parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		pos[i] = make_float4(parts[i-j]);
		// initializing density
		//       float rho = m_physparams.rho0*pow(1.+g*(H-pos[i].z)/m_physparams.bcoeff,1/m_physparams.gammacoeff);
		//        vel[i] = make_float4(0, 0, 0, rho);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
	    info[i]= make_particleinfo(FLUIDPART,0,i);

	}
	j += parts.size();
	std::cout << "Fluid part mass:" << pos[j-1].w << "\n";

	std::cout << " Everything uploaded" <<"\n";
}

#undef MK_par