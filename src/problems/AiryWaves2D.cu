/*  Copyright 2011-2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

    Istituto Nazionale di Geofisica e Vulcanologia
        Sezione di Catania, Catania, Italy

    Università di Catania, Catania, Italy

    Johns Hopkins University, Baltimore, MD

    This file is modified based on the Wave Tank problem of GPUSPH by Morteza Derakhti derakhti@jhu.edu

*/

#include <iostream>
#include <stdexcept>

#include "AiryWaves2D.h"
#include "particledefine.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

#define MK_par 2

class LinearWaveFDPKA4 {
    public:
      LinearWaveFDPKA4(void) ;         //constructor
      LinearWaveFDPKA4(  double &T,  double &h);   //constructor 2
      ~LinearWaveFDPKA4(void) {};  
      
      double wavenum();
      
    private:
      double sig;
      double kh;
      double  k;
      double  pi;
      double  h;
      double  T;
      double  g;
      double  f;
      double  fp;
    //  double  theta;
  };  //end of class declaration
  
 LinearWaveFDPKA4 :: LinearWaveFDPKA4(  double &wavePeriod,  double &waterDepth)   
   {
     h = waterDepth;
     T = wavePeriod;
    }
    
 double LinearWaveFDPKA4 :: wavenum() 
    {
      float g, pi, sig, f, fp;
      g=9.81;
      pi=4.0*atan(1.0);
      sig= 2*pi/T;   
      double k0=sig*sig/g;
      k =k0*pow(1.-exp(-pow(k0*h,1.25)),-0.4);
      for (int i=1;i<4;i++) 
        {
          f=k0-k*tanh(k*h);
          fp=-tanh(k*h)-k*h*(1-pow(tanh(k*h),2));
          k=k-f/fp;
          if (f<0.000001)
             break;
         }     
return k;
  }
  
AiryWaves2D::AiryWaves2D(GlobalData *_gdata) : XProblem(_gdata)
{
	m_usePlanes = get_option("use-planes", true); // --use-planes true to enable use of planes for boundaries

	// density diffusion terms: 0 none, 1 Molteni & Colagrossi, 2 Ferrari
	const int rhodiff = get_option("density-diffusion", 1);
	
	const float SS = get_option("Sound_Speed",90.0);
	const float Cd = get_option("Cd",0.12);
	const float nu = get_option("Nu",1.0e-6);
	
	SETUP_FRAMEWORK(
		//viscosity<SPSVISC>,
		boundary<LJ_BOUNDARY>,
		periodicity<PERIODIC_Y>,
		add_flags<ENABLE_CSPM>
	).select_options(
	COLAGROSSI,
		//rhodiff, FlagSwitch<ENABLE_NONE, ENABLE_DENSITY_DIFFUSION, ENABLE_FERRARI>(),
		m_usePlanes, add_flags<ENABLE_PLANES>()
	);

	
	set_deltap(1.0/75.0);
	//set_deltap(0.01f);
	
	beta = 0.1f;
	horizontal_flat = 5.0f;
	float z_slope = 2.0f;
	lx = horizontal_flat + z_slope/beta;
	slope_length = z_slope/beta;
	ly = 15.0f * m_deltap;
	lz = 3.0f;
	H  = 0.5f; // Still water level

	resize_neiblist(145);

        std::cout << "beach_slope is:"<< beta <<"\n";
        std::cout << "lx is:"<< lx <<"\n";
        std::cout << "ly is:"<< ly <<"\n";
	
	m_size = make_double3(lx, ly, lz);
	m_origin = make_double3(0, 0, 0);

	//addFilter(SHEPARD_FILTER, 1);

	// use a plane for the bottom
	use_bottom_plane = 1;  //1 for plane; 0 for particles

	//WavePeriod & WaveHeight
        WavePeriod = 1.2f;
	WaveHeight = 0.1f;
	
	LinearWaveFDPKA4 wave(WavePeriod,H);
	WaveNumber=wave.wavenum();	  

        std::cout << "wave period is:"<< WavePeriod <<"\n";
	std::cout << "Wave Number k is:"<< WaveNumber <<"\n";

	// paddle stroke length
	const float KH = WaveNumber*H;
        stroke 	       = WaveHeight / (4.0*(sinh(KH)/KH)*(KH*sinh(KH)-cosh(KH)+1.0)/(sinh(2.0*KH)+2.0*KH));
        std::cout << "stroke value is:"<< stroke <<"\n";
        std::cout << "wave height is:"<< WaveHeight <<"\n";

	// SPH parameters
	simparams()->dt = 0.0001;
	simparams()->sfactor= 1.5;
	simparams()->dtadaptfactor = 0.45;
	simparams()->buildneibsfreq = 10;
	//simparams()->tend = 50*WavePeriod; //seconds

	// Physical parameters
	setWaterLevel(H);
	physparams()->gravity = make_float3(0.0f, 0.0f, -9.81f);
	float g = length(physparams()->gravity);
	float r0 = m_deltap;
	physparams()->r0 = r0;
	
	//const float maxvel = sqrt(2*g*H);
	const float c0 = SS;  //numerical sound speed
	add_fluid(1000.0f);
	set_equation_of_state(0,  1.0f, c0);
	//double bkg_press = c0*c0*1000/7.0; // background pressure
	//physparams()->set_bkgpressure(bkg_press);
	set_kinematic_visc(0,nu);
		
	physparams()->artvisccoeff =  0.025;
	physparams()->smagfactor = Cd*Cd*m_deltap*m_deltap; //CSM: wa have C=0.12 as a standard value
	physparams()->kspsfactor = (2.0/3.0)*0.0066*m_deltap*m_deltap; //CI = 2/3*6.6*10^-3*dp^2
	physparams()->epsartvisc = 0.01*simparams()->slength*simparams()->slength;
	//physparams()->spsclosure = 2U; // 1U Constant Smagorinsky model, 2U WALE model added by Morteza Derakhti derakhti@jhu.edu
	
	// BC when using LJ
	physparams()->dcoeff = 0.5*g*H;
	//set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5

	// BC when using MK
	physparams()->MK_K = g*H;
	physparams()->MK_d = 1.1*m_deltap/MK_par;
	physparams()->MK_beta = MK_par;

	//Wave paddle definition:  location, start & stop times, stroke and frequency (2 \pi/period)
        int m_mbnumber = 1;
        std::cout << "the number of paddles is:	"<< m_mbnumber <<"\n";	
	paddle_width= ly/m_mbnumber;	
        std::cout << "the width of paddles is:	"<< paddle_width <<"\n";	
	paddle_length = 2.0-r0;
	paddle_tstart = 0.12;
	paddle_tend = simparams()->tend;//seconds	
	paddle_origin = make_double3(2.0, r0/2, 0.0f);

	paddle_amplitude = atan(stroke/(2.0*(H + 19.5)));
	paddle_omega = 2.0*M_PI/WavePeriod;		// the angular frequency of the paddle motion
	cout << "\npaddle_amplitude (radians): " << paddle_amplitude << "\n";
	cout << "\npaddle_omega: " << paddle_omega << "\n";

	add_gage (3, ly/2.0);
	add_gage (7, ly/2.0);
	add_gage (10, ly/2.0);
	add_gage (13, ly/2.0);

	// Drawing and saving times
	add_writer(VTKWRITER, 1.2f);  //second argument is saving time in seconds
	add_writer(COMMONWRITER, 0.01);
	//add_writer(TEXTWRITER,WavePeriod/25.0f);
	// Name of problem used for directory creation
	m_name = "AiryWaves";

	time_t  rawtime;
	char	time_str[17];
	time(&rawtime);
	strftime(time_str, 17, "%Y-%m-%dT%Hh%M", localtime(&rawtime));
	time_str[16] = '\0';
	std::cout << "The starting time for the case run: " << string(time_str) << "\n";
	
	// Building the geometry
	const float br = (simparams()->boundarytype == MK_BOUNDARY ? m_deltap/MK_par : r0);
	setPositioning(PP_CORNER);

  	const float amplitude = 0.0f; //- paddle_amplitude; // this is the initial tilt of the paddles
	for (uint i=0; i<m_mbnumber; i++) {
		//pad_origin[i] = make_double3(2.0,i*paddle_width, 0.0f);	
		GeometryID paddle = addBox(GT_MOVING_BODY, FT_BORDER,
		Point(make_double3(2.0 - 0*m_deltap,i*paddle_width+r0/2, 0.0f)), 0*m_deltap, paddle_width-r0, paddle_length);
		//rotate(paddle, 0,-amplitude, 0);
		disableCollisions(paddle);	
	}  	

	
	if (!use_bottom_plane) {
		GeometryID bottom = addBox(GT_FIXED_BOUNDARY, FT_BORDER,
				Point(horizontal_flat, 0, 0), 0, ly, paddle_length);
		//	Vector(slope_length/cos(beta), 0.0, slope_length*tan(beta)));
		disableCollisions(bottom);
	}

	// It has been assumed that all padles are at the same initial titlting degrees
	//GeometryID fluid;
	float z = 0;
	int n = 0;
	while (z < H) {
	      z = n*m_deltap + 1.0*r0 + 0.0001*n; 
	      float x = paddle_origin.x + (z - paddle_origin.z)*tan(amplitude) + 1.0*r0/cos(amplitude);
	      float l = horizontal_flat + z/tan(beta) - (1.0*r0+0.0001)/sin(beta) - x;	      
	      //fluid = addRect(GT_FLUID, FT_SOLID, Point(x,  r0/2, z),  //r0 if use sidewall
	      addRect(GT_FLUID, FT_SOLID, Point(x,  r0/2, z),  //r0 if use sidewall
	      l, ly-1.0*r0);  //-2.0*r0 if use sidewall
	      ++n;
	}
}


void
AiryWaves2D::moving_bodies_callback(const uint index, Object* object, const double t0, const double t1,
			const float3& force, const float3& torque, const KinematicData& initial_kdata,
			KinematicData& kdata, double3& dx, EulerParameters& dr)
{

	dx= make_double3(0.0);
	kdata.lvel=make_double3(0.0f, 0.0f, 0.0f);
	kdata.crot = make_double3(2.0,index*paddle_width, -19.5f);
	float arg;
	float dthetadt = 0.;
	//if (t1 >= paddle_tstart && t1 < paddle_tend) {	
	if (t1 >= paddle_tstart) {	
		arg = paddle_omega*(t1- paddle_tstart);
		dthetadt = paddle_amplitude*paddle_omega*cos(arg);
		kdata.avel = make_double3(0.0, dthetadt, 0.0);		
       		EulerParameters dqdt = 0.5*EulerParameters(kdata.avel)*kdata.orientation;
       		dr = EulerParameters::Identity() + (t1-t0)*dqdt*kdata.orientation.Inverse();
       		dr.Normalize();
		kdata.orientation = kdata.orientation + (t1 - t0)*dqdt;
	   	kdata.orientation.Normalize();			
		}
	else {
	        kdata.avel = make_double3(0.0,0.0,0.0);
	   	kdata.orientation = kdata.orientation;
	   	dr.Identity();
	}	
}

void AiryWaves2D::copy_planes(PlaneList &planes)
{
	const double l = horizontal_flat + slope_length;

	//  plane is defined as a x + by +c z + d= 0
	planes.push_back( implicit_plane(0, 0, 1.0, 0) );   //bottom, where the first three numbers are the normal, and the last is d.
	//planes.push_back( implicit_plane(0, 1.0, 0, 0) );   //wall
	//planes.push_back( implicit_plane(0, -1.0, 0, w) ); //far wall
	planes.push_back( implicit_plane(1.0, 0, 0, 0) );  //end
	planes.push_back( implicit_plane(-1.0, 0, 0, l) );  //one end
	if (use_bottom_plane)  {
		planes.push_back( implicit_plane(-sin(beta),0,cos(beta), horizontal_flat*sin(beta)) );  //sloping bottom starting at x=horizontal_flat
	}
}


void AiryWaves2D::fillDeviceMap()
{
	fillDeviceMapByAxisBalanced(X_AXIS);
}

// Mass and density initialization
void
AiryWaves2D::initializeParticles(BufferList &buffers, const uint numParticles)
{

	// 1. warn the user if this is expected to take much time
	printf("Initializing particles density and mass...\n");

	// 2. grab the particle arrays from the buffer list
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();
	double4 *pos_global = buffers.getData<BUFFER_POS_GLOBAL>();
	float4 *pos = buffers.getData<BUFFER_POS>();

	// 3. iterate on the particles
	for (uint i = 0; i < numParticles; i++) {
		double depth = H - pos_global[i].z + m_origin.z;
		float rho = hydrostatic_density(depth, 0);
		pos[i].w = m_deltap*m_deltap*m_deltap*physical_density(rho, 0);
		vel[i].w = rho;
	}
}

bool AiryWaves2D::need_write(double time) const
{
	return false;
}
#undef MK_par
