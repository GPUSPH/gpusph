#include "RepackAlgo.h"
#include "GPUSPH.h"
#include "GlobalData.h"
#include "HDF5SphReader.h"
#include "HDF5SphWriter.h"
#include "H5Fpublic.h"

/**
  Constructor
*/
RepackAlgo::RepackAlgo()
: max_c0( 0 )
{
}

/**
  Destructor
*/
RepackAlgo::~RepackAlgo()
{
}

/**
  Initialization
  @param _gpusph a pointer to GPUSPH solver
  @param _gdata a pointer to global data of solver
  @param _problem a pointer to current problem
*/
void RepackAlgo::Init( GPUSPH* _gpusph, GlobalData* _gdata, Problem* _problem )
{
	gdata = _gdata;
	if( gdata->mode!=REPACK )
		return;

	printf( "Repacking algorithm initialization\n" );

	gpusph = _gpusph;
	problem = _problem;

	std::string data_file_name = GetFileName(true);
	std::string conf_file_name = GetFileName(false);

	printf( "Repacking flags: %i\n", gdata->repack_flags );

	bool isForced = ( (gdata->repack_flags & REPACK_FORCED) != 0 );

	bool canReuseData = !isForced && HdfFileExists( data_file_name );
	if( canReuseData )
	{
		printf( "HDF file exists\n" );

		canReuseData = ParametersEqual( conf_file_name );
		if( canReuseData )
			printf( "Parameters are equal\n" );
		else
			printf( "Parameters are different\n" );

		if( canReuseData )
		{
			canReuseData = Problem::getMTime(data_file_name.c_str()) > problem->getInputMTime();
			if( canReuseData )
				printf( "Packed data file is younger than input\n" );
			else
				printf( "Packed data file is older than input\n" );
		}
	}

	if ( canReuseData )
	{
		gdata->repack_flags = (gdata->repack_flags | REPACK_REUSE);
		printf( "---\n");
		printf( "Particles repacking results will be reused\n");
		printf( "repack_a=%e\n", gdata->repack_a );
		printf( "repack_alpha=%e\n", gdata->repack_alpha );

		printf( "---\n");

	}
}

/**
  Calculate norm of vector
  @param v a vector
*/
float norm2( float4 v )
{
  return v.x*v.x + v.y*v.y + v.z*v.z;
}

/**
  Get maximal "numerical" speed of sound
  @return maximal "numerical" speed of sound
*/
float RepackAlgo::maxC0() const
{
	return max_c0;
}

/**
  Get total kinetic energy for all particles
  @return total kinetic energy
*/
float RepackAlgo::TotalKE() const
{
	float ke = 0;
	particleinfo *info = gdata->s_hBuffers.getData<BUFFER_INFO>();
	const float4 *vel = gdata->s_hBuffers.getData<BUFFER_VEL>();
	int n = gdata->totParticles;
	for( int i=0; i<n; i++ )
	{
		if( FLUID( info[i] ) )
		{
			float v2 = norm2( vel[i] );
			ke += v2/2;
		}
	}
	//printf( "repack_a = %f, repack_al = %f, max_c0 = %f\n", gdata->repack_a, gdata->repack_alpha, maxC0() );
	ke = ke / gdata->repack_a / ( maxC0() * maxC0() );
	return ke;
}

/**
  Set repacking algorithm parameters.
*/
bool RepackAlgo::SetParams()
{
	if( gdata->mode!=REPACK )
		return true;

	bool canReuseData = ( (gdata->repack_flags & REPACK_REUSE) != 0 );

	if( !canReuseData )
	{
		printf( "The repacking data will be re-written\n" );
		// Initialize repacking parameters if old results can not be reused.
		const double dr = gdata->problem->get_deltap();
		max_c0 = NAN;

		for (uint f = 0; f < gdata->problem->physparams()->numFluids(); ++f)
			max_c0 = fmaxf(max_c0,gdata->problem->physparams()->sscoeff[f]);

		// Fill in constants
		//gdata->repack_a = 1.f;
		//gdata->repack_alpha  = 0.1f;

		//gdata->dt   = CFL * dr / (sqrt(repack_a)*c0);
		// Keep original dt
		dt = gdata->dt;
		gdata->dt   = 1.*dr/max_c0;

		problem->simparams()->repack_a = gdata->repack_a;
		problem->simparams()->repack_alpha = gdata->repack_alpha;

		printf( "---\n");
		printf( "Particle repacking is enabled\n");
		printf( "repack_a=%e\n", gdata->repack_a );
		printf( "repack_alpha=%e\n", gdata->repack_alpha );
		printf( "dt=%e\n", gdata->dt );
		printf( "dr=%e\n", dr );

		printf( "---\n");
	}
}

/**
  Check if the repacking algorithm can be omitted
  (i.e. the already prepared repacking data can be re-used)
  @return true if the repacking data can be re-used
*/
bool RepackAlgo::Start()
{
	if( gdata->mode!=REPACK )
		return true;

	printf( "REPACKING ALGORITHM:\n" );
	bool canReuseData = ( (gdata->repack_flags & REPACK_REUSE) != 0 );

	if( canReuseData )
	{
		printf( "The repacking data will be reused\n" );
		std::string data_file_name = GetFileName(true);
		LoadData( data_file_name );
		gdata->mode = STANDARD; // nothing to do with repacking
		return true;
	}
	return false;
}

/**
  The final stage of the repacking algorithm,
  the repacking data and parameters are stored in the files
*/
void RepackAlgo::Stop()
{
	if( gdata->mode != REPACK )
		return;
 
	std::string data_file_name = GetFileName(true);
	std::string conf_file_name = GetFileName(false);

	// Save fluid particles
	SaveData( data_file_name );
	// Save repacking parameters
	SaveParams( conf_file_name );
	// Restore standard mode parameters
	gdata->dt = dt;
	gdata->t = 0;
	gdata->mode = STANDARD;
}

/**
  Get file name for repacking data for the current problem
  @param isHdf true means HDF file (for repacking data), false means text file (for repacking parameters)
  @return the generated file name
*/
std::string RepackAlgo::GetFileName( bool isHdf ) const
{
	const std::string DIR = "./data_files/repacked/";
	std::string aFileName = problem->m_name + ( isHdf ? ".h5" : ".conf" );

	return DIR + aFileName;
}

/**
  Check if HDF file exists
  @param theFileName the name of HDF file
  @return if HDF file exists
*/
bool RepackAlgo::HdfFileExists( const std::string& theFileName ) const
{
	return (H5Fis_hdf5( theFileName.c_str() ) > 0);
}

/**
  Check if repacking parameters corresponds to file data
  @param theFileName the name of text file with stored repacking parameters 
  @return if parameters are equal to current ones
*/
bool RepackAlgo::ParametersEqual( const std::string& theFileName ) const
{
	printf( "Loading parameters file: %s\n", theFileName.c_str() );
	FILE* f = fopen( theFileName.c_str(), "r" );
	if( !f )
		return false;

	char b1[128], b2[128];
	float repack_a, repack_alpha;
	int n1 = fscanf( f, "%s %s %f", b1, b2, &repack_a );
	int n2 = fscanf( f, "%s %s %f", b1, b2, &repack_alpha );
	fclose( f );
	
	printf( "Conf a = %f\n", repack_a );
	printf( "Conf alpha = %f\n", repack_alpha );

	const float EPS = 1E-6;
	return fabs( repack_a - gdata->repack_a ) < EPS &&
				 fabs( repack_alpha - gdata->repack_alpha ) < EPS;
}

/**
  Load repacking data from HDF file
  @param theFileName the name of HDF file with repacking data
*/
void RepackAlgo::LoadData( const std::string& theFileName ) const
{
	printf( "Loading repacking data: %s\n", theFileName.c_str() );
	HDF5SphReader reader;
	reader.setFilename( theFileName );
	reader.read();
	
	ReadParticles* part = reader.buf;

	float4 *coords     = gdata->s_hBuffers.getData<BUFFER_POS>();
	particleinfo *info = gdata->s_hBuffers.getData<BUFFER_INFO>();
	
	int n = gdata->totParticles;
	for( int i=0, j=0; i<n; i++ )
	{
		if( FLUID( info[i] ) )
		{
			coords[i].x = part[j].Coords_0;
			coords[i].y = part[j].Coords_1;
			coords[i].z = part[j].Coords_2;
			j++;
		}
	}
}

/**
  Save repacking data to HDF file
  @param theFileName the name of HDF file with repacking data
*/
void RepackAlgo::SaveData( const std::string& theFileName ) const
{
	remove( theFileName.c_str() );
	printf( "Saving repacking data: %s\n", theFileName.c_str() );
	HDF5SphWriter writer( gdata );
	writer.write( theFileName.c_str() );
}

/**
  Save repacking parameters to text file
  @param theFileName the name of text file with repacking parameters
*/
void RepackAlgo::SaveParams( const std::string& theFileName ) const
{
	FILE* f = fopen( theFileName.c_str(), "w" );
	if( f )
	{
		fprintf( f, "repack_a = %f\n", gdata->repack_a );
		fprintf( f, "repack_alpha = %f\n", gdata->repack_alpha );
		fclose( f );
	}
}
