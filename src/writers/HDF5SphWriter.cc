
#include <HDF5SphWriter.h>
#include <GlobalData.h>
#include <stdexcept>

#if USE_HDF5
#include <hdf5.h>
#else
#include <stdexcept>
#define NO_HDF5_ERR throw runtime_error("HDF5 support not compiled in")
#endif

HDF5SphWriter::HDF5SphWriter( GlobalData* _gdata )
	: gdata( _gdata )
{
}

HDF5SphWriter::~HDF5SphWriter()
{
}

ReadParticles* HDF5SphWriter::fillData( int& npart ) const
{
	const double4 *coords = gdata->s_hBuffers.getData<BUFFER_POS_GLOBAL>();
	const particleinfo *info = gdata->s_hBuffers.getData<BUFFER_INFO>();
	
	int n = gdata->totParticles;
	npart = 0;
	ReadParticles* part = new ReadParticles[n];
	
	const double dr = gdata->problem->get_deltap();
	const double volume = pow( dr, 3 ); 
	
	for( int i=0, j=0; i<n; i++ )
	{
		if( FLUID( info[i] ) )
		{
			part[j].Coords_0 = coords[i].x;
			part[j].Coords_1 = coords[i].y;
			part[j].Coords_2 = coords[i].z;
			part[j].Normal_0 = 0;
			part[j].Normal_1 = 0;
			part[j].Normal_2 = 0;
			part[j].Volume = volume;
			part[j].Surface = 0;
			part[j].ParticleType = CRIXUS_FLUID;
			part[j].FluidType = 0;
			part[j].KENT = 0;
			part[j].MovingBoundary = 0;
			part[j].AbsoluteIndex = j;
			part[j].VertexParticle1 = 0;
			part[j].VertexParticle2 = 0;
			part[j].VertexParticle3 = 0;
			
			j++;
			npart++;
		}
	}
	
	return part;
}
	
void HDF5SphWriter::write( const char* filename )
{
#if USE_HDF5
    int npart;
    ReadParticles* parts = fillData( npart );

	hid_t		mem_type_id, loc_id, dataset_id, mem_space_id;
	hsize_t		count[D_RANK], offset[D_RANK];
	herr_t		status;

	loc_id = H5Fopen( filename, H5F_ACC_CREAT | H5F_ACC_RDWR, H5P_DEFAULT );

	// Create the memory data type
	mem_type_id = createType();

	count[0] = npart;
	offset[0] = 0;
	mem_space_id = H5Screate_simple (D_RANK, count, NULL);
	
	hid_t aLinkProperties = H5Pcreate( H5P_LINK_CREATE );
	H5Pset_create_intermediate_group( aLinkProperties, 1 );
	
	//hid_t aCreateProperties = H5Pcreate( H5P_DATASET_CREATE );

	dataset_id = H5Dcreate( loc_id, DATASETNAME, mem_type_id, mem_space_id, aLinkProperties, H5P_DEFAULT, H5P_DEFAULT );
	if (dataset_id < 0) {
		throw std::runtime_error("HDF dataset is not created");
	}
	
	H5Pclose( aLinkProperties );
	//H5Pclose( aCreateProperties );
	
	// write data 
	status = H5Dwrite(dataset_id, mem_type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, parts );
	if (status < 0) {
		throw std::runtime_error("Writing HDF5 data failed");
	}

	H5Dclose(dataset_id);
	H5Sclose(mem_space_id);
	H5Tclose(mem_type_id);
	H5Fclose(loc_id);
	
	delete[] parts;
#else
	NO_HDF5_ERR;
#endif
}
