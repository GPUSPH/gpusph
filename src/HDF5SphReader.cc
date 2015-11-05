/*
  This file has been extracted from "Sphynx" code.
  Originally developed by Arno Mayrhofer (?), Christophe Kassiotis (?), Martin Ferrand (?).
  It contains a class for reading *.h5sph files - input files in hdf5 format.
*/

#if USE_HDF5
#include <hdf5.h>
#else
#include <stdexcept>
#define NO_HDF5_ERR throw std::runtime_error("HDF5 support not compiled in")
#endif

#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <limits.h> // UINT_MAX

#include "hdf5_select.opt"

#include <stdexcept>

#include "HDF5SphReader.h"

// Name of dataset_id to create in loc_id
#define DATASETNAME "Compound"

// Dataset dimensions
#define RANK 1

int
HDF5SphReader::getNParts()
{
#if USE_HDF5
	if (npart != UINT_MAX)
		return npart;
	hid_t		loc_id, dataset_id, file_space_id;
	hsize_t		*dims;
	int		ndim;

	loc_id = H5Fopen(filename.c_str(),H5F_ACC_RDONLY, H5P_DEFAULT);
	dataset_id = H5Dopen2(loc_id, DATASETNAME, H5P_DEFAULT);
	file_space_id = H5Dget_space(dataset_id);

	ndim = H5Sget_simple_extent_ndims(file_space_id);
	dims = new hsize_t[ndim]; //(hsize_t)malloc(ndim*sizeof(hsize_t));
	ndim = H5Sget_simple_extent_dims(file_space_id, dims, NULL);
	npart = dims[0];

	H5Sclose(file_space_id);
	H5Dclose(dataset_id);
	H5Fclose(loc_id);

	return npart;
#else
	return 0;
#endif
}

void
HDF5SphReader::read()
{
#if USE_HDF5
	// read npart if it was yet uninitialized
	if (npart == UINT_MAX)
		getNParts();
	std::cout << "Reading particle data from the input: " << filename << std::endl;
	if(buf == NULL)
		buf = new ReadParticles[npart];
	else{
		delete [] buf;
		buf = new ReadParticles[npart];
	}
	hid_t		mem_type_id, loc_id, dataset_id, file_space_id, mem_space_id;
	hsize_t		count[RANK], offset[RANK];
	herr_t		status;

	loc_id = H5Fopen(filename.c_str(),H5F_ACC_RDONLY, H5P_DEFAULT);
	dataset_id = H5Dopen2(loc_id, DATASETNAME, H5P_DEFAULT);

	// Create the memory data type
	mem_type_id = H5Tcreate (H5T_COMPOUND, sizeof(ReadParticles));
	H5Tinsert(mem_type_id, "Coords_0"       , HOFFSET(ReadParticles, Coords_0),        H5T_NATIVE_DOUBLE);
	H5Tinsert(mem_type_id, "Coords_1"       , HOFFSET(ReadParticles, Coords_1),        H5T_NATIVE_DOUBLE);
	H5Tinsert(mem_type_id, "Coords_2"       , HOFFSET(ReadParticles, Coords_2),        H5T_NATIVE_DOUBLE);
	H5Tinsert(mem_type_id, "Normal_0"       , HOFFSET(ReadParticles, Normal_0),        H5T_NATIVE_DOUBLE);
	H5Tinsert(mem_type_id, "Normal_1"       , HOFFSET(ReadParticles, Normal_1),        H5T_NATIVE_DOUBLE);
	H5Tinsert(mem_type_id, "Normal_2"       , HOFFSET(ReadParticles, Normal_2),        H5T_NATIVE_DOUBLE);
	H5Tinsert(mem_type_id, "Volume"         , HOFFSET(ReadParticles, Volume),          H5T_NATIVE_DOUBLE);
	H5Tinsert(mem_type_id, "Surface"        , HOFFSET(ReadParticles, Surface),         H5T_NATIVE_DOUBLE);
	H5Tinsert(mem_type_id, "ParticleType"   , HOFFSET(ReadParticles, ParticleType),    H5T_NATIVE_INT);
	H5Tinsert(mem_type_id, "FluidType"      , HOFFSET(ReadParticles, FluidType),       H5T_NATIVE_INT);
	H5Tinsert(mem_type_id, "KENT"           , HOFFSET(ReadParticles, KENT),            H5T_NATIVE_INT);
	H5Tinsert(mem_type_id, "MovingBoundary" , HOFFSET(ReadParticles, MovingBoundary),  H5T_NATIVE_INT);
	H5Tinsert(mem_type_id, "AbsoluteIndex"  , HOFFSET(ReadParticles, AbsoluteIndex),   H5T_NATIVE_INT);
	H5Tinsert(mem_type_id, "VertexParticle1", HOFFSET(ReadParticles, VertexParticle1), H5T_NATIVE_INT);
	H5Tinsert(mem_type_id, "VertexParticle2", HOFFSET(ReadParticles, VertexParticle2), H5T_NATIVE_INT);
	H5Tinsert(mem_type_id, "VertexParticle3", HOFFSET(ReadParticles, VertexParticle3), H5T_NATIVE_INT);

	//create a memory file_space_id independently
	count[0] = npart;
	offset[0] = 0;
	mem_space_id = H5Screate_simple (RANK, count, NULL);

	// set up dimensions of the slab this process accesses
	file_space_id = H5Dget_space(dataset_id);
	status = H5Sselect_hyperslab(file_space_id, H5S_SELECT_SET, offset, NULL, count, NULL);
	if (status < 0) {
		throw std::runtime_error("reading HDF5 hyperslab");
	}

	// read data independently
	status = H5Dread(dataset_id, mem_type_id, mem_space_id, file_space_id, H5P_DEFAULT, buf);
	if (status < 0) {
		throw std::runtime_error("reading HDF5 data");
	}

	H5Dclose(dataset_id);
	H5Sclose(file_space_id);
	H5Sclose(mem_space_id);
	H5Fclose(loc_id);
	H5Tclose(mem_type_id);
#else
	NO_HDF5_ERR;
#endif
}
