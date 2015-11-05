/*
  This file has been extracted from "Sphynx" code.
  Originally developed by Arno Mayrhofer (2013), Christophe Kassiotis (2013), Martin Ferrand (2013).
  It contains a class for reading *.h5sph files - input files in hdf5 format.
*/

#ifndef _HDF5SPHREADER_H
#define _HDF5SPHREADER_H

#include <string>
#include <iostream>

#include "Reader.h"

class HDF5SphReader : public Reader{
public:
	// returns the number of particles in the h5sph file
	int getNParts(void);

	// allocates the buffer and reads the data from the h5sph file
	void read(void);
};

#endif
