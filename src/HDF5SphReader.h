/*
  This file has been extracted from "Sphynx" code.
  Originally developed by Arno Mayrhofer (2013), Christophe Kassiotis (2013), Martin Ferrand (2013).
  It contains a class for reading *.h5sph files - input files in hdf5 format.
*/
#include <string>
#include <iostream>

class HDF5SphReader {
private:
	std::string	filename;
	unsigned int	npart;

	struct ReadParticles {
		double Coords_0;
		double Coords_1;
		double Coords_2;
		double Normal_0;
		double Normal_1;
		double Normal_2;
		double Volume;
		double Surface;
		int ParticleType;
		int FluidType;
		int KENT;
		int MovingBoundary;
		int AbsoluteIndex;
		int VertexParticle1;
		int VertexParticle2;
		int VertexParticle3;
	};

public:
	// constructor
	HDF5SphReader(void);

	// returns the number of particles in the h5sph file
	int getNParts(void);

	// allocates the buffer and reads the data from the h5sph file
	void read(void);

	// frees the buffer
	void empty(void);

	// sets the filename
	void setFilename(std::string const&);

	ReadParticles *buf;
};
