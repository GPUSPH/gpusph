#ifndef _READER_H
#define _READER_H

#include <string>
#include <iostream>

#define CRIXUS_FLUID 1
#define CRIXUS_VERTEX 2
#define CRIXUS_BOUNDARY 3

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

class Reader
{
protected:
	std::string		filename;
	unsigned int	npart;
public:
	Reader(void);
	~Reader(void);

	// returns the number of particles in the input file
	virtual int getNParts(void) = 0;

	// allocates the buffer and reads the data from the input file
	virtual void read(void) = 0;

	// frees the buffer
	void empty(void);

	// free the buffer, reset npart and filename
	void reset();

	// sets the filename
	void setFilename(std::string const&);

	ReadParticles *buf;
};

#endif	/* _READER_H */
