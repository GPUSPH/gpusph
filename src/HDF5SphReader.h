/*
  This file has been extracted from "Sphynx" code.
  Originally developed by Arno Mayrhofer (?), Christophe Kassiotis (?), Martin Ferrand (?).
  It contains a class for reading *.h5sph files - input files in hdf5 format.
*/

class HDF5SphReader {
public:
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

	static int getNParts(const char *filename);

	static void readParticles(ReadParticles *buf, const char *filename, int num);
};
