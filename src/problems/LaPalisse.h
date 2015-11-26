#ifndef _LAPALISSE_H
#define	_LAPALISSE_H

#include <string>

#include "Problem.h"
#include "HDF5SphReader.h"

// Water level simulated by the pressure inlet
#define INLET_WATER_LEVEL	1.34475f
// Water level after pre-processing fill
#define INITIAL_WATER_LEVEL	0.145f
// Time [s] over which the water should rise from INITIAL to
// INLET_WATER_LEVEL
#define RISE_TIME	6.0f

class LaPalisse: public Problem {
	private:
		string			inputfile;
		PointVect		test_points;
		double			w, l, h;
		double			H;				// water level (used to set D constant)
		HDF5SphReader	h5File;


	public:
		LaPalisse(GlobalData *);
		~LaPalisse(void) {};

		int fill_parts(void);
		void copy_to_array(BufferList &);
		void init_keps(float*, float*, uint, particleinfo*, float4*, hashKey*);
		uint max_parts(uint);

		void
		imposeBoundaryConditionHost(
			MultiBufferList::iterator		bufwrite,
			MultiBufferList::const_iterator	bufread,
					uint*			IOwaterdepth,
			const	float			t,
			const	uint			numParticles,
			const	uint			numOpenBoundaries,
			const	uint			particleRangeEnd);

		void release_memory(void) {};

		void fillDeviceMap();

};

#endif	/* _LAPALISSE_H */
