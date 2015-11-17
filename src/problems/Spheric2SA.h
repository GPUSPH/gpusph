#ifndef _SPHERIC2SA_H
#define	_SPHERIC2SA_H

#include <string>

#include "Problem.h"
#include "HDF5SphReader.h"

class Spheric2SA: public Problem {
	private:
		string			inputfile;
		PointVect		test_points;
		double			w, l, h;
		double			H;				// water level (used to set D constant)
		HDF5SphReader	h5File;


	public:
		Spheric2SA(GlobalData *);
		~Spheric2SA(void) {};

		int fill_parts(void);
		void copy_to_array(BufferList &);
		void init_keps(float*, float*, uint, particleinfo*, float4*, hashKey*);
		uint max_parts(uint);

		void release_memory(void) {};

		void fillDeviceMap();

};

#endif	/* _SPHERIC2SA_H */
