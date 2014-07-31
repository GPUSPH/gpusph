#ifndef _INPUTPROBLEM_H
#define	_INPUTPROBLEM_H

#include <string>

#include "Problem.h"
#include "HDF5SphReader.h"

class InputProblem: public Problem {
	private:
		string			inputfile;
		PointVect		test_points;
		double			w, l, h;
		double			H;				// water level (used to set D constant)
		HDF5SphReader	h5File;


	public:
		InputProblem(const GlobalData *);
		~InputProblem(void) {};

		int fill_parts(void);
		void copy_to_array(BufferList &);
		void init_keps(float*, float*, uint, particleinfo*, float4*, hashKey*);
		uint max_parts(uint);

		void release_memory(void) {};
};

#endif	/* _INPUTPROBLEM_H */
