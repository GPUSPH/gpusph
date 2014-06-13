#ifndef _INPUTPROBLEM_H
#define	_INPUTPROBLEM_H

#include <string>

#include "Problem.h"

class InputProblem: public Problem {
	private:
		string		inputfile;
		int			numparticles;
		PointVect	test_points;
		double		w, l, h;
		double		H;				// water level (used to set D constant)

	public:
		InputProblem(const GlobalData *);
		~InputProblem(void) {};

		int fill_parts(void);
		void copy_to_array(BufferList &);
		void init_keps(float*, float*, uint, particleinfo*, float4*, hashKey*);

		void release_memory(void) {};
};

#endif	/* _INPUTPROBLEM_H */
