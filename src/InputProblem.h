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
		InputProblem(const Options &);
		~InputProblem(void) {};

		int fill_parts(void);
		void draw_boundary(float) {};
		void copy_to_array(float4 *, float4 *, particleinfo *, uint *hash) {};
		void copy_to_array(float4 *, float4 *, particleinfo *, vertexinfo *, float4 *, uint *hash);

		void release_memory(void) {};
};

#endif	/* _INPUTPROBLEM_H */
