#ifndef _SPHERIC2SA_H
#define	_SPHERIC2SA_H

#include <string>

#define PROBLEM_API 1
#include "Problem.h"

class Spheric2SA: public Problem {
	private:
		double			w, l, h;
		double			H;				// water level (used to set D constant)

	public:
		Spheric2SA(GlobalData *);
		void initializeParticles(BufferList &buffers, const uint numParticles);
		
		uint max_parts(uint);
		void fillDeviceMap();
		bool need_write(double) const;
};

#endif	/* _SPHERIC2SA_H */
