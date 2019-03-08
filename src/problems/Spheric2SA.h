#ifndef _SPHERIC2SA_H
#define	_SPHERIC2SA_H

#include <string>

#include "XProblem.h"

class Spheric2SA: public XProblem {
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
