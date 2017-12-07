#ifndef _SPHERIC2SA_H
#define	_SPHERIC2SA_H

#include <string>

#include "XProblem.h"
#include "HDF5SphReader.h"

class Spheric2SA: public XProblem {
	private:
		double			w, l, h;
		double			H;				// water level (used to set D constant)

	public:
		Spheric2SA(GlobalData *);
		virtual void initializeParticles(BufferList &buffers, const uint numParticles);
		
		uint max_parts(uint);
		void fillDeviceMap();

};

#endif	/* _SPHERIC2SA_H */
