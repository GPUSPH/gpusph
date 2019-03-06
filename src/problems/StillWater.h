#ifndef _STILLWATER_H
#define	_STILLWATER_H

#include "XProblem.h"

class StillWater: public XProblem {
	private:
		double		h, w, l;
		double		H; // still water level
		uint		dyn_layers; // layers of dynamic boundaries particles
		bool		m_usePlanes; // use planes or boundaries
		double3		m_fluidOrigin; // bottom level

	public:
		StillWater(GlobalData *);
		void copy_planes(PlaneList &);

};


#endif	/* _STILLWATER_H */
