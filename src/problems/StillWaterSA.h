


#ifndef _STILLWATERSA_H
#define	_STILLWATERSA_H

#include "XProblem.h"


class StillWaterSA: public XProblem {
	private:
		uint U;
	public:
		StillWaterSA(GlobalData *);
		void fillDeviceMap();
};
#endif	/* _STILLWATERSA_H */

