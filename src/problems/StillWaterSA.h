


#ifndef _STILLWATERSA_H
#define	_STILLWATERSA_H

#define PROBLEM_API 1
#include "Problem.h"


class StillWaterSA: public Problem {
	private:
		uint U;
	public:
		StillWaterSA(GlobalData *);
		void fillDeviceMap();
};
#endif	/* _STILLWATERSA_H */

