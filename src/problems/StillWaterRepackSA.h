#ifndef _STILLWATERREPACKSA_H
#define	_STILLWATERREPACKSA_H

#include <string>

#define PROBLEM_API 1
#include "Problem.h"
#include "HDF5SphReader.h"

class StillWaterRepackSA: public Problem {
	public:
		StillWaterRepackSA(GlobalData *);

		void fillDeviceMap();
};

#endif	/* _STILLWATERREPACKSA_H */
