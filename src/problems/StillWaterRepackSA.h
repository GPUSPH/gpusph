#ifndef _STILLWATERREPACKSA_H
#define	_STILLWATERREPACKSA_H

#include <string>

#include "XProblem.h"
#include "HDF5SphReader.h"

class StillWaterRepackSA: public XProblem {
	public:
		StillWaterRepackSA(GlobalData *);

		void fillDeviceMap();
};

#endif	/* _STILLWATERREPACKSA_H */
