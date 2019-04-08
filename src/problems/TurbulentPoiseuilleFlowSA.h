

#ifndef _TURBULENTPOISEUILLEFLOWSA_H
#define	_TURBULENTPOISEUILLEFLOWSA_H

#include "XProblem.h"

class TurbulentPoiseuilleFlowSA: public XProblem {
	private:
		uint U;
	public:
		TurbulentPoiseuilleFlowSA(GlobalData *);
		virtual void initializeParticles(BufferList &buffers, const uint numParticles);
};
#endif	/* _TURBULENTPOISEUILLEFLOWSA_H */


