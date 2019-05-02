

#ifndef _TURBULENTPOISEUILLEFLOWSA_H
#define	_TURBULENTPOISEUILLEFLOWSA_H

#define PROBLEM_API 1
#include "Problem.h"

class TurbulentPoiseuilleFlowSA: public Problem {
	private:
		uint U;
	public:
		TurbulentPoiseuilleFlowSA(GlobalData *);
		virtual void initializeParticles(BufferList &buffers, const uint numParticles);
};
#endif	/* _TURBULENTPOISEUILLEFLOWSA_H */


