#ifndef POISEUILLE_H_
#define POISEUILLE_H_

#include "XProblem.h"

#ifndef POISEUILLE_PROBLEM
#define POISEUILLE_PROBLEM Poiseuille
#endif

class POISEUILLE_PROBLEM: public XProblem {

	private:
		// box dimensions. The origin is assumed to be in the middle,
		// so that the domains extents are [-lx/2, lx/2], [-ly/2, ly/2], [-lz/2, lz/2]
		const double lz, ly, lx; // declared in reverse order because we initialize lz first

		// fluid density: defaults at 1, can be set to something different
		// to ensure the correctness of the use of kinematic vs dynamic viscosity
		const float rho;
		// kinematic viscosity of the fluid
		const float kinvisc;
		// driving force magnitude: can be changed to check the behavior for
		// different Reynolds numbers
		const float driving_force;

		// maximum theoretical flow velocity (computed from other flow parameters)
		const float max_vel;
		// Reynolds number (computed from other flow parameters)
		const float Re;

	public:
		POISEUILLE_PROBLEM(GlobalData *);
		void initializeParticles(BufferList &, const uint);
		float compute_poiseuille_vel(float);
};
#endif

