#ifndef _LAPALISSE_H
#define	_LAPALISSE_H

#include <string>

#include "Problem.h"
#include "HDF5SphReader.h"

// Water level simulated by the pressure inlet
#define INLET_WATER_LEVEL	1.34475f

class LaPalisse: public Problem {
	private:
		string			inputfile;
		PointVect		test_points;
		double			world_w, world_l, world_h;			// world size (i.e. incl. margins and inlet box)
		double			box_w, box_l, box_h;	// size of the main box (excl. margins, no inlet box)
		double			initial_water_level;			// used for initial hydrostatic filling
		double			expected_final_water_level;		// used to set D constant
		HDF5SphReader	h5File;

	public:
		LaPalisse(GlobalData *);
		~LaPalisse(void);

		int fill_parts(void);
		void copy_to_array(BufferList &);
		void init_keps(float*, float*, uint, particleinfo*, float4*, hashKey*);
		uint max_parts(uint);

		void
		setboundconstants(
			const	PhysParams	*physparams,
			float3	const&		worldOrigin,
			uint3	const&		gridSize,
			float3	const&		cellSize);

		void
		imposeBoundaryConditionHost(
					float4*			newVel,
					float4*			newEulerVel,
					float*			newTke,
					float*			newEpsilon,
			const	particleinfo*	info,
			const	float4*			oldPos,
					uint*			IOwaterdepth,
			const	float			t,
			const	uint			numParticles,
			const	uint			numObjects,
			const	uint			particleRangeEnd,
			const	hashKey*		particleHash);

		void imposeForcedMovingObjects(
					float3	&gravityCenters,
					float3	&translations,
					float*	rotationMatrices,
			const	uint	ob,
			const	double	t,
			const	float	dt);

		void release_memory(void) {};

		void fillDeviceMap();

};

#endif	/* _LAPALISSE_H */
