#ifndef _COMPLETESAEXAMPLE_H
#define	_COMPLETESAEXAMPLE_H

#include <string>

#include "Problem.h"
#include "HDF5SphReader.h"

// for loading meshes as ODE objs
#include "STLMesh.h"

class CompleteSaExample: public Problem {
	private:
		string			inputfile;
		PointVect		test_points;
		double			world_w, world_l, world_h;			// world size (i.e. incl. margins and inlet box)
		double			box_w, box_l, box_h;	// size of the main box (excl. margins, no inlet box)
		double			initial_water_level;			// used for initial hydrostatic filling
		double			expected_final_water_level;		// used to set D constant
		HDF5SphReader	h5File;
		STLMesh		*container;
		STLMesh		*cube;
		dGeomID		m_box_planes[5];	// planes to model the main tank

	public:
		CompleteSaExample(const GlobalData *);
		~CompleteSaExample(void);

		int fill_parts(void);
		void copy_to_array(BufferList &);
		void init_keps(float*, float*, uint, particleinfo*, float4*, hashKey*);
		uint max_parts(uint);

		void ODE_near_callback(void * data, dGeomID o1, dGeomID o2);

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
					float3*	gravityCenters,
					float3*	translations,
					float*	rotationMatrices,
			const	uint*	ODEobjectId,
			const	uint	numObjects,
			const	double	t,
			const	float	dt);

		void release_memory(void) {};

		void fillDeviceMap();

};

#endif	/* _COMPLETESAEXAMPLE_H */
