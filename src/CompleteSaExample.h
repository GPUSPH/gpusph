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
		double			w, l, h;
		double			H;				// water level (used to set D constant)
		HDF5SphReader	h5File;
		STLMesh		*container;
		STLMesh		*cube;

	public:
		CompleteSaExample(const GlobalData *);
		~CompleteSaExample(void) {};

		int fill_parts(void);
		void copy_to_array(BufferList &);
		void init_keps(float*, float*, uint, particleinfo*, float4*, hashKey*);
		uint max_parts(uint);

		void
		setioboundconstants(
			const	PhysParams	*physparams,
			float3	const&		worldOrigin,
			uint3	const&		gridSize,
			float3	const&		cellSize);

		void
		imposeOpenBoundaryConditionHost(
					float4*			newEulerVel,
					float*			newTke,
					float*			newEpsilon,
			const	particleinfo*	info,
			const	float4*			oldPos,
					uint*			IOwaterdepth,
			const	uint			numParticles,
			const	uint			numObjects,
			const	uint			particleRangeEnd,
			const	hashKey*		particleHash);

		void release_memory(void) {};

		void fillDeviceMap();

};

#endif	/* _COMPLETESAEXAMPLE_H */
