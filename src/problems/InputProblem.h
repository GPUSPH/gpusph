#ifndef _INPUTPROBLEM_H
#define	_INPUTPROBLEM_H

#include <string>

#include "Problem.h"
#include "HDF5SphReader.h"
#include "VTUReader.h"

// Implemented problems:
//
//		Keyword					Id	// Description
//*******************************************************************************************************
#define	Box							3	// Small dambreak in a box
#define	BoxCorner					4	// Small dambreak in a box with a corner
#define	SmallChannelFlow			5	// Small channel flow for debugging
#define	SmallChannelFlowKEPS		6	// Small channel flow for debugging using k-epsilon
#define	SmallChannelFlowIO			7	// Small channel flow for debugging i/o
#define	SmallChannelFlowIOPer		8	// Small channel flow for debugging i/o with periodicty (<=> 2d poiseuille)
#define	SmallChannelFlowIOKeps		9	// Small channel flow for debugging i/o with periodicty and keps
#define	IOWithoutWalls				10	// I/O Debugging with periodicity and no walls
#define LaPalisseSmallTest			11	// Small testcase for La Palisse (pressure in/out with free-surface)
#define PeriodicWave				12	// Periodic wave with IO
#define	SmallChannelFlowIOPerOpen	13	// Small channel flow for debugging i/o with periodicty and gravity
#define SolitaryWave				14	// Solitary wave with IO
#define LaPalisseSmallerTest		15	// Smaller testcase for La Palisse (pressure in/out with free-surface)
//*******************************************************************************************************
// Choose one of the problems above
#define SPECIFIC_PROBLEM LaPalisseSmallerTest

class InputProblem: public Problem {
	private:
		string			inputfile;
		PointVect		test_points;
		double			w, l, h;
		double			H;				// water level (used to set D constant)
		HDF5SphReader	h5File;
		VTUReader		vtuFile;


	public:
		InputProblem(GlobalData *);
		~InputProblem(void) {};

		int fill_parts(void);
		void copy_to_array(BufferList &);
		void init_keps(float*, float*, uint, particleinfo*, float4*, hashKey*);
		uint max_parts(uint);

		void
		imposeBoundaryConditionHost(
			MultiBufferList::iterator		bufwrite,
			MultiBufferList::const_iterator	bufread,
					uint*			IOwaterdepth,
			const	float			t,
			const	uint			numParticles,
			const	uint			numOpenBoundaries,
			const	uint			particleRangeEnd);

		void release_memory(void) {};

		void fillDeviceMap();

};

#endif	/* _INPUTPROBLEM_H */
