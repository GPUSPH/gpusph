#ifndef _INPUTPROBLEM_H
#define	_INPUTPROBLEM_H

#include <string>

#include "XProblem.h"
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
#define PeriodicWave				12	// Periodic wave with IO
#define	SmallChannelFlowIOPerOpen	13	// Small channel flow for debugging i/o with periodicty and gravity
#define SolitaryWave				14	// Solitary wave with IO
//*******************************************************************************************************
// Choose one of the problems above
#define SPECIFIC_PROBLEM BoxCorner

#define	PRESSURE_DRIVEN			0
#define	VELOCITY_DRIVEN			1

class InputProblem: public XProblem {
	private:
		double			w, l, h;
		double			H;				// water level (used to set D constant)

	public:
		InputProblem(GlobalData *);

		virtual void initializeParticles(BufferList &buffers, const uint numParticles);

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

		void fillDeviceMap();

};

#endif	/* _INPUTPROBLEM_H */
