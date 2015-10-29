#ifndef _STILLWATER_H
#define	_STILLWATER_H

#include "Problem.h"
#include "Point.h"
#include "Rect.h"
#include "Cube.h"

class StillWater: public Problem {
	private:
		Cube		experiment_box;
		PointVect	parts;
		PointVect	boundary_parts;
		PointVect	boundary_elems;
		PointVect	vertex_parts;
		VertexVect	vertex_indexes;

		double		h, w, l;
		double		H; // still water level
		uint		dyn_layers; // layers of dynamic boundaries particles
		bool		m_usePlanes; // use planes or boundaries
		double3		m_fluidOrigin; // bottom level

	public:
		StillWater(GlobalData *);
		~StillWater(void);

		int fill_parts(void);
		void copy_to_array(BufferList &);
		void copy_planes(PlaneList &);

		void release_memory(void);
};


#endif	/* _STILLWATER_H */
