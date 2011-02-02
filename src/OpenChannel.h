#ifndef _OPENCHANNEL_H
#define	_OPENCHANNEL_H

#include "Problem.h"
#include "Point.h"
#include "Rect.h"
#include "Cube.h"

class OpenChannel: public Problem {
	private:
		Rect		rect1, rect2, rect3;
		Cube		experiment_box;
		PointVect	parts;
		PointVect	boundary_parts;
		float		a, h, l;  // experiment box dimension
		float		H; // still water level

	public:
		OpenChannel(const Options &);
		~OpenChannel(void);

		int fill_parts(void);
		void draw_boundary(float);
		void copy_to_array(float4 *, float4 *, particleinfo *);

		void release_memory(void);
};


#endif	/* _POWERLAW_H */
