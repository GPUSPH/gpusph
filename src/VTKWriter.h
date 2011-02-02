#ifndef _VTKWRITER_H
#define	_VTKWRITER_H

#include "Writer.h"

using namespace std;

class VTKWriter : public Writer
{
public:
	VTKWriter(const Problem *problem);
	~VTKWriter();

	void write(uint numParts, const float4 *pos, const float4 *vel,
			const particleinfo *info, const float3 *vort, float t);
};

#endif	/* _VTKWRITER_H */
