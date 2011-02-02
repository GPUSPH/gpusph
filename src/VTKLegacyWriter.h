#ifndef _VTKLEGACYWRITER_H
#define	_VTKLEGACYWRITER_H

#include "Writer.h"

using namespace std;

class VTKLegacyWriter : public Writer
{
public:
	VTKLegacyWriter(const Problem *problem);
	~VTKLegacyWriter();

	void write(uint numParts, const float4 *pos, const float4 *vel,
			const particleinfo *info, const float3 *vort, float t);
};

#endif	/* _VTKLEGACYWRITER_H */
