#ifndef H_CUSTOMTEXTWRITER_H
#define H_CUSTOMTEXTWRITER_H

#include "Writer.h"

using namespace std;

class CustomTextWriter : public Writer
{
public:
	CustomTextWriter(const Problem *problem);
	~CustomTextWriter();

	void write(uint numParts, const float4 *pos, const float4 *vel,
		const particleinfo *info, const float3 *vort, float t);
};

#endif
