#ifndef H_TEXTWRITER_H
#define H_TEXTWRITER_H

#include "Writer.h"

using namespace std;

class TextWriter : public Writer
{
public:
	TextWriter(const Problem *problem);
	~TextWriter();

	void write(uint numParts, const float4 *pos, const float4 *vel,
		const particleinfo *info, const float3 *vort, float t);
};

#endif
