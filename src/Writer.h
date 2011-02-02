#ifndef _WRITER_H
#define	_WRITER_H

// Standard C/C++ Library Includes
#include <fstream>
#include <string>
#include <stdlib.h>
// TODO on Windows it's direct.h
#include <sys/stat.h>

// Problem class
#include "Problem.h"

using namespace std;

class Writer
{
public:
	// maximum number of files
	static const int MAX_FILES = 99999;
	// number of characters needed to represent MAX_FILES
	static const int FNUM_WIDTH = 5;

	Writer(const Problem *problem);
	virtual ~Writer();

	virtual void write(uint numParts, const float4 *pos, const float4 *vel,
			const particleinfo *info, const float3 *vort, float t) = 0;

protected:
	string			m_dirname;
	uint			m_FileCounter;
	FILE*			m_timefile;
	const Problem	*m_problem;
	string			next_filenum();
};

#endif	/* _VTKWRITER_H */

