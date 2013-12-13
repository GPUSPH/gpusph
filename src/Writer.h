/*  Copyright 2011-2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

    Istituto Nazionale di Geofisica e Vulcanologia
        Sezione di Catania, Catania, Italy

    Università di Catania, Catania, Italy

    Johns Hopkins University, Baltimore, MD

    This file is part of GPUSPH.

    GPUSPH is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    GPUSPH is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with GPUSPH.  If not, see <http://www.gnu.org/licenses/>.
*/
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
#include "particledefine.h"

// used for dumping the device index
#include "GlobalData.h"

using namespace std;

class Writer
{
public:
	// maximum number of files
	static const uint MAX_FILES = 99999;
	// number of characters needed to represent MAX_FILES
	static const uint FNUM_WIDTH = 5;

	Writer(const Problem *problem);
	virtual ~Writer();

	virtual void write(uint numParts, const double4 *pos, const float4 *vel,
			const particleinfo *info, const float3 *vort, float t, const bool testpoints, const float4 *normals, const float4 *gradGamma = 0, const float *tke = 0, const float *turbvisc = 0) = 0;

	virtual void write_energy(float t, float4 *energy);
	//WaveGage
	virtual void write_WaveGage(float t, GageList const& gage);

	void setGlobalData(GlobalData *_gdata);

	uint getLastFilenum();

protected:
	string			m_dirname;
	uint			m_FileCounter;
	FILE*			m_timefile;
	FILE*			m_energyfile;
	//WaveGage
	FILE*			m_WaveGagefile;
	const Problem	*m_problem;
	string			next_filenum();
	GlobalData*		m_gdata;
};

#endif	/* _VTKWRITER_H */

