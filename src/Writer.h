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

#include "particledefine.h"

// BufferList
#include "buffer.h"

// GageList
#include "simparams.h"

// Forward declaration of GlobalData and Problem, instead of inclusion
// of the respective headers, to avoid cross-include messes

struct GlobalData;
class Problem;

using namespace std;

enum WriterType
{
	TEXTWRITER,
	VTKWRITER,
	VTKLEGACYWRITER,
	CUSTOMTEXTWRITER,
	UDPWRITER
};

// list of writer type, write freq pairs
typedef vector<pair<WriterType, uint> > WriterList;

class Writer
{
	static vector<Writer*> m_writers;

	static float m_timer_tick;

public:
	// maximum number of files
	static const uint MAX_FILES = 99999;
	// number of characters needed to represent MAX_FILES
	static const uint FNUM_WIDTH = 5;

	// create a specific writer based on the problem set in _gdata
	static void
	Create(GlobalData *_gdata);

	// does any of the writer need to write?
	static bool
	NeedWrite(float t);

	// mark writers as done if they needed to save
	// at the given time // (optionally force)
	static void
	MarkWritten(float t, bool force=false);

	// write points
	static void
	Write(uint numParts, BufferList const& buffers, uint node_offset, float t, const bool testpoints);

	// write wave gages
	static void
	WriteWaveGage(float t, GageList const& gage);

	static inline void
	SetTimerTick(float t)
	{ m_timer_tick = t; }

	static inline float
	GetTimerTick() { return m_timer_tick; }

	// destroy
	static void
	Destroy();

protected:

	Writer(const Problem *problem);
	virtual ~Writer();

	void set_write_freq(int f);

	bool need_write(float t);

	void setGlobalData(GlobalData *_gdata);

	virtual void
	write(uint numParts, BufferList const& buffers, uint node_offset, float t, const bool testpoints) = 0;

	inline void mark_written(float t) { m_last_write_time = t; }

	virtual void
	write_energy(float t, float4 *energy);

	virtual void
	write_WaveGage(float t, GageList const& gage);

	uint getLastFilenum();

	float			m_last_write_time;
	int				m_writefreq;

	string			m_dirname;
	uint			m_FileCounter;
	FILE*			m_timefile;
	FILE*			m_energyfile;
	//WaveGage
	FILE*			m_WaveGagefile;
	const Problem	*m_problem;
	string			next_filenum();
	string			current_filenum();
	GlobalData*		gdata;
};

#endif	/* _WRITER_H */

