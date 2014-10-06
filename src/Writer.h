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

// Writer types. Define new ones here and remember to include the corresponding
// header in Writer.cc and the switch case in the implementation of Writer::Create

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

/*! The Writer class acts both as base class for the actual writers,
 * and a dispatcher. It holds a (static) list of writers
 * (whose content is decided by the Problem) and passes all requests
 * over to all the writers in the list.
 */
class Writer
{
	// list of actual writers
	static vector<Writer*> m_writers;

	// base writing timer tick. Each writer has a write frequency which is
	// a multiple of this
	static double m_timer_tick;

	// should we be force saving regardless of timer ticks
	// and frequencies?
	// TODO FIXME might not be the most thread-safe way
	// to handle this
	static bool m_forced;

public:
	// maximum number of files
	static const uint MAX_FILES = 99999;
	// number of characters needed to represent MAX_FILES
	static const uint FNUM_WIDTH = 5;

	// fill in the list of writers from the WriterList provided by Problem,
	// and set the global data pointer in all of them
	static void
	Create(GlobalData *_gdata);

	// does any of the writers need to write at the given time?
	static bool NeedWrite(double t);

	// mark writers as done if they needed to save
	// at the given time (optionally force)
	static void
	MarkWritten(double t, bool force=false);

	// write points
	static void
	Write(uint numParts, BufferList const& buffers, uint node_offset, double t, const bool testpoints);

	// write wave gages
	static void
	WriteWaveGage(double t, GageList const& gage);

	// set the timer tick
	static inline void SetTimerTick(double t)
	{ m_timer_tick = t; }

	// get the timer tick value
	static inline float GetTimerTick()
	{ return m_timer_tick; }

	// record that the upcoming write requests should be forced (regardless of write frequency)
	static inline void
	SetForced(bool force)
	{ m_forced = force; }

	// delete writers and clear the list
	static void
	Destroy();

protected:

	Writer(const GlobalData *_gdata);
	virtual ~Writer();

	void set_write_freq(int f);

	bool need_write(double t) const;

	virtual void
	write(uint numParts, BufferList const& buffers, uint node_offset, double t, const bool testpoints) = 0;

	inline void mark_written(double t) { m_last_write_time = t; }

	virtual void
	write_energy(double t, float4 *energy);

	virtual void
	write_WaveGage(double t, GageList const& gage);

	uint getLastFilenum();

	double			m_last_write_time;
	// default suffix (extension) for data files)
	string			m_fname_sfx;

	/* open a data file on stream `out` assembling the file name from the provided
	 * base, the current node (in case of multi-node simulaions), the provided sequence
	 * number and the provided suffix
	 *
	 * Returns the file name (without the directory part)
	 */
	string
	open_data_file(ofstream &out, const char* base, string const& num, string const& sfx);

	inline string
	open_data_file(ofstream &out, const char* base, string const& num)
	{ return open_data_file(out, base, num, m_fname_sfx); }

	int				m_writefreq;

	string			m_dirname;
	uint			m_FileCounter;
	ofstream		m_timefile;
	ofstream		m_energyfile;
	ofstream		m_WaveGagefile;

	const Problem	*m_problem;
	string			next_filenum();
	string			current_filenum();
	const GlobalData*		gdata;
};

#endif	/* _WRITER_H */

