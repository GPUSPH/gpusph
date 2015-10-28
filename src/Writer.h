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
#include <map>
#include <stdlib.h>
// TODO on Windows it's direct.h
#include <sys/stat.h>

#include "particledefine.h"

// BufferList
#include "buffer.h"

// GageList
#include "simparams.h"

// Object
#include "Object.h"

// deprecation macros
#include "deprecation.h"

// Forward declaration of GlobalData and Problem, instead of inclusion
// of the respective headers, to avoid cross-include messes

struct GlobalData;
class Problem;

using namespace std;

// Writer types. Define new ones here and remember to include the corresponding
// header in Writer.cc and the switch case in the implementation of Writer::Create

enum WriterType
{
	COMMONWRITER,
	TEXTWRITER,
	VTKWRITER,
	VTKLEGACYWRITER,
	CALLBACKWRITER,
	CUSTOMTEXTWRITER,
	UDPWRITER,
	HOTWRITER
};

// list of writer type, write freq pairs
typedef vector<pair<WriterType, double> > WriterList;

class Writer;

// hash of WriterType, pointer to actual writer
typedef map<WriterType, Writer*> WriterMap;

// ditto, const
typedef map<WriterType, const Writer*> ConstWriterMap;

/*! The Writer class acts both as base class for the actual writers,
 * and a dispatcher. It holds a (static) list of writers
 * (whose content is decided by the Problem) and passes all requests
 * over to all the writers in the list.
 */
class Writer
{
	// list of actual writers
	static WriterMap m_writers;

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

	// return a WriterMap of the writers that need to write
	static ConstWriterMap
	NeedWrite(double t);

	// tell writers that we're starting to send write requests
	// returns the list of writers that will be involved
	static WriterMap
	StartWriting(double t, bool force=false);

	// mark writers as done if they needed to save at the given time
	static void
	MarkWritten(WriterMap writers, double t);

	// mark writers as written even though they didn't actually save
	static void
	FakeMarkWritten(ConstWriterMap writers, double t);

	// write points
	static void
	Write(WriterMap writers, uint numParts, BufferList const& buffers, uint node_offset, double t, const bool testpoints);

	// write wave gages
	static void
	WriteWaveGage(WriterMap writers, double t, GageList const& gage);

	// write object data
	static void
	WriteObjects(WriterMap writers, double t);

	// write object forces
	static void
	WriteObjectForces(WriterMap writers, double t, uint numobjects,
		const float3* computedforces, const float3* computedtorques,
		const float3* appliedforces, const float3* appliedtorques);

	// delete writers and clear the list
	static void
	Destroy();

	double get_write_freq() const
	{ return m_writefreq; }

	/* return the last file number as string */
	string last_filenum() const;

protected:

	Writer(const GlobalData *_gdata);
	virtual ~Writer();

	void set_write_freq(double f);

	// does this writer need special treatment?
	// (This is only used for the COMMONWRITER presently.)
	bool is_special() const
	{ return isnan(m_writefreq); }

	// Writers that need to do special things before starting to write
	// should override this
	virtual void
	start_writing() {}

	// finish writing. Writers that need to do special things when done
	// can override this problem, but they should call Writer::mark_written
	// inside
	virtual void
	mark_written(double t)
	{
		m_last_write_time = t;
		++m_FileCounter;
	}

	virtual bool
	need_write(double t) const;

	virtual void
	write(uint numParts, BufferList const& buffers, uint node_offset, double t, const bool testpoints) = 0;

	virtual void
	write_energy(double t, float4 *energy) {}

	virtual void
	write_WaveGage(double t, GageList const& gage) {}

	virtual void
	write_objects(double t) {}

	virtual void
	write_objectforces(double t, uint numobjects,
		const float3* computedforces, const float3* computedtorques,
		const float3* appliedforces, const float3* appliedtorques) {}

	uint getFilenum() const;

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

	inline string
	open_data_file(ofstream &out, const char* base)
	{ return open_data_file(out, base, string(), m_fname_sfx); }


	// time of last write
	double			m_last_write_time;
	// time between writes. Special values:
	// zero means write every time
	// negative values means don't write (writer disabled)
	double			m_writefreq;

	string			m_dirname;
	uint			m_FileCounter;
	ofstream		m_timefile;

	const Problem	*m_problem;
	string			current_filenum() const;
	const GlobalData*		gdata;
};

#endif	/* _WRITER_H */

