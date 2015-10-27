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
#include <cstdlib>
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
typedef std::vector<std::pair<WriterType, double> > WriterList;

class Writer;

// hash of WriterType, pointer to actual writer
typedef std::map<WriterType, Writer*> WriterMap;

// ditto, const
typedef std::map<WriterType, const Writer*> ConstWriterMap;

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

	// mark writers as done if they needed to save
	// at the given time (optionally force)
	static void
	MarkWritten(double t, bool force=false);

	// write points
	static void
	Write(uint numParts, BufferList const& buffers, uint node_offset, double t, const bool testpoints);

	// write wave gages
	static void
	WriteWaveGage(double t, GageList const& );

	// write object data
	static void
	WriteObjects(double t);

	// write object forces
	static void
	WriteObjectForces(double t, uint numobjects,
		const float3* computedforces, const float3* computedtorques,
		const float3* appliedforces, const float3* appliedtorques);

	// record that the upcoming write requests should be forced (regardless of write frequency)
	static inline void
	SetForced(bool force)
	{ m_forced = force; }

	// delete writers and clear the list
	static void
	Destroy();

	double get_write_freq() const
	{ return m_writefreq; }

	/* return the last file number as string */
	std::string last_filenum() const;

protected:

	Writer(const GlobalData *_gdata);
	virtual ~Writer();

	void set_write_freq(double f);

	// does this writer need special treatment?
	// (This is only used for the COMMONWRITER presently.)
	bool is_special() const
	{ return std::isnan(m_writefreq); }

	inline void
	mark_written(double t)
	{ m_last_write_time = t; }

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

	uint getLastFilenum() const;

	// default suffix (extension) for data files)
	std::string			m_fname_sfx;

	/* open a data file on stream `out` assembling the file name from the provided
	 * base, the current node (in case of multi-node simulaions), the provided sequence
	 * number and the provided suffix
	 *
	 * Returns the file name (without the directory part)
	 */
	std::string
	open_data_file(std::ofstream &out, const char* base, std::string const& num, std::string const& sfx);

	inline std::string
	open_data_file(std::ofstream &out, const char* base, std::string const& num)
	{ return open_data_file(out, base, num, m_fname_sfx); }

	inline std::string
	open_data_file(std::ofstream &out, const char* base)
	{ return open_data_file(out, base, std::string(), m_fname_sfx); }


	// time of last write
	double			m_last_write_time;
	// time between writes. Special values:
	// zero means write every time
	// negative values means don't write (writer disabled)
	double			m_writefreq;

	std::string			m_dirname;
	uint			m_FileCounter;
	std::ofstream		m_timefile;

	const Problem	*m_problem;
	std::string			next_filenum();
	std::string			current_filenum() const;
	const GlobalData*		gdata;
};

#endif	/* _WRITER_H */

