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

#ifndef _VTKWRITER_H
#define	_VTKWRITER_H

#include "Writer.h"

class VTKWriter : public Writer
{
	// name of the planes file. since planes are static (currently),
	// we only save one and reference it at each timestep
	std::string m_planes_fname;

	// index of the last written block
	int m_blockidx;

	// Save planes to a VTU file
	void save_planes();

	// Add a block (.vtu file) to the timefile
	void add_block(std::string const& blockname, std::string const& fname, double t);

	// this method is used to close the XML in the timefile,
	// so that the timefile is always valid, and then seek back to the pre-close
	// position so that the next entry is properly inserted
	void mark_timefile();

public:
	VTKWriter(const GlobalData *_gdata);
	~VTKWriter();

	void start_writing(double t);
	void mark_written(double t);

	virtual void write(uint numParts, BufferList const& buffers, uint node_offset, double t, const bool testpoints);
	virtual void write_WaveGage(double t, GageList const& gage);
};

#endif	/* _VTKWRITER_H */
