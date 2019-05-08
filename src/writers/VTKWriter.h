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
	// name of the saved DEM file. again, only one, for all timesteps
	std::string m_dem_fname;

	// string representation of the current time of writing;
	// this includes an (optional) indication of the current integration
	// step for intermediate saves (e.g. with inspect_preforce)
	std::string m_current_time;

	// index of the last written block
	int m_blockidx;

	// neighbors list structural information, used when neighbors list debugging is enabled
	const uint m_neiblist_stride; ///< stride between two neighbors of the same particle
	const uint m_neiblist_size; ///< maximum number of neighbors for one particle
	const uint m_neiblist_end; ///< end of the whole neighbors list
	const uint m_neib_bound_pos; ///< local neighbors list index of the first boundary neighbor

	// Save planes to a VTU file
	void save_planes();
	// Save DEM to a VTS file
	void save_dem();

	// Add a block (.vtp file) to the timefile
	void add_block(std::string const& blockname, std::string const& fname);

	// this method is used to close the XML in the timefile,
	// so that the timefile is always valid, and then seek back to the pre-close
	// position so that the next entry is properly inserted
	void mark_timefile();

public:
	VTKWriter(const GlobalData *_gdata);
	~VTKWriter();

	void start_writing(double t, WriteFlags const& write_flags);
	void mark_written(double t);

	virtual void write(uint numParts, BufferList const& buffers, uint node_offset, double t, const bool testpoints);
	virtual void write_WaveGage(double t, GageList const& gage);
};

#endif	/* _VTKWRITER_H */
