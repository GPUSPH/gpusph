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

#ifndef _VTKLEGACYWRITER_H
#define	_VTKLEGACYWRITER_H

#include "Writer.h"

class VTKLegacyWriter : public Writer
{
public:
	VTKLegacyWriter(const GlobalData *_gdata);
	~VTKLegacyWriter();

	virtual void write(uint numParts, BufferList const& buffers, uint node_offset, double t, const bool testpoints);

	// this method is used to close the XML in the timefile,
	// so that the timefile is always valid, and then seek back to the pre-close
	// position so that the next entry is properly inserted
	void mark_timefile();
};

#endif	/* _VTKLEGACYWRITER_H */
