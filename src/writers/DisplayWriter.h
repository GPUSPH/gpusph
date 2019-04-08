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

#ifndef _DISPLAYWRITER_H
#define _DISPLAYWRITER_H

#include "Writer.h"

#include <vtkUnstructuredGrid.h>
#include <vtkSmartPointer.h>

class VTKCPAdaptor;

/*!
 * This class transforms simulation data to VTK data object and pass it co-processor adaptor.
 */
class DisplayWriter : public Writer
{
public:
	/*!
	 * Constructor.
	 * \param[in] _gdata pointer to the global data
	 */
	DisplayWriter(const GlobalData *_gdata);

	/*!
	 * Destructor.
	 */
	~DisplayWriter();

	virtual void write(uint numParts, BufferList const& buffers, uint node_offset, double t, const bool testpoints);

protected:
	/*!
	 * Transform simulation data to vtkUnstructuredGrid.
	 * \param[in] numParts number of particles
	 * \param[in] buffers data buffers
	 * \param[in] node_offset node offset
	 * \return vtkUnstructuredGrid pointer
	 */
	virtual vtkSmartPointer<vtkUnstructuredGrid> buildGrid(uint numParts, BufferList const& buffers, uint node_offset);

private:
	VTKCPAdaptor* m_adaptor;
};

#endif /* _DISPLAYWRITER_H */
