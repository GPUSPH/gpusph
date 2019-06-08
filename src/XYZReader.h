/*  Copyright (c) 2011-2018 INGV, EDF, UniCT, JHU

    Istituto Nazionale di Geofisica e Vulcanologia, Sezione di Catania, Italy
    Électricité de France, Paris, France
    Università di Catania, Catania, Italy
    Johns Hopkins University, Baltimore (MD), USA

    This file is part of GPUSPH. Project founders:
        Alexis Hérault, Giuseppe Bilotta, Robert A. Dalrymple,
        Eugenio Rustico, Ciro Del Negro
    For a full list of authors and project partners, consult the logs
    and the project website <https://www.gpusph.org>

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

#include <string>
#include "Point.h"
#include "Reader.h"

class XYZReader : public Reader
{
public:
	//! returns the number of particles in the XYZ file
	size_t getNParts() override;

	//! allocates the buffer and reads the data from the XYZ file
	//! TODO FIXME this currently actually fills points
	//! rather than Reader::buf like the other Readers
	void read() override;
	//! read() variant that also sets the bounding box
	//! TODO this should be part of the common Reader interface
	void read(Point *bbox_min, Point *bbox_max);
};
