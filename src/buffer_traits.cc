/*  Copyright 2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

/* This file is only used to hold the actual strings
 * defining the buffer printable names
 */

#include "GlobalData.h"

// re-include define-buffers to set the printable name
#undef DEFINED_BUFFERS
#undef SET_BUFFER_TRAITS
#define SET_BUFFER_TRAITS(code, _type, _nbufs, _name) \
const char BufferTraits<code>::name[] = _name
#include "define_buffers.h"
#undef SET_BUFFER_TRAITS

