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

#ifndef _BUFFER_TRAITS_H
#define _BUFFER_TRAITS_H

// flag_t
#include "common_types.h"

/* BufferTraits: traits struct used to associate a buffer key
 * with the respective buffer type, number of arrays,
 * and printable name */

template<flag_t Key>
struct BufferTraits
{
	// type of the buffer
	typedef void element_type;
	// number of buffers. Defaults to zero, so that it may generate an error
	enum { num_buffers = 0 };
	// printable name of the buffer
	static char name[];
};

/* Define buffer keys and set buffer traits:
 * data type
 * number of arrays in buffer
 * printable name of buffer
 */
#define SET_BUFFER_TRAITS(code, _type, _nbufs, _name) \
template<> struct BufferTraits<code> \
{ \
	typedef _type element_type; \
	enum { num_buffers = _nbufs} ; \
	static const char name[]; \
}

#endif

