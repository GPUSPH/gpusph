/*  Copyright (c) 2011-2019 INGV, EDF, UniCT, JHU

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

/*! \file
 * BufferTraits structure and quick-definition macro
 */

#ifndef _BUFFER_TRAITS_H
#define _BUFFER_TRAITS_H

// flag_t
#include "common_types.h"

/*! BufferTraits: traits struct used to associate a buffer key
 * with the respective buffer type, number of arrays,
 * and printable name */

template<flag_t Key>
struct BufferTraits
{
	//! type of the elements stored in the buffer
	typedef void element_type;
	//! number of buffers. Defaults to zero, so that it may generate an error
	enum { num_buffers = 0 };
	//! printable name of the buffer
	static char name[];
};

/*! Convert a Buffer key to a Buffer string, at runtime.
 *
 * The BufferTraits<Key>::name property is only accessible
 * when Key is known at compile time, so we also implement a function
 * to access it a runtime
 */

const char * getBufferName(flag_t key);

/*! Macro to define buffer keys and set buffer traits:
 *  * data type
 *  * number of arrays in buffer
 *  * printable name of buffer
 */
#define SET_BUFFER_TRAITS(code, _type, _nbufs, _name) \
template<> struct BufferTraits<code> \
{ \
	typedef _type element_type; \
	enum { num_buffers = _nbufs} ; \
	static const char name[]; \
}

#endif

