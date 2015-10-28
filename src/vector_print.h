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

/* Definition of the << operators for vector types */

#ifndef _VECTOR_PRINT_H
#define _VECTOR_PRINT_H

#include <ostream>

#include "common_types.h"

// operator to stream any 4-vector type
// we can't use a function template because it's far from trivial
// having it work with the CUDA vector types
#define DEFINE_STREAM_XYZW(type) \
template< class Traits> \
std::basic_ostream<char,Traits>& operator<< \
(std::basic_ostream<char,Traits>& os, type##4 const& t) \
{ return os << "(" << t.x << ", " << t.y << ", " << t.z << ", " << t.w << ")"; }

// ditto, 3-vector types
#define DEFINE_STREAM_XYZ(type) \
template< class Traits> \
std::basic_ostream<char,Traits>& operator<< \
(std::basic_ostream<char,Traits>& os, type##3 const& t) \
{ return os << "(" << t.x << ", " << t.y << ", " << t.z << ")"; }

#define DEFINE_STREAM(type) \
	DEFINE_STREAM_XYZ(type) \
	DEFINE_STREAM_XYZW(type)

DEFINE_STREAM(char)
DEFINE_STREAM(uchar)
DEFINE_STREAM(short)
DEFINE_STREAM(ushort)
DEFINE_STREAM(int)
DEFINE_STREAM(uint)
DEFINE_STREAM(long)
DEFINE_STREAM(ulong)

DEFINE_STREAM(float)
DEFINE_STREAM(double)

#endif

