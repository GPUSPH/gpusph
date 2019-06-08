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

/*! \file
 * Data structures and associated definitions for geometric planes
 */

#ifndef _PLANES_H
#define _PLANES_H

#include <vector>

#include "common_types.h"
#include "vector_math.h"

/*! A plane is defined by its normal and a point it passes through. The
 *  reference point is defined in grid+local coordinates
 */
struct plane_t {
	float3 normal; /// normal to the plane
	int3 gridPos; /// grid position of the reference point
	float3 pos; /// local position of the reference point
};

__host__ __device__
static inline
plane_t make_plane(float3 const& normal, int3 const& gridPos, float3 const& pos)
{
	plane_t plane;
	plane.normal = normal;
	plane.gridPos = gridPos;
	plane.pos = pos;

	return plane;
}

typedef std::vector<plane_t> PlaneList;

#endif
