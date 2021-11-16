/*  Copyright (c) 2021 INGV, EDF, UniCT, JHU

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

#ifndef INTERSECTION_TYPE_H
#define INTERSECTION_TYPE_H

//! Intersection type
//! This describes how an object handles particles from other objects.
//! For global objects in Problem API 1, it determines if the object keeps existing particles
//! that intersects its volume unchanged (IT_NONE),
//! or keeps only particles inside (IT_INTERSECT) or outside (IT_SUBTRACT) of its volume.
//! For the IntersectionObject, effectively only IT_SUBTRACT has a special meaning, indicating
//! that the corresponding component should flip the inside/outside logic.
enum IntersectionType {	IT_NONE,		// don't intersect (API 1). This is ignored for the IntersectionObject
						IT_INTERSECT,	// keep only particles inside
						IT_SUBTRACT		// keep only particles outside
};

#endif

