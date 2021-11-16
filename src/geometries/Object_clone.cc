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

//! \file: implementation of the Object.clone() method

#include "Object.h"

#include "Cone.h"
#include "Cube.h"
#include "Cylinder.h"
#include "Disk.h"
#include "Plane.h"
#include "Rect.h"
#include "Segment.h"
#include "Sphere.h"
#include "Torus.h"
#include "IntersectionObject.h"

ObjectPtr
Object::clone() const
{
#define TRY(klass) do  {\
	const klass * ptr = dynamic_cast<const klass*>(this); \
	if (ptr) return std::make_shared<klass>(*ptr); \
} while (0)

	TRY(Cone);
	TRY(Cube);
	TRY(Cylinder);
	TRY(Disk);
	TRY(Plane);
	TRY(Rect);
	TRY(Segment);
	TRY(Sphere);
	TRY(Torus);
	TRY(IntersectionObject);

	throw std::runtime_error("Cannot clone this object");
}
