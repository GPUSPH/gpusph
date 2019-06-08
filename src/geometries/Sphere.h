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

#ifndef _SPHERE_H
#define	_SPHERE_H

#include "Object.h"
#include "Point.h"
#include "Vector.h"


class Sphere: public Object {
	private:
		double	m_r;
		int FillBorder(PointVect&, const double, const double, const bool fill = true);
	public:
		Sphere(void);
		Sphere(const Point &, const double);
		virtual ~Sphere(void) {};

		double Volume(const double) const;
		void SetInertia(const double);

		void setEulerParameters(const EulerParameters &ep);
		void getBoundingBox(Point &output_min, Point &output_max);
		void shift(const double3 &offset);

		void FillBorder(PointVect&, const double);
		void FillIn(PointVect&, const double, const int);
		int Fill(PointVect&, const double, const bool fill = true);

		bool IsInside(const Point&, const double) const;

#if USE_CHRONO == 1
		void BodyCreate(::chrono::ChSystem * bodies_physical_system, const double dx, const bool collide,
			const ::chrono::ChQuaternion<> & orientation_diff);
#endif
};

#endif	/* _SPHERE_H */

