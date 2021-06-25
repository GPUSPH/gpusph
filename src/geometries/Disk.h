/*  Copyright (c) 2011-2017 INGV, EDF, UniCT, JHU

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

#ifndef _DISK_H
#define	_DISK_H

#include "Object.h"
#include "Point.h"
#include "Vector.h"


class Disk: public Object {
	private:
		double	m_r;

	public:
		Disk(void);
		Disk(const Point&, const double);
		Disk(const Point&, const double, const Vector&);
		Disk(const Point&, const double, const EulerParameters&);
		Disk(const Point&, const Vector&, const Vector &);
		virtual ~Disk(void) {};

		double Volume(const double) const;
		void SetInertia(const double);

		void setEulerParameters(const EulerParameters &ep);
		void getBoundingBox(Point &output_min, Point &output_max);
		void shift(const double3 &offset);

		void FillBorder(PointVect&, const double);

		int Fill(PointVect&, const double, const bool fill = true);
		void FillIn(PointVect& points, const double dx, const int layers);
		void FillIn2D(PointVect& points, const double dx, const int layers);
		void FillIn3D(PointVect& points, const double dx, const int layers);

		bool IsInside(const Point&, const double) const;

#if USE_CHRONO == 1
		void BodyCreate(::chrono::ChSystem *bodies_physical_system, const double dx, const bool collide)
		{ throw std::runtime_error("Disk::BodyCreate not implemented !"); }
#else
		void BodyCreate(void *p1, const double p2, const bool p3)
		{ Object::BodyCreate(p1, p2, p3); }
#endif
};
#endif	/* _CIRCLE_H */


