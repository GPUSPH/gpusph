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

#ifndef SEGMENT_H
#define	SEGMENT_H

#include "Object.h"
#include "Point.h"
#include "Vector.h"


class Segment: public Object {
	private:
		Point   m_origin;
		double	m_lx;
		Vector	m_vx;

		// FillIn has different meanings (and implementations!) depending on
		// the world dimensions
		void FillIn1D(PointVect& points, const double dx, const int layers);
		void FillIn2D(PointVect& points, const double dx, const int layers);
		// TODO
		// void FillIn3D(PointVect& points, const double dx, const int layers);

	public:
		Segment(void);
		Segment(const Point&, const double, const EulerParameters & = EulerParameters());
		Segment(const Point&, const Vector&);
		virtual ~Segment(void) {};

		double Volume(const double) const;
		void SetInertia(const double);

		void FillBorder(PointVect&, const double);

		int Fill(PointVect&, const double, const bool fill = true);
		void FillIn(PointVect& points, const double dx, const int layers);
		bool IsInside(const Point&, const double) const;

		void setEulerParameters(const EulerParameters &ep);
		void getBoundingBox(Point &output_min, Point &output_max);
		void shift(const double3 &offset);

		void BodyCreate(::chrono::ChSystem *bodies_physical_system, const double dx, const bool collide,
			const EulerParameters & orientation_diff)
		{
			if (Object::world_dimensions == 1)
				Object::BodyCreate(bodies_physical_system, dx, collide, orientation_diff);
			else
				throw std::runtime_error("Segment::BodyCreate not implemented !");
		};
};
#endif	/* SEGMENT_H */
