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

#ifndef _CYLINDER_H
#define	_CYLINDER_H

#include "Object.h"
#include "Point.h"
#include "Vector.h"


class Cylinder: public Object {
	private:
		Point	m_origin;
		double	m_r;
		double	m_ri;
		double	m_h;

		uint3	m_nels;		///< number of fea elements in thickness, circumference and height 

	public:
		Cylinder(void);
		Cylinder(const Point&, const double, const Vector&);
		Cylinder(const Point&, const Vector&, const Vector&);
		Cylinder(const Point& origin, const double outer_r, const double inner_r, const double height, uint nelst, uint nelsc, uint nelsh, const EulerParameters& = EulerParameters());
		Cylinder(const Point& o, const double r, const double h, const EulerParameters& ep = EulerParameters()) :
			Cylinder(o, r, 0, h, 1, 1, 1, ep) {}
		virtual ~Cylinder(void) {};

		double Volume(const double) const;
		void SetInertia(const double);

		void setEulerParameters(const EulerParameters &ep);
		void getBoundingBox(Point &output_min, Point &output_max);
		void shift(const double3 &offset);

		void FillBorder(PointVect&, const double, const bool, const bool);
		void FillBorder(PointVect& points, const double dx)
		{
			FillBorder(points, dx, true, true);
		}

		int Fill(PointVect&, const double, const bool fill = true);

		// for dyn bounds layers
		void FillIn(PointVect& points, const double dx, const int layers, const bool fill_tops);
		void FillIn(PointVect& points, const double dx, const int layers);

		bool IsInside(const Point&, const double) const;

#if USE_CHRONO == 1
		float4 getNaturalCoords(const double4 abs_coords);
		int4 getOwningNodes(const double4 abs_coords);

		void BodyCreate(::chrono::ChSystem * bodies_physical_system, const double dx, const bool collide,
			const EulerParameters & orientation_diff);
		void CreateFemMesh(::chrono::ChSystem * fea_system);
#endif
};
#endif	/* _CYLINDER_H */
