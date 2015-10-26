/*  Copyright 2011-2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

// Created by Andrew 12/2009

#ifndef _CONE_H
#define	_CONE_H

#include "Object.h"
#include "Point.h"
#include "Vector.h"


class Cone: public Object {
	private:
		Point	m_origin;
		double	m_rt;
		double	m_rb;
		double	m_h;
		double	m_hg;
		double	m_halfaperture;

	public:
		Cone(void);
		Cone(const Point&, const double, const double, const Vector&);
		Cone(const Point&, const double, const double, const double, const EulerParameters& = EulerParameters());
		Cone(const Point&, const Vector&, const Vector&, const Vector&);
		virtual ~Cone(void) {};

		double Volume(const double) const;
		void SetInertia(const double);

		void setEulerParameters(const EulerParameters &ep);
		void getBoundingBox(Point &output_min, Point &output_max);
		void shift(const double3 &offset);

		void FillBorder(PointVect& points, const double, const bool, const bool);
		void FillBorder(PointVect& points, double dx)
		{
			FillBorder(points, dx, true, true);
		}

		int Fill(PointVect& points, const double, const bool fill = true);
		void FillIn(PointVect& points, const double dx, const int layers)
		{ throw std::runtime_error("FillIn not implemented for this object!"); }

		bool IsInside(const Point&, const double) const;
};

#endif	/* _CONE_H */
