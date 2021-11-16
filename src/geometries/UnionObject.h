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

#ifndef UNION_OBJECT_H
#define	UNION_OBJECT_H

#include <vector>
#include <string>
#include <cfloat>

/* Object that is the union of multiple other objects */

// TODO this is fairly incomplete yet.

#include "Object.h"

#define UNION_UNIMPLEMENTED throw std::runtime_error(std::string(__func__) + " unimplemented")

class UnionObject : public Object
{
	std::vector<ObjectPtr> m_components;

public:
	//! Add a copy of the given object as a component
	template<typename T>
	void addComponent(T const& obj) { m_components.push_back(std::make_shared<T>(obj)); }
	void addComponent(ObjectPtr const& obj) { m_components.push_back(obj->clone()); }

	template<typename H>
	void addComponents(H const& obj) { addComponent(obj); }

	template<typename H, typename ...T>
	void addComponents(H const& obj, T const&... rest)
	{ addComponent(obj) ; addComponents(rest...); }

	UnionObject() : m_components() {}

	UnionObject(UnionObject const& o) : UnionObject()
	{ for (auto& c : o.m_components) { addComponent(c); } }

	template<typename ...T>
	UnionObject(T const&... objs) : UnionObject() { addComponents(objs...); }

	std::vector<ObjectPtr>& components() { return m_components; }

	// TODO
	void FillBorder(PointVect& points, const double dx) override { UNION_UNIMPLEMENTED; }
	// TODO
	int Fill(PointVect& points, const double dx, const bool fill = true) override { UNION_UNIMPLEMENTED; }
	// TODO
	void FillIn(PointVect& points, const double dx, const int layers) override { UNION_UNIMPLEMENTED; }

	bool IsInside(const Point& p, const double dx) const override
	{
		for (auto c : m_components) {
			if (c->IsInside(p, dx)) return true;
		}
		return false;
	}

	void getBoundingBox(Point &output_min, Point& output_max) override
	{
		Point globalMin = Point(DBL_MAX, DBL_MAX, DBL_MAX);
		Point globalMax = Point(-DBL_MAX, -DBL_MAX, -DBL_MAX);
		for (auto c : m_components) {
			Point pMin, pMax;
			c->getBoundingBox(pMin, pMax);
			setMinPerElement(globalMin, pMin);
			setMaxPerElement(globalMax, pMax);
		}
		output_min = globalMin;
		output_max = globalMax;
	}

	// TODO
	void setEulerParameters(EulerParameters const& ep) override { UNION_UNIMPLEMENTED; }
	// TODO
	void shift(const double3& offset) override { UNION_UNIMPLEMENTED; }


};
#endif
