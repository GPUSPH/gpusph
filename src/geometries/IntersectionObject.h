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

#ifndef INTERSECTION_OBJECT_H
#define	INTERSECTION_OBJECT_H

#include <vector>
#include <string>
#include <cfloat>

/* Object that is the intersection of multiple other objects */

#include "Object.h"
#include "IntersectionType.h"

#define INTERSECTION_UNIMPLEMENTED throw std::runtime_error(std::string(__func__) + " unimplemented")

struct IntersectionComponent
{
	ObjectPtr			object; // actual component
	IntersectionType	intersection_type;	// intersection type (inside/outside)
	double				unfill_radius; // dx to be applied to the geometry when unfilling

	template<typename T>
	IntersectionComponent(T const& arg_object,
		IntersectionType arg_intersection_type=IT_INTERSECT, double arg_unfill_radius=NAN)
	:
		object(std::make_shared<T>(arg_object)),
		intersection_type(arg_intersection_type),
		unfill_radius(arg_unfill_radius)
	{}

	IntersectionComponent(ObjectPtr const& arg_object,
		IntersectionType arg_intersection_type=IT_INTERSECT, double arg_unfill_radius=NAN)
	:
		object(arg_object->clone()),
		intersection_type(arg_intersection_type),
		unfill_radius(arg_unfill_radius)
	{}

	IntersectionComponent(IntersectionComponent const& other) :
		object(other.object->clone()),
		intersection_type(other.intersection_type),
		unfill_radius(other.unfill_radius)
	{}

	inline bool IsInside(const Point& p, const double dx) const
	{
		const double used_dx = std::isfinite(unfill_radius) ? unfill_radius : dx;
		const bool inside = object->IsInside(p, used_dx);
		return intersection_type == IT_SUBTRACT ? !inside : inside;
	}

	inline void Unfill(PointVect& v, const double dx) const
	{
		const double used_dx = std::isfinite(unfill_radius) ? unfill_radius : dx;
		intersection_type == IT_SUBTRACT ?
			object->Intersect(v, -used_dx) :
			object->Unfill(v, used_dx);
	}

	inline void Intersect(PointVect& v, const double dx) const
	{
		const double used_dx = std::isfinite(unfill_radius) ? unfill_radius : dx;
		intersection_type == IT_SUBTRACT ?
			object->Unfill(v, used_dx) :
			object->Intersect(v, used_dx);
	}
};

class IntersectionObject : public Object
{
	std::vector<IntersectionComponent> m_components;

public:
	//! Add a copy of the given object as a component
	template<typename T>
	void addComponent(T const& obj,
		IntersectionType intersection_type=IT_INTERSECT, double unfill_radius=NAN)
	{ m_components.push_back(IntersectionComponent(obj, intersection_type, unfill_radius)); }

	std::vector<IntersectionComponent>& components() { return m_components; }
	std::vector<IntersectionComponent> const& components() const { return m_components; }

	//! Volume of the intersection
	//! TODO FIXME currently we return the volume of the first component
	//! \seealso Fill() where we only used the first component if fill=false
	//! (so that the results for the particles' mass and density are at least consistent
	double Volume(const double dx) const override
	{ return m_components.empty() ? 0 : m_components[0].object->Volume(dx); }

	// TODO
	void FillBorder(PointVect& points, const double dx) override { INTERSECTION_UNIMPLEMENTED; }

	//! Fill the first component and then intersect/subtract the others
	//! NOTE: this will fail if the first component has intersection type IT_SUBTRACT
	int Fill(PointVect& points, const double dx, const bool fill = true) override;

	//! Fill-in the first component and then intersect/subtract the others
	//! NOTE: this will fail if the first component has intersection type IT_SUBTRACT
	//! NOTE: this is not a “proper” FillIn, in the sense that it does not follow
	//! the geometry of the intersection.
	//! ** This may change in the future **
	void FillIn(PointVect& points, const double dx, const int layers) override;

	bool IsInside(const Point& p, const double dx) const override
	{
		for (auto const& c : m_components) {
			if (!c.IsInside(p, dx)) return false;
		}
		return true;
	}

	// for the time being we return the smallest of the bounding boxes.
	// this could be refined
	void getBoundingBox(Point &output_min, Point& output_max) override;

	// TODO
	void setEulerParameters(EulerParameters const& ep) override { INTERSECTION_UNIMPLEMENTED; }
	// TODO
	void shift(const double3& offset) override { INTERSECTION_UNIMPLEMENTED; }
	// TODO
	void SetInertia(const double dx) override { INTERSECTION_UNIMPLEMENTED; }


};
#endif
