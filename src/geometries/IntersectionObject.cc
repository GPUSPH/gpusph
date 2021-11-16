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

#include "IntersectionObject.h"

int IntersectionObject::Fill(PointVect& points, const double dx, const bool fill)
{
	if (m_components.empty()) return 0;
	// TODO FIXME fill=false is only used when computing particle properties
	// such as mass or density, in relation to the volume of geometry.
	// Since we don't have an easy way to compute the volume of an intersection
	// (for the time being) we simply operate on the first component in these cases
	if (m_components.size() == 1 || ! fill) return m_components[0].object->Fill(points, dx, fill);

	PointVect all;
	m_components[0].object->Fill(all, dx, fill);

	for (auto c = begin(m_components)+1; c != end(m_components); ++c)
		c->Intersect(all, dx);

	int count = 0;
	for (auto const& p : all) {
		points.push_back(p);
		++count;
	}
	return count;
}

void IntersectionObject::FillIn(PointVect& points, const double dx, const int layers)
{
	if (m_components.empty()) return;
	if (m_components.size() == 1) return m_components[0].object->FillIn(points, dx, layers);

	PointVect all;
	m_components[0].object->FillIn(all, dx, layers);

	for (auto c = begin(m_components)+1; c != end(m_components); ++c)
		c->Intersect(all, dx);
	for (auto const& p: all)
		points.push_back(p);
}

void IntersectionObject::getBoundingBox(Point &output_min, Point& output_max)
{
	Point globalMin = Point(-DBL_MAX, -DBL_MAX, -DBL_MAX);
	Point globalMax = Point(DBL_MAX, DBL_MAX, DBL_MAX);
	for (auto const& c : m_components) {
		if (c.intersection_type == IT_SUBTRACT)
			continue; // TODO how should we handle these components?
		Point pMin, pMax;
		c.object->getBoundingBox(pMin, pMax);
		// note the inversion
		setMaxPerElement(globalMin, pMin);
		setMinPerElement(globalMax, pMax);
	}
	output_min = globalMin;
	output_max = globalMax;
}

