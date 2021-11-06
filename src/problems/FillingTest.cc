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

#include "FillingTest.h"
#include "GlobalData.h"

/* The purpose of the FillingTest problem is simply to generate a number of geometries
 * and see how they get filled. It is intended to be used to validate filling patterns. 
 */

FillingTest::FillingTest(GlobalData *gdata) :
	Problem(gdata)
{
	setup_framework();
	// we only care about seeing the result
	gdata->maxiter = 1;

	// No external forces
	set_gravity(0.0);

	// Fluid with unitary density;
	auto fluid = add_fluid(1.0);
	set_equation_of_state(fluid, 1.0, 1.0);

	// avoid rounding issues and make it easy to debug also in decimal
	set_deltap(1.0);

	// The basics: a filled box.
	const double box_side = 16.0;

	// The default filling method is “border-centered”, in the sense that the particles of the first layer
	// have their centers on the border.
	// This requires us to shift the geometries so that they don't interfere with each other
	const double shift = m_deltap/2;
	const double double_shift = m_deltap; // 2*shift, needed to fix up the length of the box

	// We also have to take positioning into consideration. The default is PP_CENTER, meaning that
	// for the box we don't need to do any shifting. We set up two boxes,
	// one with PP_CENTER, and one with PP_CORNER positioning policies.
	// The PP_CORNER one will be placed with corner in (0, 0, 0),
	// while the PP_CENTER one will be placed so that its center is 2*box_side away
	// from the PP_CORNER case along the y direction (without accounting for the shift!)

	const Point shifted_corner  = Point(shift, shift, shift);
	const Point centered_center = Point(box_side/2, 2*box_side, box_side/2);

	// PP_CENTER
	auto fluid_box_centered = addCube(GT_FLUID, FT_SOLID, centered_center, box_side - double_shift);
	setEraseOperation(fluid_box_centered, ET_ERASE_NOTHING);

	auto border_box_centered = addCube(GT_FIXED_BOUNDARY, FT_OUTER_BORDER, centered_center, box_side + double_shift);
	setEraseOperation(border_box_centered, ET_ERASE_NOTHING);

	// Let's also place a sphere, right above the centered box
	auto sphere_radius = box_side/2;
	auto sphere_center = centered_center + Vector(0, 0, box_side + sphere_radius);

	// Note that while for the box we correct the side with double_shift, for the sphere we only correct with
	// a SINGLE shift, since it's the radius we're talking about
	auto fluid_sphere = addSphere(GT_FLUID, FT_SOLID, sphere_center, sphere_radius - shift);
	setEraseOperation(fluid_sphere, ET_ERASE_NOTHING);
	auto border_sphere = addSphere(GT_FIXED_BOUNDARY, FT_OUTER_BORDER, sphere_center, sphere_radius + shift);
	setEraseOperation(border_sphere, ET_ERASE_NOTHING);

	setPositioning(PP_CORNER);
	auto fluid_box_corner = addCube(GT_FLUID, FT_SOLID, shifted_corner, box_side - double_shift);
	setEraseOperation(fluid_box_corner, ET_ERASE_NOTHING);

	auto border_box_corner = addCube(GT_FIXED_BOUNDARY, FT_OUTER_BORDER, -shifted_corner, box_side + double_shift);
	setEraseOperation(border_box_corner, ET_ERASE_NOTHING);

	// The cylinder is placed by PP_BOTTOM_CENTER
	setPositioning(PP_BOTTOM_CENTER);

	auto cyl_radius = sphere_radius;
	auto cyl_height = box_side;

	auto cyl_bottom = sphere_center - Vector(0, 3*box_side/2, sphere_radius);

	auto fluid_cyl = addCylinder(GT_FLUID, FT_SOLID, cyl_bottom + Vector(0, 0, shift), cyl_radius - shift, cyl_height - double_shift);
	setEraseOperation(fluid_cyl, ET_ERASE_NOTHING);
	auto border_cyl = addCylinder(GT_FIXED_BOUNDARY, FT_OUTER_BORDER, cyl_bottom - Vector(0, 0, shift), cyl_radius + shift, cyl_height + double_shift);
	setEraseOperation(border_cyl, ET_ERASE_NOTHING);

}

