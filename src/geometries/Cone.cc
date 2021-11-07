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

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "Cone.h"

#if USE_CHRONO
#include "chrono/physics/ChBody.h"
#endif
#include "EulerParametersQuaternion.h"


Cone::Cone(void)
{
	m_origin = Point(0, 0, 0);
	m_rt = 0.0;
	m_rb = 0.0;
	m_h = 0.0;
	m_hg = 0.0;
	m_halfaperture = 0;
	m_ep = EulerParameters();
}


Cone::Cone(const Point& center, const double radiusbottom, const double radiustop, const Vector& height)
{
	m_origin = center;
	m_rb = radiusbottom;
	m_rt = radiustop;
	m_h = height.norm();

	m_halfaperture = atan((m_rb - m_rt)/m_h);

	Vector v(0, 0, 1);
	const double angle = acos(height*v/m_h);
	Vector rotdir = -height.cross(v);
	if (rotdir.norm() == 0)
		rotdir = Vector(0, 1, 0);
	m_ep = EulerParameters(rotdir, angle);
	m_ep.ComputeRot();

	m_hg = m_h*(m_rb*m_rb + 2.0*m_rb*m_rt + 3.0*m_rt*m_rt)/
				(4.0*(m_rb *m_rb + m_rb*m_rt + m_rt*m_rt));

	m_center = m_origin + m_ep.Rot(m_hg*v);
}


Cone::Cone(const Point& center, const double radiusbottom, const double radiustop, const double height, const EulerParameters& ep)
{
	m_origin = center;
	m_rb = radiusbottom;
	m_rt = radiustop;
	m_h = height;

	if (m_h <= 0)
		throw std::invalid_argument("Cone height must be positive");
	if (m_rb < 0)
		throw std::invalid_argument("Cone bottom radius cannot be negative");
	if (m_rt < 0)
		throw std::invalid_argument("Cone top radius cannot be negative");

	// There are a few places with hidden assumptions about the relationship between the bases.
	// Keep the assumption but make it explicit
	if (m_rb <= m_rt)
		throw std::invalid_argument("Cone bottom radius must be larger than top radius");

	// TODO maybe atan2 instead?
	m_halfaperture = atan((m_rb - m_rt)/m_h);

	m_hg = m_h*(m_rb*m_rb + 2.0*m_rb*m_rt + 3.0*m_rt*m_rt)/
				(4.0*(m_rb *m_rb + m_rb*m_rt + m_rt*m_rt));

	setEulerParameters(ep);
}


Cone::Cone(const Point& center, const Vector& radiusbottom, const Vector& radiustop, const Vector& height)
{
	if (fabs(radiusbottom*height) > 1e-8*radiusbottom.norm()*height.norm()
		|| fabs(radiustop*height) > 1e-8*radiustop.norm()*height.norm()) {
		std::cout << "Trying to construct a cone with non perpendicular radius and axis\n";
		exit(1);
	}

	m_origin = center;
	m_rb = radiusbottom.norm();
	m_rt = radiustop.norm();
	m_h = height.norm();

	Vector radiusdir = height.Normal();
	Vector generatrix(m_origin + m_rb*radiusdir, m_origin + height + m_rt*radiusdir);
	m_halfaperture = acos(height*generatrix/(height.norm()*generatrix.norm()));

	Vector v(0, 0, 1);
	const double angle = acos(height*v/m_h);
	Vector rotdir = height.cross(v);
	if (rotdir.norm() == 0)
		rotdir = Vector(0, 1, 0);
	m_ep = EulerParameters(rotdir, angle);
	m_ep.ComputeRot();

	m_hg = m_h*(m_rb*m_rb + 2.0*m_rb*m_rt + 3.0*m_rt*m_rt)/
				(4.0*(m_rb *m_rb + m_rb*m_rt + m_rt*m_rt));

	m_center = m_origin + m_hg*m_ep.Rot(Vector(0, 0, 1));

}


double
Cone::Volume(const double dx) const
{
	const double h = m_h + dx;
	const double rb = m_rb + dx/2.0;
	const double rt = m_rt + dx/2.0;

	const double volume = M_PI*h/3.0*(rb*rb + rb*rt + rt*rt);
	return volume;
}


void
Cone::SetInertia(const double dx)
{
	const double h = m_h + dx;
	const double rb = m_rb + dx/2.0;
	const double rt = m_rt + dx/2.0;

	const double d = 20.0*M_PI*(rb*rt + rb*rb + rt*rt);
	const double n1 = 2.0*h*h*(rb*rb + 3.0*rb*rt + 6.0*rt*rt);
	const double n2 = 3.0*(rb*rb*rb*rt + rb*rb*rt*rt + rb*rt*rt*rt + rb*rb*rb*rb + rt*rt*rt*rt);

	m_inertia[0] = m_mass*(n1 + n2)/d;
	m_inertia[1] = m_inertia[0];
	m_inertia[2] = 2.0*n2*m_mass/d;

	std::cout << "Inertia: " << m_inertia[0] << " " << m_inertia[1] << " " << m_inertia[2] << "\n";

}

void Cone::setEulerParameters(const EulerParameters &ep)
{
	m_ep = ep;
	m_ep.ComputeRot();
	m_center = m_origin + m_hg*m_ep.Rot(Vector(0, 0, 1));
}

// TODO: here assuming the cone is right (i.e. height == length(height_vector))
void Cone::getBoundingBox(Point &output_min, Point &output_max)
{
	double radius = max(m_rt, m_rb);
	Point corner_origin = m_origin - Vector( -radius, -radius, 0.0 );
	getBoundingBoxOfCube(output_min, output_max, corner_origin,
		Vector(2*radius, 0, 0), Vector(0, 2*radius, 0), Vector(0, 0, m_h) );
}

void Cone::shift(const double3 &offset)
{
	const Point poff = Point(offset);
	m_origin += poff;
	m_center += poff;
}

void
Cone::FillBorder(PointVect& points, const double dx, const bool bottom, const bool top)
{
	m_origin(3) = m_center(3);
	const int nz = (int) ceil(m_h/dx);
	const double dz = m_h/nz;
	for (int i = 0; i <= nz; i++)
		FillDiskBorder(points, m_ep, m_origin, m_rb - i*dz*tan(m_halfaperture), i*dz, dx, 2.0*M_PI*rand()/RAND_MAX);
	if (bottom && m_rb > 0)
		FillDisk(points, m_ep, m_origin, fmax(m_rb - dx, 0), 0.0, dx);
	if (top && m_rt > 0)
		FillDisk(points, m_ep, m_origin, fmax(m_rt - dx, 0), nz*dz, dx);
}

void
Cone::FillIn(PointVect& points, const double dx, const int layers)
{
	// TODO check for degenerate cases that should be replaced by Fill
	const int lmin = layers < 0 ? layers + 1 : 0;
	const int lmax = layers < 0 ? 0 : layers - 1;

	// For the cone, there is a complexity associated with the nesting.
	// Take the triangular cross-section of the Cone.
	// In the general case, this is an isosceles trapezium with bases 2*m_rb, 2*m_rt and height m_h.
	// The nested figure must have m_h reduced (increased, for negative layers) by 2*dx (one dx per base) per layer.
	// The sides must also move inside (outside, for negative layers) by 2*dx (each) per layer.
	// How much do m_rb, m_rt change then?
	// Let's call B, T the centers of the bottom and top base, and C, D the points where the bases respectively meet one of the sides,
	// so that B C D T is the right trapezium that is half of the original isosceles trapezium.
	// This trapezium is described by the lines:
	// y = By (bottom)
	// y = Ty (top)
	// (x - Cx)/(Dx - Cx) = (y - Cy)/(Dy - Cy) (oblique side)
	// y = By (right side)
	// with the additional conditions:
	// Tx = Bx
	// Ty = By + m_h
	// Cx = Bx + m_rb
	// Cy = By
	// Dx = Tx + m_rt = Bx + m_rt
	// Dy = Ty = By + m_h
	// and for simplicity we can take Bx = By = 0
	//
	// The oblique side can also be written
	// (Dy - Cy)(x - Cx) = (Dx - Cx)(y - Cy)
	// or
	// m_h (x - m_rb) = (m_rt - m_rb) y
	// and finally
	// m_h x - (m_rt - m_rb) y - m_h m_rb = 0
	// Assuming m_h ≠ 0 and letting slope = (m_rt - m_rb)/m_h, this reduces to:
	// x - slope y - m_rb = 0
	//
	// If ∆ is the nesting distance, the top and bottom are shifted by +∆, -∆ respectively.
	// For the oblique side, we need to find a prallel line with distance ∆, i.e. with
	// a coefficient K such that ∆ = (K + m_rb)/sqrt(1 + slope)
	// that solves to K = ∆ sqrt(1 + slope) - m_rb
	// The new oblique side equation is thus
	// x = slope y + ∆ sqrt(1 + slope) + m_rb
	// to be paired with y = B'y, y = T'y respectively,
	// where B'y = ∆ and T'y = Ty - ∆ = m_h - ∆
	//
	// This gives us C'x = slope ∆ + ∆ sqrt(1+slope) + m_rb
	// and D'x = slope (m_h - ∆) + ∆ sqrt(1+slope) + m_rb
	//
	// The new radii are:
	// rb' = m_rb + (sqrt(1+slope) + slope) * ∆
	// rt' = m_rb + slope*m_h - (sqrt(1+slope) - slope) * ∆ = m_rt + (sqrt(1+slope) - slope)*∆
	//
	// (note that depending on direction the actual sign for ∆ might be positive or negative)
	//
	// If any of the new radii is negative, it means that x = slope y + ∆ sqrt(1 + slope) +  m_rb
	// hits the x = 0 line before the base, at the special point
	//
	// y = -( m_rb + ∆ sqrt(1 + slope) ) / slope
	//
	// In this case, the actual height should be adjusted to the distance between the remaining base
	// and the new point. NOTE: since we always have m_rb > m_rt, this can only happen on the top side,
	// in which

	const double slope = (m_rb - m_rt)/m_h;
	const double s = sqrt(1.0 + slope);

	for (int l = lmin; l <= lmax; l++) {

		const double delta = l*dx;
		const Point smaller_origin = m_origin + Vector(0, 0, delta);

		double smaller_rb = m_rb - (s + slope)*delta;
		double smaller_rt = m_rt - (s - slope)*delta;
		double smaller_h = m_h - 2*delta;

		// The m_rt == 0 condition is to ensure that when expanding a pointy cone,
		// our expansion is still pointy.
		if (m_rt == 0 || smaller_rt < 0) {
			smaller_rt = 0;
			smaller_h = ( m_rb - delta*s ) / slope - delta;
		}

		Cone(smaller_origin, smaller_rb, smaller_rt, smaller_h, m_ep).
			FillBorder(points, dx, true, true);

	}
}




int
Cone::Fill(PointVect& points, const double dx, const bool fill)
{
	m_origin(3) = m_center(3);
	int nparts = 0;
	const int nz = (int) ceil(m_h/dx);
	const double dz = m_h/nz;
	const double dz_tan = dz*tan(m_halfaperture);
	for (int i = 0; i <= nz; i++) {
		// due to FMA, m_rb - i*dz_tan may actually become negative. clamp to zero
		const double r = fmax(m_rb - i*dz_tan, 0.0);
		nparts += FillDisk(points, m_ep, m_origin, r, i*dz, dx, fill);
	}

	return nparts;
}


bool
Cone::IsInside(const Point& p, const double dx) const
{
	Point lp = m_ep.TransposeRot(p - m_origin);
	const double h = m_h + dx;
	bool inside = false;
	const double z = lp(2);
	if (z > -dx && z < h) {
		const double r = m_rb - z*tan(m_halfaperture) + dx;
		if (lp(0)*lp(0) + lp(1)*lp(1) < r*r)
			inside = true;
	}

	return inside;
}

#if USE_CHRONO == 1
/* Create a cube Chrono body inside a specified Chrono physical system. If
 * collide is true this method also enables collision detection in Chrono.
 * Here we have to specialize this function for the Cone because the Chrono cone
 * is by default in the Y direction and ours in the Z direction.
 *	\param bodies_physical_system : Chrono physical system
 *	\param dx : particle spacing
 *	\param collide : add collision handling
 */
void
Cone::BodyCreate(::chrono::ChSystem *bodies_physical_system, const double dx, const bool collide)
{
	Object::BodyCreate(bodies_physical_system, dx, collide, EulerParameters(Q_from_AngAxis(::chrono::CH_C_PI/2., ::chrono::VECT_X)));
	if (collide)
		GeomCreate(dx);
}
#endif

/// Create a Chrono collision model
/* Create a Chrono collsion model for the cube.
 *	\param dx : particle spacing
 */
void
Cone::GeomCreate(const double dx) {
#if USE_CHRONO == 1
	m_body->GetCollisionModel()->ClearModel();
	const double rb = m_rb + dx/2.;
	const double rt = m_rt + dx/2.;
	const double h = m_h + dx;
	/*
	m_body->GetCollisionModel()->AddCone(rb, rt, h);
	m_body->GetCollisionModel()->BuildModel();
	m_body->SetCollide(true);
	*/
#else
	throw std::runtime_error("Chrono not active, cannot create geometry for Cone");
#endif
}
