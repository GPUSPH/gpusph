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

#include <cmath>
#include <cstdlib>

#include "Object.h"


/// Compute the particle mass according to object volume and density
/*! The mass of object particles is computed dividing the object volume
 *  by the number of particles needed for filling and multiplying the
 *	result by the density.
 *
 *	The resulting mass is internally stored and returned for convenience.
 *	\param dx : particle spacing
 *	\param rho : density
 *	\return mass of particle
 *
 *  Beware, particle mass should be set before any filling operation
 */
double
Object::SetPartMass(const double dx, const double rho)
{
	PointVect points;
	const int nparts = Fill(points, dx, false);
	const double mass = Volume(dx)*rho/nparts;
	m_center(3) = mass;
	return mass;
}


/// Set the mass of object particles
/*! Directly set the mass of object particles without any computation.
 *
 *	\param mass : particle mass
 *
 *  Beware, particle mass should be set before any filling operation
 */
void
Object::SetPartMass(double mass)
{
	m_center(3) = mass;
}


/// Compute the object mass according to object volume and density
/*! The mass of object is computed by multiplying its volume (computed using Volume()) by its density.
 *
 *	The resulting mass is internally stored and returned for convenience.
 *	\param dx : particle spacing
 *	\param rho : density
 *	\return mass of object
 */
double
Object::SetMass(const double dx, const double rho)
{
	const double mass = Volume(dx)*rho;
	m_mass = mass;
	return mass;
}


/// Set the mass of the object
/*! Directly set the object mass without any computation.
 * \param mass : object mass
 */
void
Object::SetMass(const double mass)
{
	m_mass = mass;
}


/// Set the object principal moments of inertia
/*! Directly set the object principal moments of inertia.
 *	\param inertia : pointer to the array containing principal moments of inertia (3 values)
 */
void
Object::SetInertia(const double* inertia)
{
	m_inertia[0] = inertia[0];
	m_inertia[1] = inertia[1];
	m_inertia[2] = inertia[2];
}



/// Retrieve the object inertial data
/*! Respectively fill the parameters passed by reference with:
 *		- the object center of gravity
 *		- the object mass
 *		- the object principal moments of inertia
 *		- the Euler parameters defining the orientation of object principal axis of inertia respect to rest frame
 *	\param cg : center of gravity
 *	\param mass : mass
 *	\param inertia : pointer to an 3 values array
 *	\param ep : orientation of object principal axis of inertia
 */
void
Object::GetInertialFrameData(double* cg, double& mass, double* inertia, EulerParameters& ep) const
{
	cg[0] = m_center(0);
	cg[1] = m_center(1);
	cg[2] = m_center(2);
	mass = m_mass;
	inertia[0] = m_inertia[0];
	inertia[1] = m_inertia[1];
	inertia[2] = m_inertia[2];
	ep = m_ep;
}


/// Return the particle vector associated with the object
/*! \return a reference to the particles vector associated with the object
 */
PointVect&
Object::GetParts(void)
{
	return m_parts;
}

/// Fill the object with particles
/*! Fill the object by calling Fill(points, dx, true).
 *
 *	\param points : particle vector
 *	\param dx : particle spacing
 */
void
Object::Fill(PointVect& points, const double dx)
{
	Fill(points, dx, true);
}


/// Fill a disk
/*! Fill a disk defined by its radius, center, orientation and an offset value along the circle normal direction.
 *
 *  If the fill parameter is set to false the function just count the number of
 *  particles needed otherwise the particles are added to the particle vector.
 *	\param ep : orientation
 *	\param center : translation to apply
 *	\param r : radius
 *  \param z : offset along z axis
 *	\param dx : particle spacing
 *  \param fill : fill flag
 *	\return number of particles needed to fill the object
 */
int
Object::FillDisk(PointVect& points, const EulerParameters& ep, const Point& center,
		const double r, const double z, const double dx, const bool fill) const
{
	const int nr = (int) ceil(r/dx);
	const double dr = r/nr;
	int nparts = 0;
	for (int i = 0; i <= nr; i++)
		nparts += FillDiskBorder(points, ep, center, i*dr, z, dx, 2.0*M_PI*rand()/RAND_MAX, fill);

	return nparts;
}


/// Fill a portion of disk
/*! Fill a portion of disk defined by its minimum and maximum radius, center,
 *  orientation and an offset value along the circle normal direction.
 *
 *  If the fill parameter is set to false the function just count the number of
 *  particles needed otherwise the particles are added to the particle vector.
 *	\param ep : orientation
 *	\param center : translation to apply
 *	\param rmin : minimum radius
 *	\param rmax : maximum radius
 *  \param z : offset along z axis
 *	\param dx : particle spacing
 *  \param fill : fill flag
 *	\return number of particles needed to fill the object
 */
int
Object::FillDisk(PointVect& points, const EulerParameters& ep, const Point& center, const double rmin,
		const double rmax, const double z, const double dx, const bool fill) const
{
	const int nr = (int) ceil((rmax - rmin)/dx);
	const double dr = (rmax - rmin)/nr;
	int nparts = 0;
	for (int i = 0; i <= nr; i++)
		nparts += FillDiskBorder(points, ep, center, rmin + i*dr, z, dx, 2.0*M_PI*rand()/RAND_MAX, fill);

	return nparts;
}


/// Fill disk border
/*! Fill the border of the disk defined by its radius, center,
 *  orientation and an offset value along the circle normal direction.
 *  The particles are filled starting at angle theta0
 *
 *  If the fill parameter is set to false the function just count the number of
 *  particles needed otherwise the particles are added to the particle vector.
 *	\param ep : orientation
 *	\param center : translation to apply
 *	\param rmin : minimum radius
 *	\param rmax : maximum radius
 *  \param z : offset
 *	\param dx : particle spacing
 *	\param theta0 : starting angle
 *  \param fill : fill flag
 *	\return number of particles needed to fill the object
 */
int
Object::FillDiskBorder(PointVect& points, const EulerParameters& ep, const Point& center,
		const double r, const double z, const double dx, const double theta0, const bool fill) const
{
	const int np = (int) ceil(2.0*M_PI*r/dx);
	const double angle = 2.0*M_PI/np;
	int nparts = 0;
	for (int i = 0; i < np; i++) {
		const double theta = theta0 + angle*i;
		nparts++;
		if (fill) {
			Point p = ep.Rot(Point(r*cos(theta), r*sin(theta), z)) + center;
			p(3) = center(3);
			points.push_back(p);
		}
	}
	if (np == 0) {
		nparts++;
		if (fill) {
			Point p = ep.Rot(Point(0, 0, z)) + center;
			p(3) = center(3);
			points.push_back(p);
		}
	}

	return nparts;
}


/// Remove particles from particle vector
/*! Remove the particles of particles vector lying inside the object
 * 	within a tolerance off dx.
 *  This method used IsInside().
 *	\param points : particle vector
 *	\param dx : tolerance
 */
void Object::Unfill(PointVect& points, const double dx) const
{
	PointVect new_points;
	new_points.reserve(points.size());

	for (uint i = 0; i < points.size(); i++) {
		const Point & p = points[i];

		if (!IsInside(p, dx))
			new_points.push_back(p);
	}

	points.clear();

	points = new_points;
}

// auxiliary function for computing the bounding box
void Object::getBoundingBoxOfCube(Point &out_min, Point &out_max,
	Point &origin, Vector v1, Vector v2, Vector v3)
{
	// init min and max to origin
	Point currMin = origin;
	Point currMax = origin;
	// compare to corners adjacent to origin
	setMinMaxPerElement(currMin, currMax, origin + v1);
	setMinMaxPerElement(currMin, currMax, origin + v2);
	setMinMaxPerElement(currMin, currMax, origin + v3);
	// compare to other corners
	setMinMaxPerElement(currMin, currMax, origin + v1 + v2);
	setMinMaxPerElement(currMin, currMax, origin + v1 + v3);
	setMinMaxPerElement(currMin, currMax, origin + v2 + v3);
	setMinMaxPerElement(currMin, currMax, origin + v1 + v2 + v3);
	// output in double3
	out_min(0) = currMin(0);
	out_min(1) = currMin(1);
	out_min(2) = currMin(2);
	out_max(0) = currMax(0);
	out_max(1) = currMax(1);
	out_max(2) = currMax(2);
}
