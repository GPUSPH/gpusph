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

/// Get the mass of object particles
/*! Get the mass of object particles.
 *
 *	\return mass of the object particles
 *
 *  NOTE: in case of SA boundaries, this should refer to boundary particles
 */
double Object::GetPartMass()
{
	return m_center(3);
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

/// Get the mass of the object
/*! Get the mass of the object.
 *
 *	\return mass of the object
 */
double Object::GetMass()
{
	return m_mass;
}


/// Set the objects center of gravity
/*! Directly set the object center of gravity
 * \param cg : center of gravity
 */
void
Object::SetCenterOfGravity(const double* cg)
{
	m_center(0) = cg[0];
	m_center(1) = cg[1];
	m_center(2) = cg[2];
}


/// Returns objects center of gravity
/*! Returns the object center of gravity
 * \return center of gravity
 */
double3
Object::GetCenterOfGravity(void) const
{
	return(make_double3(m_center(0), m_center(1), m_center(2)));
}


/// Returns object orientation
/*! Returns the object orientation
 * \return object orientation
 */
EulerParameters
Object::GetOrientation(void) const
{
	return m_ep;
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


/// Print ODE-related information such as position, CG, geometry bounding box (if any), etc.
// TODO: could be useful to print also the rotation matrix
void Object::ODEPrintInformation(const bool print_geom)
{
	if (m_ODEBody) {
		const dReal* cpos = dBodyGetPosition(m_ODEBody);
		dMass mass;
		dBodyGetMass(m_ODEBody, &mass);
		printf("ODE Body ID: %p\n", m_ODEBody);
		printf("   Mass: %e\n", mass.mass);
		printf("   Pos:	 %e\t%e\t%e\n", cpos[0], cpos[1], cpos[2]);
		printf("   CG:   %e\t%e\t%e\n", mass.c[0], mass.c[1], mass.c[2]);
		printf("   I:    %e\t%e\t%e\n", mass.I[0], mass.I[1], mass.I[2]);
		printf("         %e\t%e\t%e\n", mass.I[4], mass.I[5], mass.I[6]);
		printf("         %e\t%e\t%e\n", mass.I[8], mass.I[9], mass.I[10]);
		const dReal* rot = dBodyGetRotation(m_ODEBody);
		printf("   R:    %e\t%e\t%e\n", rot[0], rot[1], rot[2]);
		printf("         %e\t%e\t%e\n", rot[4], rot[5], rot[6]);
		printf("         %e\t%e\t%e\n", rot[8], rot[9], rot[10]);
		const dReal* quat = dBodyGetQuaternion(m_ODEBody);
		printf("   Q:    %e\t%e\t%e\t%e\n", quat[0], quat[1], quat[2], quat[3]);
	}
	// not only check if an ODE geometry is associated, but also it must not be a plane
	if (print_geom && m_ODEGeom && dGeomGetClass(m_ODEGeom) != dPlaneClass) {
		dReal bbox[6];
		const dReal* gpos = dGeomGetPosition(m_ODEGeom);
		dGeomGetAABB(m_ODEGeom, bbox);
		printf("ODE Geom ID: %p\n", m_ODEGeom);
		printf("   Position: %g\t%g\t%g\n", gpos[0], gpos[1], gpos[2]);
		printf("   B. box:   X [%g,%g], Y [%g,%g], Z [%g,%g]\n",
			bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
		printf("   size:     X [%g] Y [%g] Z [%g]\n", bbox[1] - bbox[0],
			bbox[3] - bbox[2], bbox[5] - bbox[4]);
	}
}


/// Return the particle vector associated with the object
/*! \return a reference to the particles vector associated with the object
 */
PointVect&
Object::GetParts(void)
{
	return m_parts;
}


/// Sets the number of particles associated with an object
void Object::SetNumParts(const int numParts)
{
	m_numParts = numParts;
}


/// Gets the number of particles associated with an object
/*! This function either returns the set number of particles which is used
 *  in case of a loaded STL mesh or the number of particles set in m_parts.
 *  NOTE: in case of SA_BOUNDARIES, SetNumParts() is called with number of
 *  boundary parts only, thus GetNumParts() returns the number of particles
 *  excluding vertices
 */
uint Object::GetNumParts()
{
	// if the number of particles was not explicitly set then obtain it from
	// m_parts
	if (m_numParts == 0)
		m_numParts = m_parts.size();

	return m_numParts;
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

/// Remove particles from particle vector
/*! Remove the particles of particles vector lying outside the object,
 * 	within a tolerance off dx.
 *  This method uses IsInside().
 *	\param points : particle vector
 *	\param dx : tolerance
 */
void Object::Intersect(PointVect& points, const double dx) const
{
	PointVect new_points;
	new_points.reserve(points.size());

	for (uint i = 0; i < points.size(); i++) {
		const Point & p = points[i];

		if (IsInside(p, -dx))
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

// Update the ODE rotation matrix according to m_ep (EulerParameters)
// NOTE: should not be called for planes, since they are "non-placeable" objects
// in ODE and ODE does not support isPlaceable() or similar
void Object::updateODERotMatrix()
{
	// alternative way:
	//double phi, theta, psi;
	//m_ep.ExtractEulerZXZ(psi, theta, phi);
	//dRFromEulerAngles(m_ODERot, -phi, -theta, -psi);

	dQuaternion quaternion;
	m_ep.ToODEQuaternion(quaternion);
	// if obj has body *and* geom, one setQuaternion() is enough - that's why "else"
	if (m_ODEBody)
		dBodySetQuaternion(m_ODEBody, quaternion);
	else
	if (m_ODEGeom)
		dGeomSetQuaternion(m_ODEGeom, quaternion);
}
