/*  Copyright 2011 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

	Istituto de Nazionale di Geofisica e Vulcanologia
          Sezione di Catania, Catania, Italy

    Universita di Catania, Catania, Italy

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

#ifdef __APPLE__
#include <OpenGl/gl.h>
#else
#include <GL/gl.h>
#endif
#include <math.h>

#include "Object.h"


/// Compute the particle mass according to object volume (computed using Volume()) and density
/*! The mass of object particles is computed dividing the object volume 
 *  by the number of particles nedded for filling and multiplying the
 *	result by the density. 
 * 
 *	The resulting mass is interally stored and returned for convenience.
 *	\param dx : particle spacing
 *	\param rho : density
 *	\return mass of particle
 * 
 *  Beware, particle mass sould be set before any filling operation
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
/*! Directly set the mas of object particles without any computation.
 * 
 *	\param mass : particle mass
 * 
 *  Beware, particle mass sould be set before any filling operation
 */
void
Object::SetPartMass(double mass)
{
	m_center(3) = mass;
}


/// Compute the object mass according to object volume and density
/*! The mass of object is computed by multiplyig its volume (computed using Volume()) by its density. 
 * 
 *	The resulting mass is interally stored and returned for convenience.
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
 * 
 *	\param mass : object mass
 */
void
Object::SetMass(const double mass)
{
	m_mass = mass;
}


/// Set the object principal moments of inertia 
/*! Directly set the object principal moments of inertia.
 * 
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
/*! Respectivly fill the parameters passed by reference with:
 *		- the object center of gravity 
 *		- the object mass
 *		- the object principal moments of inertia
 *		- the Euler parameters defining the orientation of object principal axis of inertia respect to rest frame   
 * 
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


/// Draw a pallelogram 
/*! Draw the parallelogram of summits p1, p2, p3, p4   
 * 
 *	\param p1, p2, p3, p3 : summits of parallelogram
 */
void
Object::GLDrawQuad(const Point& p1, const Point& p2,
		const Point& p3, const Point& p4) const
{
	glBegin(GL_QUADS);
	glVertex3f((float) p1(0), (float) p1(1), (float) p1(2));
	glVertex3f((float) p2(0), (float) p2(1), (float) p2(2));
	glVertex3f((float) p3(0), (float) p3(1), (float) p3(2));
	glVertex3f((float) p4(0), (float) p4(1), (float) p4(2));
	glEnd();
}


/// Draw a pallelogram 
/*! Apply a rotation defined by Eulerparameters and a translation to points p1, p2, p3, p4 
 *  and draw the resulting parallelogram  
 * 
 *	\param ep : rotation to apply
 *	\param trans : translation to apply
 *	\param p1, p2, p3, p3 : parallelogram summits
 */
void
Object::GLDrawQuad(const EulerParameters& ep, const Point& p1, const Point& p2,
		const Point& p3, const Point& p4, const Point &trans) const
{
	glBegin(GL_QUADS);
	Point p = ep.Rot(p1) + trans;
	glVertex3f((float) p(0), (float) p(1), (float) p(2));
	p = ep.Rot(p2) + trans;
	glVertex3f((float) p(0), (float) p(1), (float) p(2));
	p = ep.Rot(p3) + trans;
	glVertex3f((float) p(0), (float) p(1), (float) p(2));
	p = ep.Rot(p4) + trans;
	glVertex3f((float) p(0), (float) p(1), (float) p(2));
	glEnd();
}


void
Object::GLDrawQuad(const EulerParameters& ep, const Point& p1, const Point& p2,
		const Point& p3, const Point& p4, const Vector &trans) const
{
	glBegin(GL_QUADS);
	Point p = ep.Rot(p1) + trans;
	glVertex3f((float) p(0), (float) p(1), (float) p(2));
	p = ep.Rot(p2) + trans;
	glVertex3f((float) p(0), (float) p(1), (float) p(2));
	p = ep.Rot(p3) + trans;
	glVertex3f((float) p(0), (float) p(1), (float) p(2));
	p = ep.Rot(p4) + trans;
	glVertex3f((float) p(0), (float) p(1), (float) p(2));
	glEnd();
}


/// Draw a segment 
/*! Apply a segment between points p1 and p2
 * 
 *	\param p1, p2, p3, p3 : summits of parallelogram
 */
void
Object::GLDrawLine(const Point& p1, const Point& p2) const
{
	glBegin(GL_LINES);
	glVertex3f((float) p1(0), (float) p1(1), (float) p1(2));
	glVertex3f((float) p2(0), (float) p2(1), (float) p2(2));
	glEnd();
}


/// Draw a circle 
/*! Draw a circle defined by its radius, center, orientation and an offset value along the circle normal direction.
 * 
 *	\param ep : orientation
 *	\param center : translation to apply
 *	\param r : radius
 *  \param z : offset 
 */
void
Object::GLDrawCircle(const EulerParameters& ep, const Point& center, const double r, const double z) const
{
	#define CIRCLE_LINES 36
	const double angle = 2.0*M_PI/CIRCLE_LINES;
	glBegin(GL_POLYGON);
	for (int i = 0; i < CIRCLE_LINES; ++i) {
			double u = i*angle;
			Point p = ep.Rot(Point(r*cos(u), r*sin(u), z));
			p += center;
			glVertex3f(p(0), p(1), p(2));
		}
	glEnd();
	#undef CIRCLE_LINES
}


/// Fill the object with particles 
/*! Fill the object by callin Fill(points, dx, true).
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
	
 * 
 *	\param ep : orientation
 *	\param center : translation to apply
 *	\param r : radius
 *  \param z : offset 
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
 * 
 *	\param ep : orientation
 *	\param center : translation to apply
 *	\param rmin : minimum radius
 *	\param rmax : maximum radius
 *  \param z : offset 
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
 * 
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
			Point p(r*cos(theta), r*sin(theta), z, center(3));
			points.push_back(ep.Rot(p) + center);
		}
	}
	if (np == 0) {
		nparts++;
		if (fill) {
			Point p(0.0, 0.0, z, center(3));
			points.push_back(ep.Rot(p) + center);
		}
	}
	
	return nparts;
}


//void Object::GetBoundingBox(const PointVect& points, const double dx, 
//							float3 & min, float3 & max)
//{
//	int size = points.size();
//	
//	if (size == 0)
//		return;
//	
//	min = make_float3(points[0]);
//	max = make_float3(points[0]);
//	
//	for (int i = 1; i < size; i++) {
//		const Point & p = points[i];
//		min.x = p(0) < min.x ? p(0) : min.x;
//		min.y = p(1) < min.y ? p(1) : min.y;
//		min.z = p(2) < min.z ? p(2) : min.z;
//		
//		max.x = p(0) > max.x ? p(0) : max.x;
//		max.y = p(1) > max.y ? p(1) : max.y;
//		max.z = p(2) > max.z ? p(2) : max.z;
//	}
//	
//	min.x -= dx;
//	min.y -= dx;
//	min.z -= dx;
//	
//	max.x += dx;
//	max.y += dx;
//	max.z += dx;
//}


//bool Object::IsInside(const Point& p, const float3& min, const float3& max)
//{
//	bool isinside = false;
//	
//	if (p(0) >= min.x && p(0) <= max.x)
//		isinside = true;
//	
//	if (p(1) >= min.y && p(1) <= max.y)
//		isinside = true;
//	
//	if (p(2) >= min.z && p(2) <= max.z)
//		isinside = true;
//	
//	return isinside;
//}


//bool Object::IsInside(const Point& p, const double dx)
//{
//	return false;
//}


//void Object::Unfill(PointVect& points, const double dx)
//{
//	PointVect obj_points;
//	obj_points.reserve(1000);
//	this->Fill(obj_points, dx);
//	
//	PointVect new_points;
//	new_points.reserve(points.size());
//	
//	float3 min, max;
//	GetBoundingBox(obj_points, dx, min, max);
//	
//	for (int i = 0; i < points.size(); i++) {
//		const Point & p = points[i];
//		bool pushback = true;
//		
//		if (IsInside(p, min, max)) {
//			pushback = false;
//			
//			double mindist  = dx;
//			for (int j = 0; j < obj_points.size(); j++) {
//				double dist = p.Dist(obj_points[j]);
//				mindist = dist < mindist ? dist : mindist;
//			}
//			
//			if (mindist >= dx)
//				pushback = true;
//		}
//		
//		if (pushback)
//			new_points.push_back(p);
//	}
//	
//	obj_points.clear();
//	points.clear();
//	
//	points = new_points;
//}

/// Remove particles from particle vector 
/*! Remove the particles of particles vector lying inside the object.
 *  This method used IsInside(). 
 *
 *	\param points : particle vector
 *	\param dx : particle spacing
 */
void Object::Unfill(PointVect& points, const double dx) const
{
	PointVect new_points;
	new_points.reserve(points.size());
	
	for (int i = 0; i < points.size(); i++) {
		const Point & p = points[i];
		
		if (!IsInside(p, dx))
			new_points.push_back(p);
	}
	
	points.clear();
	
	points = new_points;
}