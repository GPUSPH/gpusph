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

#include <cstdio>
#include <cstdlib>
#include <iostream>

// for smart pointers
#include <memory>

#include "chrono_select.opt"
#if USE_CHRONO == 1
#include "chrono/physics/ChBodyEasy.h"
#endif

#include "Cube.h"
#include "Rect.h"

using namespace std;

/// Empty constructor
/*! Return a cube with all class variables set to 0.
 */
Cube::Cube(void):m_lx(0), m_ly(0), m_lz(0)
{
	m_origin = Point(0, 0, 0);
	m_vx = Vector(0, 0, 0);
	m_vy = Vector(0, 0, 0);
	m_vz = Vector(0, 0, 0);
	m_ep = EulerParameters();
}


/// Constructor from edges length and optional orientation (Euler paramaters)
/*! Construct a cube of given dimension with an orientation given by
 *  Euler parameters.
 *	lx, ly, lz parameters are the dimension of the cube along the X', Y'
 *	and Z' axis.
 *	\param origin : cube origin (bottom left corner)
 *	\param lx : length along X' axis
 *	\param ly : length along Y' axis
 *	\param lz : length along Z' axis
 *	\param ep : (optional) Euler parameters defining the orientation
 *
 *  Beware, particle mass should be set before any filling operation.
 *
 *  If the orientation is not specified, it is assumed that the local axes are parallel to the system one.
 */
Cube::Cube(const Point &origin, const double lx, const double ly, const double lz, const EulerParameters &ep)
{
	m_origin = origin;

	m_lx = lx;
	m_ly = ly;
	m_lz = lz;

	// set the EulerParameters and update sizes, center
	setEulerParameters(ep);
}

/// DEPRECATED Constructor from edge vectors
/*! Construct a cube according to 3 vectors defining the edges along
 *  X', Y' and Z' axis.
 *  Those three vectors should be orthogonal in pairs. This method is
 *  for compatibility only and should not be used.
 *	\param origin : cube origin (bottom left corner)
 *	\param vx : vector representing the edge along X'
 *	\param vy : vector representing the edge along Y'
 *	\param vz : vector representing the edge along Z'
 *
 *  Beware, particle mass should be set before any filling operation
 *
 *  This method is deprecated, use the constructor with EulerParameters instead
 */
Cube::Cube(const Point& origin, const Vector& vx, const Vector& vy, const Vector& vz)
{
	// Check if the three vectors are orthogonals in pairs
	if (fabs(vx*vy) > 1e-6*vx.norm()*vy.norm() || fabs(vx*vz) > 1e-6*vx.norm()*vz.norm()
		|| fabs(vy*vz) > 1e-6*vy.norm()*vz.norm()) {
		throw runtime_error("Trying to construct a cube with non perpendicular vectors\n");
		exit(1);
	}

	m_origin = origin;
	m_vx = vx;
	m_vy = vy;
	m_vz = vz;

	// Computing edge length
	m_lx = m_vx.norm();
	m_ly = m_vy.norm();
	m_lz = m_vz.norm();

	// Computing the center of gravity of the cube
	m_center = m_origin + 0.5*(m_vx + m_vy + m_vz);

	// Compute the rotation matrix from the orientation of the
	// global reference system
	Vector axis;
	double mat[9];
	mat[0] = m_vx(0)/m_lx;
	mat[3] = m_vx(1)/m_lx;
	mat[6] = m_vx(2)/m_lx;
	mat[1] = m_vy(0)/m_ly;
	mat[4] = m_vy(1)/m_ly;
	mat[7] = m_vy(2)/m_ly;
	mat[2] = m_vz(0)/m_lz;
	mat[5] = m_vz(1)/m_lz;
	mat[8] = m_vz(2)/m_lz;

	double trace = mat[0] + mat[4] + mat[8];
	double cs = 0.5*(trace - 1.0);
	double angle = acos(cs);  // in [0,PI]

	if (angle > 0.0)
	{
		if (angle < M_PI)
		{
			axis(0) = mat[7] - mat[5];
			axis(1) = mat[2] - mat[6];
			axis(2) = mat[3] - mat[1];
			axis /= axis.norm();
		}
		else
		{
			// angle is PI
			double halfInverse;
			if (mat[0] >= mat[4])
			{
				// r00 >= r11
				if (mat[0] >= mat[8])
				{
					// r00 is maximum diagonal term
					axis(0) = 0.5*sqrt(1.0 + mat[0] - mat[4] - mat[8]);
					halfInverse = 0.5/axis(0);
					axis(1) = halfInverse*mat[1];
					axis(2) = halfInverse*mat[2];
				}
				else
				{
					// r22 is maximum diagonal term
					axis(2) = 0.5*sqrt(1.0 + mat[8] - mat[0] - mat[4]);
					halfInverse = 0.5/axis(2);
					axis(0) = halfInverse*mat[2];
					axis(1) = halfInverse*mat[5];
				}
			}
			else
			{
				// r11 > r00
				if (mat[4] >= mat[8])
				{
					// r11 is maximum diagonal term
					axis(1) = 0.5*sqrt(1.0 + mat[4] - mat[0] - mat[8]);
					halfInverse  = 0.5/axis(1);
					axis(0) = halfInverse*mat[1];
					axis(2) = halfInverse*mat[5];
				}
				else
				{
					// r22 is maximum diagonal term
					axis(2) = 0.5*sqrt(1.0 + mat[8] - mat[0] - mat[4]);
					halfInverse = 0.5/axis(2);
					axis(0) = halfInverse*mat[2];
					axis(1) = halfInverse*mat[5];
				}
			}
		}
	}
	else
	{
		// The angle is 0 and the matrix is the identity.  Any axis will
		// work, so just use the x-axis.
		axis(0) = 1.0;
		axis(1) = 0.0;
		axis(2) = 0.0;
	}

	m_ep = EulerParameters(axis, angle);
	m_ep.ComputeRot();
}


/// Compute the volume of the cube
/*! The volume of the cube depends of the particle spacing used
 *  for the filling operation. With particles having their center
 *  of gravity on the surface of the cube the effective volume
 *  occupied by the object is \f$ V = (l_x + dx)(l_y + dx)(l_z + dx) \f$
 *	\param dx : particle spacing
 *	\return the volume of the cube
 */
double
Cube::Volume(const double dx) const
{
	const double lx = m_lx + dx;
	const double ly = m_ly + dx;
	const double lz = m_lz + dx;
	const double volume = lx*ly*lz;
	return volume;
}


/// Compute the principal moment of inertia
/*! Exactly like the volume the inertia tensor depends on
 *	particle spacing used for filling. This method computes
 *	the principal moments of inertia (aka the inertia tensor
 *	in the the principal axes reference frame) :
 *
 *  \f$ I_x = \frac{m}{12}\left({(l_y + dx)}^2 + {(l_z + dx)}^2\right) \f$
 *
 *  \f$ I_y = \frac{m}{12}\left({(l_x + dx)}^2 + {(l_z + dx)}^2\right) \f$
 *
 *  \f$ I_z = \frac{m}{12}\left({(l_x + dx)}^2 + {(l_y + dx)}^2\right) \f$
 *	\param dx : particle spacing
 *
 *	Obviously the mass of the body should be set before calling this method.
 */
void
Cube::SetInertia(const double dx)
{
	const double lx = m_lx + dx;
	const double ly = m_ly + dx;
	const double lz = m_lz + dx;
	m_inertia[0] = m_mass/12.0*(ly*ly + lz*lz);
	m_inertia[1] = m_mass/12.0*(lx*lx + lz*lz);
	m_inertia[2] = m_mass/12.0*(lx*lx + ly*ly);
}


/// Fill the surface of the cube with particles
/* TODO: comment
 */
void
Cube::FillBorder(PointVect& bpoints, PointVect& belems, PointVect& vpoints,
		vector<uint4>& vindexes, const double dx, const bool fill_top_face)
{
	Point   rorigin;
	Vector  rvx, rvy;
	vector<uint> edgeparts[6][4];
	m_origin(3) = m_center(3);
	int last_face = 6;

	if (!fill_top_face)
		last_face --;
	for (int face_num = 0; face_num < last_face; face_num++) {
		switch(face_num){
			case 0:
				rorigin = m_origin;
				rvx = m_vx;
				rvy = m_vz;
				break;
			case 1:
				rorigin = m_origin + m_vx;
				rvx = m_vy;
				rvy = m_vz;
				break;
			case 2:
				rorigin = m_origin + m_vx + m_vy;
				rvx = -m_vx;
				rvy = m_vz;
				break;
			case 3:
				rorigin = m_origin + m_vy;
				rvx = -m_vy;
				rvy = m_vz;
				break;
			case 4:
				rorigin = m_origin;
				rvx = m_vx;
				rvy = m_vy;
				break;
			case 5:
				rorigin = m_origin + m_vz;
				rvx = m_vx;
				rvy = m_vy;
				break;
		}

		Rect rect = Rect(rorigin, rvx, rvy);
		rect.Fill(bpoints, belems, vpoints, vindexes, dx, face_num, edgeparts);
	}
}


/// Fill a given face of the cube with particles
/* Fill a given face of the cube with particles with a given
 * particle spacing. For the selected face the edges are filled
 * according to the edges_to_fill array of booleans.
 *	\param points : vector where the particles will be added
 *	\param dx : particle spacing
 *	\param face_num : number of face to fill
 *	\param edges_to_fill : edges to be filled
 */
void
Cube::FillBorder(PointVect& points, const double dx, const int face_num, const bool *edges_to_fill)
{
	Point   rorigin;
	Vector  rvx, rvy;

	m_origin(3) = m_center(3);

	switch(face_num){
		case 0:
			rorigin = m_origin;
			rvx = m_vx;
			rvy = m_vz;
			break;
		case 1:
			rorigin = m_origin + m_vx;
			rvx = m_vy;
			rvy = m_vz;
			break;
		case 2:
			rorigin = m_origin + m_vx + m_vy;
			rvx = -m_vx;
			rvy = m_vz;
			break;
		case 3:
			rorigin = m_origin + m_vy;
			rvx = -m_vy;
			rvy = m_vz;
			break;
		case 4:
			rorigin = m_origin;
			rvx = m_vx;
			rvy = m_vy;
			break;
		case 5:
			rorigin = m_origin + m_vz;
			rvx = m_vx;
			rvy = m_vy;
			break;
	}

	Rect rect = Rect(rorigin, rvx, rvy);
	rect.Fill(points, dx, edges_to_fill);
}


/// Fill the surface of the cube with particles
/* Fill the surface of the cube with particles except
 * eventually the top face.
 *	\param points : vector where the particles will be added
 *	\param dx : particle spacing
 *	\param fill_top_face : 1 the top face is filled, 0 is not
 */
void
Cube::FillBorder(PointVect& points, const double dx, const bool fill_top_face)
{
	m_origin(3) = m_center(3);

	bool edges_to_fill[6][4] =
		{   {true, true, true, true},
			{true, false, true, false},
			{true, true, true, true},
			{true, false, true, false},
			{false, false, false, false},
			{false, false, false, false} };

	int last_face = 6;
	if (!fill_top_face)
		last_face --;
	for (int face_num = 0; face_num < last_face; face_num++)
			FillBorder(points, dx, face_num, edges_to_fill[face_num]);
}


/// Fill the cube with particles
/* Fill the whole cube with particles with a given
 * particle spacing. If fill_faces is false only the inne
 * part off the cube (i.e the cube excluding the faces) is
 * filled with particles.
 *	\param points : vector where the particles will be added
 *	\param dx : particle spacing
 *	\param fill_faces : if true fill the cube including faces
 *	\param fill : if true add the particles to points otherwise just
 *				count the number of particles
 *	\return the number of particles used in the fill
 */
int
Cube::Fill(PointVect& points, const double dx, const bool fill_faces, const bool fill)
{
	m_origin(3) = m_center(3);
	int nparts = 0;

	const int nx = (int) (m_lx/dx);
	const int ny = (int) (m_ly/dx);
	const int nz = (int) (m_lz/dx);

	int startx = 0;
	int starty = 0;
	int startz = 0;
	int endx = nx;
	int endy = ny;
	int endz = nz;

	if (!fill_faces){
		startx++;
		starty++;
		startz++;
		endx --;
		endy --;
		endz --;
	}

	for (int i = startx; i <= endx; i++)
		for (int j = starty; j <= endy; j++)
			for (int k = startz; k <= endz; k++) {
				Point p = m_origin + i/((double) nx)*m_vx + j/((double) ny)*m_vy + k/((double) nz)*m_vz;
				if (fill)
					points.push_back(p);
				nparts ++;
			}
	return nparts;
}


/// Fill the inner part of the cube, starting at dx/2 from the boundary
/* Fill the  inner part of the cube (i.e the cube excluding
 * the faces) with particles. In contrast to Fill() without faces,
 * the filling starts at dx/2 from the boundary.
 *	\param points : vector where the particles will be added
 *	\param dx : particle spacing
 *	TODO FIXME different filling methods should be implemented
 *	in a more general way for all objects.
 */
void
Cube::InnerFill(PointVect& points, const double dx)
{
	m_origin(3) = m_center(3);
	const int nx = (int) (m_lx/dx);
	const int ny = (int) (m_ly/dx);
	const int nz = (int) (m_lz/dx);

	int startx = 0;
	int starty = 0;
	int startz = 0;
	int endx = nx;
	int endy = ny;
	int endz = nz;

	for (int i = startx; i < endx; i++)
		for (int j = starty; j < endy; j++)
			for (int k = startz; k < endz; k++) {
				Point p = m_origin + (i + 0.5)*m_vx/nx + (j + 0.5)*m_vy/ny + (k + 0.5)*m_vz/nz;
				points.push_back(p);
			}
	return;
}


/// Fill layers of particles outside of the cube, starting at dx/2 from the boundary
/* Fill the outside of the cube (i.e the cube excluding
 * the faces) with a fixed number of particle layers starting at
 * dx/2 from the boundary.
 *	\param points : vector where the particles will be added
 *	\param dx : particle spacing
 *	\param layers : number of layers to fill
 *	\param fill_top: fill the top
 */
void
Cube::FillOut(PointVect& points, const double dx, const int layers, const bool fill_top)
{
	m_origin(3) = m_center(3);
	const int nx = (int) (m_lx/dx);
	const int ny = (int) (m_ly/dx);
	const int nz = (int) (m_lz/dx);

	// Bottom face
	for (int i = -layers; i < nx + layers; i++)
		for (int j = -layers; j < ny + layers; j++)
			for (int k = -layers; k < 0; k++) {
				Point p = m_origin + (i + 0.5)*m_vx/nx + (j + 0.5)*m_vy/ny + (k + 0.5)*m_vz/nz;
				points.push_back(p);
			}

	// Lateral faces
	for (int i = -layers; i < nx + layers; i++) {
		for (int j = -layers; j < 0; j++)
			for (int k = 0; k < nz; k++) {
				Point p = m_origin + (i + 0.5)*m_vx/nx + (j + 0.5)*m_vy/ny + (k + 0.5)*m_vz/nz;
				points.push_back(p);
			}
		for (int j = ny; j < ny + layers; j++)
			for (int k = 0; k < nz; k++) {
				Point p = m_origin + (i + 0.5)*m_vx/nx + (j + 0.5)*m_vy/ny + (k + 0.5)*m_vz/nz;
				points.push_back(p);
			}
	}

	for (int j = -layers; j < ny + layers; j++) {
		for (int i = -layers; i < 0; i++)
			for (int k = 0; k < nz; k++) {
				Point p = m_origin + (i + 0.5)*m_vx/nx + (j + 0.5)*m_vy/ny + (k + 0.5)*m_vz/nz;
				points.push_back(p);
			}
		for (int i = nx; i < nx + layers; i++)
			for (int k = 0; k < nz; k++) {
				Point p = m_origin + (i + 0.5)*m_vx/nx + (j + 0.5)*m_vy/ny + (k + 0.5)*m_vz/nz;
				points.push_back(p);
			}
	}

	// Top face
	if (!fill_top)
		return;

	for (int i = -layers; i < nx + layers; i++)
		for (int j = -layers; j < ny + layers; j++)
			for (int k = nz; k < nz + layers; k++) {
				Point p = m_origin + (i + 0.5)*m_vx/nx + (j + 0.5)*m_vy/ny + (k + 0.5)*m_vz/nz;
				points.push_back(p);
			}
}


/// Fill the cube with layers of particles staring from surface
/* Fill the cube with layers of particles from the surface to
 * the inside of the cube.
 *	\param points : vector where the particles will be added
 *	\param dx : particle spacing
 *	\param layers : number of internal layers to add
 */
void
Cube::FillIn(PointVect& points, const double dx, const int layers)
{
	FillIn(points, dx, layers, true);
}


/// Fill the cube with layers of particles staring from surface
/* Fill the cube with layers of particles from the surface to
 * the inside of the cube and eventually excluding the top face.
 *	\param points : vector where the particles will be added
 *	\param dx : particle spacing
 *	\param layers : number of internal layers to add
 *	\param fill_top : if true fill also the top face
 */
void
Cube::FillIn(PointVect& points, const double dx, const int _layers, const bool fill_top)
{
	// NOTE - TODO
	// XProblem calls FillIn with negative number of layers to fill rects in the opposite
	// direction as the normal. Cubes and other primitives do not support it. This is a
	// temporary workaround until we decide a common policy for the filling of DYNAMIC
	// boundary layers consistent for any geometry.
	int layers = abs(_layers);

	m_origin(3) = m_center(3);
	const int nx = (int) (m_lx/dx);
	const int ny = (int) (m_ly/dx);
	const int nz = (int) (m_lz/dx);

	// we will have two ranges in each direction:
	// [0, layers[ , ]n - layers, n], except when
	// n <= 2*layers, the two ranges intersect and reduce to
	// [0, n]
	const int xplus_range[] = {0,
		nx <= 2*layers ? nx : layers - 1 };
	const int xminus_range[] = {
		nx <= 2*layers ? INT_MAX : nx - layers + 1,
		nx };

	const int yplus_range[] = {0,
		ny <= 2*layers ? ny : layers - 1 };
	const int yminus_range[] = {
		ny <= 2*layers ? INT_MAX : ny - layers + 1,
		ny };

	const int zplus_range[] = {0,
		nz <= 2*layers ? nz : layers - 1 };
	const int zminus_range[] = {
		nz <= 2*layers ? INT_MAX : nz - layers + 1,
		nz };

	// top and bottom layers
	for (int i = 0; i <= nx; ++i) {
		for (int j = 0; j <= ny; ++j) {
			for (int k = zplus_range[0]; k <= zplus_range[1]; ++k) {
				Point p = m_origin + i/((double) nx)*m_vx + j/((double) ny)*m_vy + k/((double) nz)*m_vz;
				points.push_back(p);
			}
			for (int k = zminus_range[0]; k <= zminus_range[1]; ++k) {
				Point p = m_origin + i/((double) nx)*m_vx + j/((double) ny)*m_vy + k/((double) nz)*m_vz;
				points.push_back(p);
			}
		}
	}

	// front and back face: to avoid overlapping particles,
	// we do it only for the gaps between the two z ranges —if there were two ranges!
	if (zminus_range[0] == INT_MAX) return;

	for (int k = zplus_range[1] + 1; k < zminus_range[0] ; ++k) {
		for (int i = 0; i <= nx; ++i) {
			for (int j = yplus_range[0]; j <= yplus_range[1]; ++j) {
				Point p = m_origin + i/((double) nx)*m_vx + j/((double) ny)*m_vy + k/((double) nz)*m_vz;
				points.push_back(p);
			}
			for (int j = yminus_range[0]; j <= yminus_range[1]; ++j) {
				Point p = m_origin + i/((double) nx)*m_vx + j/((double) ny)*m_vy + k/((double) nz)*m_vz;
				points.push_back(p);
			}
		}
		// side faces, if we have a y gap
		if (yminus_range[0] == INT_MAX) continue;
		for (int j = yplus_range[1] + 1; j < yminus_range[0] ; ++j) {
			for (int i = xplus_range[0]; i <= xplus_range[1]; ++i) {
				Point p = m_origin + i/((double) nx)*m_vx + j/((double) ny)*m_vy + k/((double) nz)*m_vz;
				points.push_back(p);
			}
			for (int i = xminus_range[0]; i <= xminus_range[1]; ++i) {
				Point p = m_origin + i/((double) nx)*m_vx + j/((double) ny)*m_vy + k/((double) nz)*m_vz;
				points.push_back(p);
			}
		}
	}

}


/// Check if a point is inside the cube
/* For a given point return true if the point is inside
 * the cube within a threshold of dx.
 *	\param p : point to test
 *	\param dx : threshold
 *	\return true if p is inside the cube within dx
 */
bool
Cube::IsInside(const Point& p, const double dx) const
{
	Point lp = m_ep.TransposeRot(p - m_origin);
	const double lx = m_lx + dx;
	const double ly = m_ly + dx;
	const double lz = m_lz + dx;
	bool inside = false;
	if (lp(0) > -dx && lp(0) < lx && lp(1) > -dx && lp(1) < ly &&
		lp(2) > -dx && lp(2) < lz )
		inside = true;

	return inside;
}

// set the given EulerParameters
void Cube::setEulerParameters(const EulerParameters &ep)
{
	m_ep = ep;
	// Before doing any rotation with Euler parameters the rotation matrix associated with
	// m_ep is computed.
	m_ep.ComputeRot();

	// Computing the edge vectors according to orientation
	m_vx = m_lx*m_ep.Rot(Vector(1, 0, 0));
	m_vy = m_ly*m_ep.Rot(Vector(0, 1, 0));
	m_vz = m_lz*m_ep.Rot(Vector(0, 0, 1));

	// Point mass is stored in the fourth component of m_center. Store and restore it after the rotation
	const double point_mass = m_center(3);
	// Computing the center of gravity of the cube
	m_center = m_origin + 0.5*m_ep.Rot(Vector(m_lx, m_ly, m_lz));
	m_center(3) = point_mass;
}

// get the object bounding box
void Cube::getBoundingBox(Point &output_min, Point &output_max)
{
	getBoundingBoxOfCube(output_min, output_max, m_origin,
		m_vx, m_vy, m_vz);
}

void Cube::shift(const double3 &offset)
{
	const Point poff = Point(offset);
	m_origin += poff;
	m_center += poff;
}


ostream& operator<<(ostream& out, const Cube& cube) // output
{
    out << "Cube size(" << cube.m_lx << ", " << cube.m_ly << ", " <<cube. m_lz
    		<< ") particles: " << cube.m_parts.size() << "\n";
    return out;
}


#if USE_CHRONO == 1
/* Create a Chrono box body.
 *	\param dx : particle spacing
 */
void
Cube::BodyCreate(::chrono::ChSystem * bodies_physical_system, const double dx, const bool collide,
	const ::chrono::ChQuaternion<> & orientation_diff)
{
	// Check if the physical system is valid
	if (!bodies_physical_system)
		throw std::runtime_error("Cube::BodyCreate Trying to create a body in an invalid physical system!\n");

	// Creating a new Chrono object
	m_body = std::make_shared< ::chrono::ChBodyEasyBox > ( m_lx + dx, m_ly + dx, m_lz + dx, m_mass/Volume(dx), collide );
	m_body->SetPos(::chrono::ChVector<>(m_center(0), m_center(1), m_center(2)));
	m_body->SetRot(orientation_diff*m_ep.ToChQuaternion());

	m_body->SetCollide(collide);
	m_body->SetBodyFixed(m_isFixed);
	// mass is automatically set according to density

	// Add the body to the physical system
	bodies_physical_system->AddBody(m_body);
}
#endif
