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

// for smart pointers
#include <memory>

#include "chrono_select.opt"
#if USE_CHRONO == 1
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/fea/ChElementHexa_8.h"
#include "chrono/fea/ChElementShellANCF.h"
#include "chrono/fea/ChElementCableANCF.h"
#include "chrono/fea/ChNodeFEAxyz.h"
#include "chrono/fea/ChNodeFEAxyzD.h"
#endif

#include "Cylinder.h"

using namespace std;

Cylinder::Cylinder(void)
{
	m_center = Point(0,0,0);
	m_h = 1;
	m_r = 1;
	m_ep = EulerParameters();
}


Cylinder::Cylinder(const Point& origin, const double radius, const Vector& height)
{
	m_origin = origin;
	m_center = m_origin + 0.5*height;
	m_r = radius;
	m_ri = 0;
	m_h = height.norm();

	Vector v(0, 0, 1);
	const double angle = acos(height*v/m_h);
	Vector rotdir = -height.cross(v);
	if (rotdir.norm() == 0)
		rotdir = Vector(0, 1, 0);

	m_ep = EulerParameters(rotdir, angle);
	m_ep.ComputeRot();
}


Cylinder::Cylinder(const Point& origin, const double radius, const double inner_radius, const double height, uint nelst, uint nelsc, uint nelsh, const EulerParameters& ep)
{
	m_origin = origin;
	m_h = height;
	m_r = radius;
	m_ri = inner_radius;

	m_nels = make_uint3(nelst, nelsc, nelsh);

	setEulerParameters(ep);
}


Cylinder::Cylinder(const Point& origin, const Vector& radius, const Vector& height)
{
	if (fabs(radius*height) > 1e-8*radius.norm()*height.norm())
		throw std::invalid_argument("Trying to construct a cylinder with non perpendicular radius and axis");
	m_origin = origin;
	m_center = m_origin + 0.5*height;
	m_r = radius.norm();
	m_h = height.norm();

	Vector v(0, 0, 1);
	const double angle = acos(height*v/m_h);
	Vector rotdir = height.cross(v);
	if (rotdir.norm() == 0)
		rotdir = Vector(0, 1, 0);

	m_ep = EulerParameters(rotdir, angle);
	m_ep.ComputeRot();
}


double
Cylinder::Volume(const double dx) const
{
	const double r = m_r + dx/2.0;
	const double h = m_h + dx;
	const double volume = M_PI*r*r*h;
	return volume;
}


void
Cylinder::SetInertia(const double dx)
{
	const double r = m_r + dx/2.0;
	const double h = m_h + dx;
	m_inertia[0] = m_mass/12.0*(3*r*r + h*h);
	m_inertia[1] = m_inertia[0];
	m_inertia[2] = m_mass/2.0*r*r;
}

void Cylinder::setEulerParameters(const EulerParameters &ep)
{
	m_ep = ep;
	m_ep.ComputeRot();

	// Point mass is stored in the fourth component of m_center. Store and restore it after the rotation
	const double point_mass = m_center(3);
	m_center = m_origin + m_ep.Rot(0.5*m_h*Vector(0, 0, 1));
	m_center(3) = point_mass;
}

void Cylinder::getBoundingBox(Point &output_min, Point &output_max)
{
	Point corner_origin = m_origin - Vector( -m_r, -m_r, 0.0 );
	getBoundingBoxOfCube(output_min, output_max, corner_origin,
		Vector(2*m_r, 0, 0), Vector(0, 2*m_r, 0), Vector(0, 0, m_h) );
}

void Cylinder::shift(const double3 &offset)
{
	const Point poff = Point(offset);
	m_origin += poff;
	m_center += poff;
}


void
Cylinder::FillBorder(PointVect& points, const double dx, const bool bottom, const bool top)
{
	m_origin(3) = m_center(3);
	const int nz = (int) ceil(m_h/dx);
	const double dz = m_h/nz;
	for (int i = 0; i <= nz; i++)
		FillDiskBorder(points, m_ep, m_origin, m_r, i*dz, dx, 2.0*M_PI*rand()/RAND_MAX);
	if (bottom)
		FillDisk(points, m_ep, m_origin, m_r - dx, 0.0, dx, true);
	if (top)
		FillDisk(points, m_ep, m_origin, m_r - dx, nz*dz, dx, true);
}


int
Cylinder::Fill(PointVect& points, const double dx, const bool fill)
{
	m_origin(3) = m_center(3);
	int nparts = 0;
	const int nz = (int) ceil(m_h/dx);
	const double dz = m_h/nz;
	for (int i = 0; i <= nz; i++)
		nparts += FillDisk(points, m_ep, m_origin, m_r, i*dz, dx, fill);

	return nparts;
}

void
Cylinder::FillIn(PointVect& points, const double dx, const int layers)
{
	FillIn(points, dx, layers, true);
}


void
Cylinder::FillIn(PointVect& points, const double dx, const int _layers, const bool fill_tops)
{
	// NOTE - TODO
	// XProblem calls FillIn with negative number of layers to fill rects in the opposite
	// direction as the normal. Cubes and other primitives do not support it. This is a
	// temporary workaround until we decide a common policy for the filling of DYNAMIC
	// boundary layers consistent for any geometry.
	uint layers = abs(_layers);

	m_origin(3) = m_center(3);

	if (layers*dx > m_r) {
		std::cerr << "WARNING: Cylinder FillIn with " << layers << " layers and " << dx << " stepping > radius " << m_r << " replaced by Fill" << std::endl;
		Fill(points, dx, true);
		return;
	}

	if (2*layers*dx > m_h) {
		std::cerr << "WARNING: Cylinder FillIn with " << layers << " layers and " << dx << " stepping > half-height " << (m_h/2) << " replaced by Fill" << std::endl;
		Fill(points, dx, true);
		return;
	}


	for (uint l = 0; l < layers; l++) {

		const double smaller_r = m_r - l * dx;
		const double smaller_h = m_h - l * 2 * dx;

		const int nz = (int) ceil(smaller_h/dx);
		const double dz = smaller_h/nz;
		for (int i = 0; i <= nz; i++)
			FillDiskBorder(points, m_ep, m_origin, smaller_r, i*dz, dx, 2.0*M_PI*rand()/RAND_MAX);
		// fill "bottom"
		if (fill_tops)
			FillDisk(points, m_ep, m_origin, smaller_r - dx, l * dx, dx, true);
		// fill "top"
		if (fill_tops)
			FillDisk(points, m_ep, m_origin, smaller_r - dx, nz*dz + l * dx, dx, true);
	}
	return;
}

bool
Cylinder::IsInside(const Point& p, const double dx) const
{
	Point lp = m_ep.TransposeRot(p - m_origin);
	const double r = m_r + dx;
	const double h = m_h + dx;
	bool inside = false;
	if (lp(0)*lp(0) + lp(1)*lp(1) < r*r && lp(2) > - dx && lp(2) < h)
		inside = true;

	return inside;
}

#if USE_CHRONO == 1
/* Create a cube Chrono body inside a specified Chrono physical system. If
 * collide is true this method also enables collision detection in Chrono.
 * Here we have to specialize this function for the Cylinder because the Chrono cylinder
 * is by default in the Y direction and ours in the Z direction.
 *	\param bodies_physical_system : Chrono physical system
 *	\param dx : particle spacing
 *	\param collide : add collision handling
 */
void
Cylinder::BodyCreate(::chrono::ChSystem * bodies_physical_system, const double dx, const bool collide,
	const ::chrono::ChQuaternion<> & orientation_diff)
{
	// Check if the physical system is valid
	if (!bodies_physical_system)
		throw std::runtime_error("Cube::BodyCreate Trying to create a body in an invalid physical system!\n");

	// Creating a new Chrono object
	m_body = std::make_shared< ::chrono::ChBodyEasyCylinder >( m_r + dx/2.0, m_h + dx, m_mass/Volume(dx), collide );
	m_body->SetPos(::chrono::ChVector<>(m_center(0), m_center(1), m_center(2)));
	m_body->SetRot(orientation_diff*m_ep.ToChQuaternion());

	m_body->SetCollide(collide);
	m_body->SetBodyFixed(m_isFixed);
	// mass is automatically set according to density

	// Add the body to the physical system
	bodies_physical_system->AddBody(m_body);
}


/*Returns the nodes associated to the element that a given point is contained in.
 * The obtained nodes are used to move deformable particles and to get forces from the SPH system, using shaping functions
*/
int4 Cylinder::getOwningNodes(const double4 abs_coords)
{
	// get relative position of the point inside the cylinder
	double4 rel_coords = make_double4(abs_coords.x - m_origin(0), abs_coords.y - m_origin(1), abs_coords.z - m_origin(2), 0);

	//get cylindrical coordinates
	double alpha = atan2(rel_coords.y, rel_coords.x);
	double rad = sqrt(rel_coords.x*rel_coords.x + rel_coords.y*rel_coords.y);

	if (alpha < 0.0)
		alpha += M_PI*2.0;


	double3 cyl_coords = make_double3(rad, alpha, rel_coords.z);
	cyl_coords.x = m_r - cyl_coords.x; // start from the cells

	// get size of the elements
	double dx = (m_r - m_ri)/m_nels.x;
	double dy = 2*M_PI/m_nels.y;
	double dz = m_h/m_nels.z;

	/*IMPORTANT: the following works in association to the order by which the elements are created*/
	// the order is in the directions x, then y and then z.

	// for the cell recognition we subtract a small quantity ( half dp would be enough,
	// but we should pass dt here) so the last layer of a cell is still considered to 
	// belong to the closer contiguous element. This works as long as dp is larger than
	// machine epsilon for double
	double3 cyl_coordsc = cyl_coords - make_double3(DBL_EPSILON, DBL_EPSILON, DBL_EPSILON);

	// ... and we take every value with positive sign
	cyl_coordsc.x = abs(cyl_coordsc.x);
	cyl_coordsc.y = abs(cyl_coordsc.y);
	cyl_coordsc.z = abs(cyl_coordsc.z);

	//number of nodes per side
	uint3 nnodes = m_nels + make_uint3(0, 0, 1);

	// get the local index of the first node associated to the element
	int node_index =  floor(cyl_coordsc.z/dz);


	// To handle different types of elements without changing shaping functions, we
	// always consider four nodes associated to the elements. In case of beam element 
	// we associate a pair of these nodes to each actual node of the element, and
	// quantities referred to each of the element nodes are distributed between the
	// two associated nodes. (We can see the beam as a shell element with zero width)
	// Then when identified the two nodes associated to the element we send them two times.

	// TODO we should find a better way to store nodes. e.g, if we pass n_in_layer the last two nodes are obtained
	// from the previous ones. Ans so on, we could pass just one node, but this means to have more computation at runtime.

	int NA = node_index;
	int NC = NA;

	int NE = NA + 1; // the two nodes in the beam element are consecutive
	int NG = NE;

	// return the offset of the nodes with respect to the first node of the geometry.
	// This will be added to the global index of the first node to get the global index
	// of the nodes. The offset is negative when reusing nodes previously created.
	NA = m_fea_nodes_offset[NA];
	NC = m_fea_nodes_offset[NC];
	NE = m_fea_nodes_offset[NE];
	NG = m_fea_nodes_offset[NG];


	return make_int4(NA, NC, NE, NG);
}

// Get natural coordinates of a point, inside the geometry, with respect to the element the point is associated to.
// The associated element is recalled by means of its nodes using the function getOwningNodes
float4 Cylinder::getNaturalCoords(const double4 abs_coords)
{
	// get relative position of the point inside the geometry
	double4 rel_coords = make_double4(abs_coords.x - m_origin(0), abs_coords.y - m_origin(1), abs_coords.z - m_origin(2), 0);

	//get cylindrical coordinates

	double alpha = atan2(rel_coords.y, rel_coords.x);
	double rad = sqrt(rel_coords.x*rel_coords.x + rel_coords.y*rel_coords.y);

	if (alpha < 0.0)
		alpha += M_PI*2.0;

	double3 cyl_coords = make_double3(rad, alpha, rel_coords.z);

	// get size of the elements
	double dx = (m_r - m_ri)/m_nels.x;
	double dy = 2*M_PI/m_nels.y;
	double dz = m_h/m_nels.z;

	/*IMPORTANT: the following works in association to the order by which the elements are created*/
	// the order is in the directions x, then y and then z.

	// for the cell recognition we subtract a small quantity ( half dp would be enough, but we should pass dt here)
	// so the last layer of a cell is still considered to belong to the closer contiguous element.
	// This works as long as dp is larger than machine epsilon for double
	double3 cyl_coordsc = cyl_coords - make_double3(DBL_EPSILON, DBL_EPSILON, DBL_EPSILON);

	// ... and we take every value with positive sign
	cyl_coordsc.x = abs(cyl_coordsc.x);
	cyl_coordsc.y = abs(cyl_coordsc.y);
	cyl_coordsc.z = abs(cyl_coordsc.z);

	// natural coordinates have origin in the center of the element
	float nat_coord_x =  (m_r - cyl_coords.x - floor((m_r - cyl_coordsc.x)/dx)*dx - dx*0.5)/(dx*0.5);
	float nat_coord_y =  (cyl_coords.y - floor(cyl_coordsc.y/dy)*dy - dy*0.5)/(dy*0.5);
	float nat_coord_z =  (cyl_coords.z - floor(cyl_coordsc.z/dz)*dz - dz*0.5)/(dz*0.5);

	// to use shaping functions we need to know what is the type of element we are referring to: we use
	// code 0 for shell elements (NOTE: this choice is functional in the use of the shaping functions)
	const int el_type_id = 0;

	return make_float4(nat_coord_x, nat_coord_y, nat_coord_z, el_type_id);
}


/*Build a Mesh of ANCF cables elements to discretize a Cylinder*/
void
Cylinder::CreateFemMesh(::chrono::ChSystem *fea_system)
{
	if (!fea_system)
		throw std::runtime_error("Cylinder::CreateFEMMesh: Trying to create a body in an invalid physical system!\n");

	// to keep track of the global index for the nodes that we are going to create,
	// let us set compute the number of nodes already created for previous geometries
	set_previous_nodes_num(fea_system);

	cout << "Creating ANCF cables FEM mesh for Cylinder" << endl;

	// create a new Chrono mesh associated to this geometry
	m_fea_mesh = std::make_shared<::chrono::fea::ChMesh>();

	// vector that will store the New nodes created for this mesh
	std::vector<std::shared_ptr<::chrono::fea::ChNodeFEAxyz>> nodes;

	// size of the elements that will constite the mesh
	const double lel_z = m_h/m_nels.z;

	const uint nodes_num = m_nels.z + 1;
	uint n_counter = 0;

	for (int j = 0; j <= m_nels.z; j++) {

		// Node postion
		Point coords = m_origin + Point(0, 0, lel_z*j);

		// Node direction
		double3 direction = make_double3(0, 0, 1);

		// create the node
		auto node = std::make_shared<::chrono::fea::ChNodeFEAxyzD>(
			::chrono::ChVector<>(coords(0), coords(1), coords(2)),
			::chrono::ChVector<>(direction.x, direction.y, direction.z));

		node->SetMass(0);

		// Check if the node would be in the place of a previously defined node
		// and in case use that one. This function can be used to join two meshes.
		// The nodes, new and reused, that compose the grid are stored in the vector "nodes"
		bool is_new = reduceNodes(node, fea_system, nodes);

		if (is_new) {
			m_fea_mesh->AddNode(node);
			m_fea_nodes.push_back(coords);

			n_counter ++;
		}

	}

	auto msection_cable = std::make_shared<::chrono::fea::ChBeamSectionCable>();
	msection_cable->SetDiameter(m_r);

	// We model a hollow cylinder by means of the momentum of inertia.
	// Compute and assign momentum of inertia of a hollow cylinder:
	double dext = 2*m_r;
	double dext4 = dext*dext*dext*dext;
	double dint = 2*m_ri;
	double dint4 = dint*dint*dint*dint;
	msection_cable->SetI(M_PI/64.0*(dext4 - dint4));

	msection_cable->SetYoungModulus(m_youngModulus);
	msection_cable->SetDensity(m_density);

	// Now we walk through the grid of nodes previously created and we apply elements
	uint n = -1;// nodes indices explorer

	for (int j = 0; j <= m_nels.z; j++) {

		n++;

		if (j == m_nels.z) continue;

		int NA = n;
		int NB = n + 1;

		// create the new element
		auto cable = std::make_shared<::chrono::fea::ChElementCableANCF>();

		// attach the element to the nodes
		cable->SetNodes(std::dynamic_pointer_cast<::chrono::fea::ChNodeFEAxyzD>(nodes[NA]),
			std::dynamic_pointer_cast<::chrono::fea::ChNodeFEAxyzD>(nodes[NB]));

		cable->SetSection(msection_cable);
		// set alpha damping TODO set from problem?
		cable->SetAlphaDamp(0.0035);

		// Add element to mesh
		m_fea_mesh->AddElement(cable);
	}
}

#endif
