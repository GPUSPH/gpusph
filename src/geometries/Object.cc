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

#include <cstdlib>

// For the automatic name determination
#include <typeinfo>
#include <cxxabi.h>

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

/// Set the Young's modulus of the object
/*! Set the object Young's modulus.
 * \param youngModulus: object youngModulus
 */
void
Object::SetYoungModulus(const double youngModulus)
{
	m_youngModulus = youngModulus;
}

/// Set the Poisson ratio of the object
/*! Set the object Poisson ratio.
 * \param poissonRatio: object poissonRatio 
 */
void
Object::SetPoissonRatio(const double poissonRatio)
{
	m_poissonRatio = poissonRatio;
}

/// Set the density of the object
/*! Set the object density.
 * \param density: object density 
 */
void
Object::SetDensity(const double density)
{
	m_density = density;
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

/// Set the object principal moments of inertia
/*! Directly set the object principal moments of inertia.
 *	\param i11 : element 1,1  of the inertia matrix
 *	\param i22 : element 2,2  of the inertia matrix
 *	\param i33 : element 3,3  of the inertia matrix
 */
void
Object::SetInertia(const double i11, const double i22, const double i33)
{
	m_inertia[0] = i11;
	m_inertia[1] = i22;
	m_inertia[2] = i33;
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

/// Return the particle vector associated to fea nodes 
/*! \return a reference to the particles vector associated with the fea nodes 
 */
PointVect&
Object::GetFeaNodes(void)
{
	return m_fea_nodes;
}

#if USE_CHRONO == 1
bool
Object::reduceNodes(std::shared_ptr<::chrono::fea::ChNodeFEAxyz> newNode,
	::chrono::ChSystem * fea_system,
	std::vector<std::shared_ptr<::chrono::fea::ChNodeFEAxyz>> & nodes)
{
	std::vector<std::shared_ptr<::chrono::fea::ChMesh>> mesh_list = fea_system->Get_meshlist();
	std::shared_ptr<::chrono::fea::ChNodeFEAxyz> chosen_node = newNode;
	bool is_new = true;

	/*The offset to the node to consider with respect to the first node of the geometry among all node positions for the geometry*/
	/*if the nodal position the node to refer to belongs to previously defined geomtries, then offset will be negative, otherwise if
	 * a new node is defined the offset wil be >= 0*/
	int offset = - m_previous_nodes;

	for (uint m = 0; m < mesh_list.size(); m++) {
		auto mesh = mesh_list[m];
		for (uint n = 0; n < mesh->GetNnodes(); n++) {
			auto node = std::dynamic_pointer_cast<::chrono::fea::ChNodeFEAxyzD>(mesh->GetNode(n));

			if ((abs(newNode->GetPos().x() - node->GetPos().x()) < FLT_EPSILON) &&
				(abs(newNode->GetPos().y() - node->GetPos().y()) < FLT_EPSILON) &&
				(abs(newNode->GetPos().z() - node->GetPos().z()) < FLT_EPSILON)) {

				std::cout << "reusing node" << std::endl;


				chosen_node = node;
				is_new = false;
				nodes.push_back(node);
				break;

			}

			offset ++;
		}
		if (!is_new)
			break;
	}

	if (is_new){
		offset = m_fea_nodes.size();
		nodes.push_back(newNode);
	}

	m_fea_nodes_offset.push_back(offset);

	return is_new;
}

void
Object::set_previous_nodes_num(::chrono::ChSystem * fea_system)
{
	std::vector<std::shared_ptr<::chrono::fea::ChMesh>> mesh_list = fea_system->Get_meshlist();

	for (uint m = 0; m < mesh_list.size(); m++) {
		auto mesh = mesh_list[m];
		m_previous_nodes += mesh->GetNnodes();
	}
	std::cout << "initialized m_previous_nodes = " << m_previous_nodes << std::endl;
}
#endif

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

/// Get the number of nodes of the mesh associated to the deformable body
uint Object::GetNumFeaNodes()
{
	if (m_numFeaNodes == 0)
		m_numFeaNodes = m_fea_nodes.size();
	//std::cout << "m_fea_nodes measured " << m_numFeaNodes << std::endl;

	return m_fea_nodes.size();
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
	return FillDisk(points, ep, center, 0, r, z, dx, fill);
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
	if (rmax < 0) throw std::invalid_argument("FillDisk with maximum radius lower than 0");
	if (rmin < 0) throw std::invalid_argument("FillDisk with minimum radius lower than 0");
	if (rmax < rmin) throw std::invalid_argument("FillDisk with maximum radius lower than minimum radius");
	const int nr = (int) ceil((rmax - rmin)/dx);
	const double dr = (nr==0)? 0 : (rmax - rmin)/nr;
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

#if USE_CHRONO == 1
uint Object::JoinFeaNodes(::chrono::ChSystem* ch_system, std::shared_ptr<::chrono::fea::ChMesh> fea_mesh, const double dx)
{
	std::shared_ptr<::chrono::fea::ChNodeFEAxyz> node;
	uint numnodes = fea_mesh->GetNnodes();

	uint nadded = 0;

	for (uint i = 0; i < numnodes; i++) {

		node = std::dynamic_pointer_cast<::chrono::fea::ChNodeFEAxyz>(fea_mesh->GetNode(i));
		if (!node) throw std::runtime_error("Error: impossible to read nodes in JointFeaNode");

		Point ncords;

		ncords(0) = node->GetPos().x();
		ncords(1) = node->GetPos().y();
		ncords(2) = node->GetPos().z();


		if (IsInside(ncords, dx)){
			std::cout << "adding node " << node->GetIndex() << " to joint" << std::endl;
			auto constraint = chrono_types::make_shared<::chrono::fea::ChLinkPointFrame>();
			constraint->Initialize(node, m_body);
			ch_system->Add(constraint);
			nadded ++;
		}

	}

	return nadded;
}

void Object::makeDynamometer(::chrono::ChSystem* ch_system,
	std::vector<std::shared_ptr<::chrono::fea::ChLinkPointFrame>>& writing_point_constr, // store pointers of nodes to write 
	std::vector<std::shared_ptr<::chrono::fea::ChLinkDirFrame>>& writing_dir_constr) // store pointers of nodes to write 
{
	m_fea_mesh = chrono_types::make_shared<::chrono::fea::ChMesh>();

	auto fixednode = chrono_types::make_shared<::chrono::fea::ChNodeFEAxyzD>(
		::chrono::ChVector<>(m_center(0), m_center(1), m_center(2)));

	fixednode->SetFixed(true);
	m_fea_mesh->AddNode(fixednode);
	ch_system->Add(m_fea_mesh);

	auto constraint_disp = chrono_types::make_shared<::chrono::fea::ChLinkPointFrame>();
	constraint_disp->Initialize(fixednode, m_body);
	ch_system->Add(constraint_disp);
	writing_point_constr.push_back(constraint_disp);

	auto constraint_rot = chrono_types::make_shared<::chrono::fea::ChLinkDirFrame>();
	constraint_rot->Initialize(fixednode, m_body);
	ch_system->Add(constraint_rot);
	writing_dir_constr.push_back(constraint_rot);

	std::cout << "Added fea Dynamics measurement point in (" << m_center(0) << ", " << m_center(1) << ", " << m_center(2) << ")" << std::endl;
}

uint Object::findNodesToJoin(std::shared_ptr<::chrono::fea::ChMesh> fea_mesh,
	const double dx,
	std::vector<feaNodeInfo>& included_nodes)
{
	std::shared_ptr<::chrono::fea::ChNodeFEAxyz> node;
	uint numnodes = fea_mesh->GetNnodes();

	uint nadded = 0;

	feaNodeInfo node_info;

	for (uint i = 0; i < numnodes; i++) {

		node = std::dynamic_pointer_cast<::chrono::fea::ChNodeFEAxyz>(fea_mesh->GetNode(i));
		if (!node) throw std::runtime_error("Error: impossible to read nodes in JointFeaNode");

		node_info.node = node;

		Point ncords;


		ncords(0) = node->GetPos().x();
		ncords(1) = node->GetPos().y();
		ncords(2) = node->GetPos().z();

		double distance = dist(ncords, m_center);
		node_info.dist = distance;

		if (IsInside(ncords, dx)){

			included_nodes.push_back(node_info);
			nadded ++;
		}

	}

	return nadded;
}

uint Object::findForceNodes(std::shared_ptr<::chrono::fea::ChMesh> fea_mesh,
	const double dx,
	const uint num_prev_nodes,
	std::vector<bool>& ext_forces_flags) // for each node says if we apply force
{
	std::shared_ptr<::chrono::fea::ChNodeFEAxyz> node;
	uint numnodes = fea_mesh->GetNnodes();

	uint nadded = 0;

	for (uint i = 0; i < numnodes; i++) {

		node = std::dynamic_pointer_cast<::chrono::fea::ChNodeFEAxyz>(fea_mesh->GetNode(i));
		if (!node) throw std::runtime_error("Error: impossible to read nodes in JointFeaNode");

		Point ncords;

		ncords(0) = node->GetPos().x();
		ncords(1) = node->GetPos().y();
		ncords(2) = node->GetPos().z();

		uint glob_idx = node->GetIndex() + num_prev_nodes;

		if (IsInside(ncords, dx)){
			std::cout << "applying force to node " << glob_idx << std::endl;
			ext_forces_flags.push_back(true);
			nadded ++;
		} else {
			ext_forces_flags.push_back(false); //TODO alternatively we can initialize everything to false and set true when required
		}
	}

	return nadded;
}

uint Object::findNodesToWrite(std::shared_ptr<::chrono::fea::ChMesh> fea_mesh,
	const double dx,
	const uint num_prev_nodes,
	std::vector<int>& writing_nodes_indices,
	std::vector<std::shared_ptr<::chrono::fea::ChNodeFEAxyz>>& writing_nodes_pointers) // store pointers of nodes to write 
{
	std::shared_ptr<::chrono::fea::ChNodeFEAxyz> node;
	uint numnodes = fea_mesh->GetNnodes();

	uint nadded = 0;

	for (uint i = 0; i < numnodes; i++) {

		node = std::dynamic_pointer_cast<::chrono::fea::ChNodeFEAxyz>(fea_mesh->GetNode(i));
		if (!node) throw std::runtime_error("Error: impossible to read nodes in JointFeaNode");

		Point ncords;

		ncords(0) = node->GetPos().x();
		ncords(1) = node->GetPos().y();
		ncords(2) = node->GetPos().z();

		uint glob_idx = node->GetIndex() + num_prev_nodes;

		if (IsInside(ncords, dx)){
			std::cout << "Writing data for node " << glob_idx << std::endl;
			writing_nodes_indices.push_back(glob_idx);
			writing_nodes_pointers.push_back(node);
			nadded ++;
		}
	}

	return nadded;
}
#endif

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

#if USE_CHRONO == 1
/// Create a Chrono body associated to the cube
/* Create a generic Chrono body inside a specified Chrono physical system.
 *	\param bodies_physical_system : Chrono physical system
 *	\param dx : particle spacing
 *	\param collide : add collision handling
 *	\param orientation_diff: additional orientation
 */
void
Object::BodyCreate(::chrono::ChSystem * bodies_physical_system, const double dx, const bool collide,
			const chrono::ChQuaternion<> & orientation_diff)
{
	// Check if the physical system is valid
	if (!bodies_physical_system)
		throw std::runtime_error("Object::BodyCreate Trying to create a body in an invalid physical system !\n");

	// Creating a new Chrono object
	m_body = chrono_types::make_shared< ::chrono::ChBody >();

	// Assign cube mass and inertial data to the Chrono object
	m_body->SetMass(m_mass);
	m_body->SetInertiaXX(::chrono::ChVector<>(m_inertia[0], m_inertia[1], m_inertia[2]));
	m_body->SetPos(::chrono::ChVector<>(m_center(0), m_center(1), m_center(2)));
	m_body->SetRot(orientation_diff*m_ep.ToChQuaternion());

	m_body->SetCollide(collide);
	m_body->SetBodyFixed(m_isFixed);

	// Add the body to the physical system
	bodies_physical_system->AddBody(m_body);
}

void
Object::BodyCreate(::chrono::ChSystem *bodies_physical_system, const double dx,
		const bool collide)
{
	BodyCreate(bodies_physical_system, dx, collide, ::chrono::ChQuaternion<>(1., 0., 0., 0.));
}

void
Object::CreateFemMesh(::chrono::ChSystem *fea_system)
{
	std::string class_name = abi::__cxa_demangle(typeid(*this).name(), NULL, 0, NULL);
	std::string error = "CreateFemMesh for " + class_name + " is not supported yet";
	throw std::runtime_error(error);
}
#endif

/// Print ODE-related information such as position, CG, geometry bounding box (if any), etc.
// TODO: could be useful to print also the rotation matrix
void Object::BodyPrintInformation(const bool print_geom)
{
#if USE_CHRONO == 1
	if (m_body) {
		const ::chrono::ChVector<> cg = m_body->GetPos();
		double mass = m_body->GetMass();
		const ::chrono::ChVector<> inertiaXX = m_body->GetInertiaXX();
		const ::chrono::ChVector<> inertiaXY = m_body->GetInertiaXY();
		printf("Chrono Body pointer: %p\n", m_body.get());
		printf("   Mass: %e\n", mass);
		printf("   CG:   %e\t%e\t%e\n", cg.x(), cg.y(), cg.z());
		printf("   I:    %e\t%e\t%e\n", inertiaXX.x(), inertiaXY.x(), inertiaXY.y());
		printf("         %e\t%e\t%e\n", inertiaXY.x(), inertiaXX.y(), inertiaXY.z());
		printf("         %e\t%e\t%e\n", inertiaXY.y(), inertiaXY.z(), inertiaXX.z());
		const ::chrono::ChQuaternion<> quat = m_body->GetRot();
		printf("   Q:    %e\t%e\t%e\t%e\n", quat.e0(), quat.e1(), quat.e2(), quat.e3());

		// not only check if an ODE geometry is associated, but also it must not be a plane
		if (print_geom && m_body->GetCollide()) {
			::chrono::ChVector<> bbmin, bbmax;
			m_body->GetCollisionModel()->GetAABB(bbmin, bbmax);
			printf("Chrono collision shape\n");
			printf("   B. box:   X [%g,%g], Y [%g,%g], Z [%g,%g]\n",
				bbmin.x(), bbmax.x(), bbmin.y(), bbmax.y(), bbmin.z(), bbmax.z());
			printf("   size:     X [%g] Y [%g] Z [%g]\n", bbmax.x() - bbmin.x(),
					bbmax.y() - bbmin.y(), bbmax.z() - bbmin.z());
		}
	}
#else
	std::cout << "No body associated with the object (USE_CHRONO not defined).\n";
#endif
}
