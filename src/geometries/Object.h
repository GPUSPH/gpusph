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

#ifndef OBJECT_H
#define	OBJECT_H

#include <stdexcept>
#include <memory>

#include "Point.h"
#include "EulerParameters.h"

#include "chrono_select.opt"
// Forward declaration to avoid including Chrono headers
namespace chrono {
class ChBody;
class ChSystem;
namespace fea {
class ChMesh;
class ChNodeFEAxyz;
class ChLinkPointFrame;
class ChLinkDirFrame;
}
}

//! Auxiliary type for joints between FEA nodes
struct feaNodeInfo {
	double dist; //distance from the center of the joint geometry
	std::shared_ptr<::chrono::fea::ChNodeFEAxyz> node; //pointer to node
};


//! Object container class
/*!
 *	The Object class is a container class for geometrical objects.
 *	It contains several utility function for drawing, filling, unfilling
 *  and setting or accessing object data along with pure virtual function
 *  that need to be implemented by his children.
 *
 *
 *  For simplicity and efficiency reasons we use, often as possible, the
 *  principal axis of inertia frame as intermediate referential system
 *  and use Euler parameters to specify its relative situation respect
 *  to the global frame.
*/
class Object {
	protected:
		//! The number of dimensions of the world the objects are immersed in.
		/*! This is used to determine how methods such as FillIn
		 *  will work for 2D objects (e.g. multiple copies of the object in 3D,
		 *  a 2-dimensional FillIn in 2D)
		 */
		static int world_dimensions;

		EulerParameters		m_ep;			///< Euler parameters associated with the object
		Point				m_center;		///< Coordinates of center of gravity in the global reference frame + particle mass (4th component)
		double				m_inertia[3];	///< Inertia matrix in the principal axes of inertia frame
		double				m_mass;			///< Mass of the object
		PointVect			m_parts;		///< Particles belonging to the object
		PointVect			m_fea_nodes;	///< FEA mesh nodes belonging to the object
		uint				m_numParts;		///< Number of particles belonging to the object
		uint				m_numFeaNodes;	///< Number of nodes of the FEA mesh associated to the object
		bool				m_isFixed;		///< Is it fixed in space?


		double				m_youngModulus;		///< Young's modulus for deformable objects
		double				m_poissonRatio;		///< Poisson ratio for deformable objects
		double				m_alphaDamping;		///< alpha damping for deformable objects
		double				m_density;		///> object density

		std::shared_ptr< ::chrono::ChBody >		m_body;		///< Chrono body linked to the object
		std::shared_ptr< ::chrono::fea::ChMesh>	m_fea_mesh;	///< Chrono mesh linked to the object

		std::vector<int>		m_fea_nodes_offset;     ///< when reusing previous nodes here we store the offset between the used and replaced node 
		uint				m_previous_nodes;      ///< number of nodes already defined in the previous geometries

		// auxiliary function for computing the bounding box
		void getBoundingBoxOfCube(Point &out_min, Point &out_max,
			Point &origin, Vector v1, Vector v2, Vector v3);
	public:
		//! Set the number of world dimensions —must be only used once
		static void set_world_dimensions(int dim);

		Object(void) {
			m_body = NULL;
			m_fea_mesh = NULL;
			m_mass = 0.0;
			m_center = Point(0,0,0);
			m_numParts = 0;
			m_previous_nodes = 0;
			m_isFixed = false;
			m_inertia[0] = NAN;
			m_inertia[1] = NAN;
			m_inertia[2] = NAN;
		};

		virtual ~Object(void)
		{
		};

		/// \name Mass related functions
		//@{
		virtual double SetPartMass(const double dx, const double rho);
		virtual void SetPartMass(const double mass);
		double GetPartMass();
		virtual double SetMass(const double dx, const double rho);
		virtual void SetMass(const double mass);
		virtual void SetYoungModulus(const double);
		virtual void SetPoissonRatio(const double);
		virtual void SetAlphaDamping(const double);
		virtual void SetDensity(const double); // TODO FIXME verify consistency with setPartMassByDensity in new Problem API
		double GetMass();
		virtual double Volume(const double dx) const = 0;
		//@}

		/// \name Inertia related functions
		//@{
		/// Compute the matrix of inertia
		/*! This function compute the matrix of inertia of the object in the inertial
		 *  frame (i.e. the 3 diagonal components) and store it in the m_inertia array.
		 *	For the same reasons as volume, the inertia depends on particle spacing.
		 *	\param dx : particle spacing
		 *
		 *	This function is pure virtual and then has to be defined at child level
		 */
		virtual void SetInertia(const double dx) = 0;
		void SetInertia(const double*);
		void SetInertia(const double i11, const double i22, const double i33);
		void SetCenterOfGravity(const double*);
		double3 GetCenterOfGravity(void) const;
		EulerParameters GetOrientation(void) const;
		virtual void GetInertialFrameData(double*, double&, double*, EulerParameters&) const;
		//@}

		/// Set body as fixed in space
		void SetFixed() { m_isFixed = true; }

		/// Returns the particle vector associated with the object
		PointVect& GetParts(void);
		/// Returns the particle vector associated with the fea nodes 
		PointVect& GetFeaNodes(void);

		bool reduceNodes(std::shared_ptr<::chrono::fea::ChNodeFEAxyz> newNode, ::chrono::ChSystem * fea_system, std::vector<std::shared_ptr<::chrono::fea::ChNodeFEAxyz>>&);

		void set_previous_nodes_num(::chrono::ChSystem * fea_system);

		/// Sets the number of particles associated with an object
		void SetNumParts(const int numParts);
		/// Gets the number of particles associated with an object
		/*! This function either returns the set number of particles which is used
		 *  in case of a loaded STL mesh or the number of particles set in m_parts
		 */
		uint GetNumParts();

		/// Gets the number of nodes of the mesh associated with an object
		uint GetNumFeaNodes();

		/// \name Chrono rigid body related functions
		/* These are not pure virtual in order to allow new GPUSPH Objects to be defined without
		 * needing a Chrono counterpart, but the default implementation will just throw
		 * an exception.
		 */
		//@{
		/// Create a Chrono body in the specified Chrono physical system
		virtual void BodyCreate(::chrono::ChSystem * bodies_physical_system, const double dx, const bool collide,
			const EulerParameters & orientation_diff);
		void BodyCreate(::chrono::ChSystem * bodies_physical_system, const double dx, const bool collide);
		std::shared_ptr< ::chrono::ChBody > GetBody(void)
		{	if (!m_body)
				throw std::runtime_error("Object::GetBody called but object not associated with a Chrono body !");
			return m_body;
		}
		// just check, without throwing
		bool HasBody() { return (!!m_body); }

		std::shared_ptr<::chrono::fea::ChMesh> GetFeaMesh(void)
		{
			if (!m_fea_mesh)
				throw std::runtime_error("Object::GetFeaMesh called but no FEA meshes associated with a Chrono body !");

			return m_fea_mesh;
		}

		bool HasFeaMesh() { return (!! m_fea_mesh); }
		virtual void CreateFemMesh(::chrono::ChSystem *fea_system);
		virtual float4 getNaturalCoords(double4 global_pos)
		{ throw std::runtime_error("Calling getNaturalCoords for a geometry that doesn't support FEA meshes yet"); }
		virtual int4 getOwningNodes(double4 global_pos)
		{ throw std::runtime_error("Calling getowningNodes for a geometry that doesn't support FEA meshes yet"); }

		/// Print body-related information such as position, CG, geometry bounding box (if any), etc.
		void BodyPrintInformation(const bool print_geom = true);
		//@}

		/// \name Filling functions
		//@{
		int FillDisk(PointVect&, const EulerParameters&, const Point&, const double,
					const double, const double, const bool fill = true) const;
		int FillDisk(PointVect&, const EulerParameters&, const Point&, const double,
					const double, const double, const double, const bool fill = true) const;
		int FillDiskBorder(PointVect&, const EulerParameters&, const Point&, const double,
					const double, const double, const double, const bool fill = true) const;
		/// Fill object surface with particles
		/*!	Fill the object surface with particle at a given particle spacing and add
		 *	the particles to the given particle vector
		 *  the number of particles added.
		 *	\param points : particle vector to add particles to
		 *	\param dx : particle spacing
		 *
		 *  This function is pure virtual and then as to be defined at child level
		 */
		virtual void FillBorder(PointVect& points, const double dx) = 0;
		/// Fill object with particles
		/*!	Fill the whole object with particle at a given particle spacing and return
		 *  the needed number of particles.
		 *
		 *  If the fill parameter is set to false the function just count the number of
		 *  particles needed otherwise the particles are added to the particle vector.
		 *	\param points : particle vector to add particles to
		 *	\param dx : particle spacing
		 *	\param fill : fill flag (true particles are generated and added to parts, false no particle is generated)
		 *	\return number of particles needed to fill the object
		 *
		 *  This function is pure virtual and then as to be defined at child level
		 */
		virtual int Fill(PointVect& points, const double dx, const bool fill = true) = 0;
		/// Fill object with a specified number of particles layer
		/*!	Fill multiple layers of particles starting from the object surface
		 *
		 *	\param points : particle vector to add particles to
		 *	\param dx : particle spacing
		 *	\param layers : number of layers
		 *
		 *  This function is pure virtual and then as to be defined at child level
		 */
		virtual void FillIn(PointVect& points, const double dx, const int layers) = 0;
		void Unfill(PointVect&, const double) const;
		void Intersect(PointVect&, const double) const;
		//@}

#if USE_CHRONO == 1
		/// Handle joining of FEA nodes
		//@{
		uint JoinFeaNodes(::chrono::ChSystem* fea_system, std::shared_ptr<::chrono::fea::ChMesh>, const double dx);
		virtual void makeDynamometer(::chrono::ChSystem* fea_system,
			std::vector<std::shared_ptr<::chrono::fea::ChLinkPointFrame>>&,
			std::vector<std::shared_ptr<::chrono::fea::ChLinkDirFrame>>&);

		uint findNodesToJoin(std::shared_ptr<::chrono::fea::ChMesh>,
			const double dx,
			std::vector<feaNodeInfo>& included_nodes);
		uint findForceNodes(std::shared_ptr<::chrono::fea::ChMesh>,
			const double dx,
			const uint num_prev_nodes,
			std::vector<bool>& ext_forces_flags);
		uint findNodesToWrite(std::shared_ptr<::chrono::fea::ChMesh>,
			const double dx,
			const uint num_prev_nodes,
			std::vector<int>& writing_nodes_indices,
			std::vector<std::shared_ptr<::chrono::fea::ChNodeFEAxyz>>& writing_nodes_pointers);
		//@}
#endif

		/// Detect if a particle is inside an object
		/*!	Detect if a particle is located inside the object or at a distance inferior
		 *  to a threshold value.
		 *	\param dx : threshold value
		 *	\return true if particle is inside the object or closer than dx
		 *
		 *  This function is pure virtual and then as to be defined at child level
		 */
		virtual bool IsInside(const Point& p, const double dx) const = 0;

		/// \name Other functions
		//@{
		/// Set the EulerParameters
		/*! This function sets the EulerParameters and updates the object accordingly
		 *	\param ep : new EulerParameters
		 *
		 *	This function is pure virtual and then has to be defined at child level
		 */
		virtual void setEulerParameters(const EulerParameters &ep) = 0;

		/// Get the EulerParameters
		/*! This function returns the EulerParameters
		 *	\return EulerParameters
		 *
		 *	This function is pure virtual and then has to be defined at child level
		 */
		const EulerParameters* getEulerParameters() {return &m_ep; }

		/// Get the bounding box
		/*! This function writes the bounding box of the object in the given parameters,
		 *  taking into account also the object rotation
		 *  \param min : minimum coordinates
		 *  \param min : maximum coordinates
		 *
		 *  This function is pure virtual and then has to be defined at child level.
		 */
		virtual void getBoundingBox(Point &output_min, Point &output_max) = 0;

		/// Shift the object (center, origin, etc.) with the given offset
		/*! This function shifts the object with the given offset. The object
		 *  internally updates everything necessary.
		 *  \param double3 : offset
		 *
		 *  This function is pure virtual and then has to be defined at child level.
		 */
		virtual void shift(const double3 &offset) = 0;
};
#endif	/* OBJECT_H */

