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

#ifndef _XPROBLEM_H
#define	_XPROBLEM_H

#include <string>
#include <vector>

// SIZE_MAX
#include <limits.h>

#include "Problem.h"

// HDF5 and XYF file readers
#include "HDF5SphReader.h"
#include "XYZReader.h"

enum GeometryType {	GT_FLUID,
					GT_FIXED_BOUNDARY,
					GT_OPENBOUNDARY,
					GT_FLOATING_BODY,
					GT_MOVING_BODY,
					GT_PLANE,
					GT_TESTPOINTS
};

enum FillType {	FT_NOFILL,
				FT_SOLID,
				FT_SOLID_BORDERLESS,
				FT_BORDER
};

enum IntersectionType {	IT_NONE,
						IT_INTERSECT,
						IT_SUBTRACT
};

// NOTE: erasing is always done with fluid or boundary point vectors.
// It is *not* done with moving or floating objects, which erase but
// are not erased. ET_ERASE_ALL does not erase moving/floating objects.
enum EraseOperation {	ET_ERASE_NOTHING,
						ET_ERASE_FLUID,
						ET_ERASE_BOUNDARY,
						ET_ERASE_ALL
};

// position of the solids with respect to the point given by user. PP_NONE is geometry-dependent
enum PositioningPolicy {	PP_NONE,
							PP_CENTER,
							PP_BOTTOM_CENTER,
							PP_CORNER
};

struct GeometryInfo {

	Object* ptr;

	GeometryType type;
	FillType fill_type;
	IntersectionType intersection_type;
	EraseOperation erase_operation;

	bool handle_collisions;
	bool handle_dynamics; // implies measure_forces
	bool measure_forces;
	bool enabled;

	bool has_hdf5_file; // little redundant but clearer
	std::string hdf5_filename;
	HDF5SphReader *hdf5_reader;
	bool flip_normals; // for HF5 generated from STL files with wrong normals

	bool has_xyz_file;  // ditto
	std::string xyz_filename;
	XYZReader *xyz_reader;

	bool has_stl_file; // ditto
	std::string stl_filename;

	// user-set inertia
	double custom_inertia[3];

	// aux vars to check if user set what he/she should
	bool mass_was_set;
	bool particle_mass_was_set;

	// flag to distinguish pressure/velocity open boundaries
	bool velocity_driven;

	GeometryInfo() {
		ptr = NULL;

		type = GT_FLUID;
		fill_type = FT_SOLID;
		intersection_type = IT_SUBTRACT;
		erase_operation = ET_ERASE_NOTHING;

		handle_collisions = false;
		handle_dynamics = false;
		measure_forces = false;

		enabled = true;

		has_hdf5_file = false;
		hdf5_filename = "";
		hdf5_reader = NULL;
		flip_normals = false;

		has_xyz_file = false;
		xyz_filename = "";
		xyz_reader = NULL;

		has_stl_file = false;
		stl_filename = "";

		custom_inertia[0] = NAN;
		custom_inertia[1] = NAN;
		custom_inertia[2] = NAN;

		mass_was_set = false;
		particle_mass_was_set = false;

		velocity_driven = false;
	}
};

typedef std::vector<GeometryInfo*> GeometryVector;

// GeometryID, aka index of the GeometryInfo in the GeometryVector
typedef size_t GeometryID;
#define INVALID_GEOMETRY	SIZE_MAX

class XProblem: public Problem {
	private:
		GeometryVector m_geometries;
		PointVect m_fluidParts;
		PointVect m_boundaryParts;
		PointVect m_testpointParts;
		//PointVect m_vertexParts;

		size_t m_numActiveGeometries;	// do NOT use it to iterate on m_geometries, since it lacks the deleted geoms
		size_t m_numForcesBodies;		// number of bodies with feedback enabled (includes floating)
		size_t m_numFloatingBodies;		// number of floating bodies (handled with ODE)
		size_t m_numPlanes;				// number of plane geometries (ODE and/or GPUSPH planes)
		size_t m_numOpenBoundaries;		// number of I/O geometries

		// extra margin to be added to computed world size
		double m_extra_world_margin;

		PositioningPolicy m_positioning;

		// initialize ODE
		void initializeODE();
		// guess what
		void cleanupODE();

		// wrapper with common operations for adding a geometry
		GeometryID addGeometry(const GeometryType otype, const FillType ftype, Object *obj_ptr,
			const char *hdf5_fname = NULL, const char *xyz_fname = NULL, const char *stl_fname = NULL);

		// check validity of given GeometryID
		bool validGeometry(GeometryID gid);

		// used for hydrostatic filling (absolute value)
		double m_waterLevel;
		// used to set LJ dcoeff and sound speed if m_maxParticleSpeed is unset
		double m_maxFall;
		// used to set sound of speed
		double m_maxParticleSpeed;

		// number of layers for filling dynamic boundaries
		uint m_numDynBoundLayers;

	protected:
		// methods for creation of new objects
		GeometryID addRect(const GeometryType otype, const FillType ftype, const Point &origin,
			const double side1, const double side2);
		GeometryID addDisk(const GeometryType otype, const FillType ftype, const Point &origin,
			const double radius);
		GeometryID addCube(const GeometryType otype, const FillType ftype, const Point &origin,
			const double side);
		GeometryID addBox(const GeometryType otype, const FillType ftype, const Point &origin,
			const double side1, const double side2, const double side3);
		GeometryID addCylinder(const GeometryType otype, const FillType ftype, const Point &origin,
			const double radius, const double height);
		GeometryID addCone(const GeometryType otype, const FillType ftype, const Point &origin,
			const double bottom_radius, const double top_radius, const double height);
		GeometryID addSphere(const GeometryType otype, const FillType ftype, const Point &origin,
			const double radius);
		GeometryID addTorus(const GeometryType otype, const FillType ftype, const Point &origin,
			const double major_radius, const double minor_radius);
		GeometryID addPlane(
			const double a_coeff, const double b_coeff, const double c_coeff, const double d_coeff);
		GeometryID addSTLMesh(const GeometryType otype, const FillType ftype, const Point &origin,
			const char *fname);
		GeometryID addHDF5File(const GeometryType otype, const Point &origin,
			const char *fname_hdf5, const char *fname_stl = NULL);
		GeometryID addXYZFile(const GeometryType otype, const Point &origin,
			const char *fname_xyz, const char *fname_stl = NULL);

		// Method to add a single testpoint.
		// NOTE: does not create a geometry since Point does not derive from Object
		size_t addTestPoint(const Point &coordinates);
		size_t addTestPoint(const double posx, const double posy, const double posz);

		// request to invert normals while loading - only for HDF5 files
		void flipNormals(const GeometryID gid, bool flip = true);

		// method for deleting a geometry (actually disabling)
		void deleteGeometry(const GeometryID gid);

		// methods to enable/disable handling of dynamics/collisions for a specific geometry
		void enableDynamics(const GeometryID gid);
		void enableCollisions(const GeometryID gid);
		void disableDynamics(const GeometryID gid);
		void disableCollisions(const GeometryID gid);

		// enable/disable the measurement of forces acting on the object
		void enableFeedback(const GeometryID gid);
		void disableFeedback(const GeometryID gid);

		// methods to set a custom inertia matrix (and overwrite the precomputed one)
		void setInertia(const GeometryID gid, const double i11, const double i22, const double i33);
		void setInertia(const GeometryID gid, const double* mainDiagonal);

		// methods for rotating an existing object
		void setOrientation(const GeometryID gid, const EulerParameters &ep);
		void setOrientation(const GeometryID gid, const dQuaternion quat);
		void rotate(const GeometryID gid, const dQuaternion quat);
		void rotate(const GeometryID gid, const double Xrot, const double Yrot, const double Zrot);

		// method for shifting an existing object
		void shift(const GeometryID gid, const double Xoffset, const double Yoffset, const double Zoffset);

		// get and customize the unfilling policy
		IntersectionType getIntersectionType(const GeometryID gid) { return m_geometries[gid]->intersection_type; }
		EraseOperation getEraseOperation(const GeometryID gid) { return m_geometries[gid]->erase_operation; }
		void setIntersectionType(const GeometryID gid, const IntersectionType i_type);
		void setEraseOperation(const GeometryID gid, const EraseOperation e_operation);

		// set particle mass
		void setParticleMass(const GeometryID gid, const double mass);
		double setParticleMassByDensity(const GeometryID gid, const double density);

		// set mass (only meaningful for floating objects)
		void setMass(const GeometryID gid, const double mass);
		double setMassByDensity(const GeometryID gid, const double density);

		// flag an open boundary as velocity driven; use with false to revert to pressure driven
		void setVelocityDriven(const GeometryID gid, bool isVelocityDriven = true);

		// get read-only information
		const GeometryInfo* getGeometryInfo(GeometryID gid);

		// set the positioning policy for geometries added after the call
		void setPositioning(PositioningPolicy positioning);

		/*! Sets the domain origin and size to match the box defined by the given corners,
		 * and adds planes for each of the sides of the box, returning their GeometryIDs.
		 * Planes are not added in the periodic direction(s), if any were selected
		 * in the simulation framework.
		 */
		std::vector<GeometryID> makeUniverseBox(const double3 corner1, const double3 corner2);

		// world size will be increased by the given margin in each dimension and direction
		void addExtraWorldMargin(const double margin);

		// set m_waterLevel for automatic hydrostatic filling
		void setWaterLevel(double waterLevel) { m_waterLevel = waterLevel; }
		// set m_maxFall (former "H") for setting sspeed
		void setMaxFall(double maxFall) { m_maxFall = maxFall; }
		// set _expected_ max particle speed
		void setMaxParticleSpeed(double maxParticleSpeed) { m_maxParticleSpeed = maxParticleSpeed; }

		// set number of layers for dynamic boundaries. Default is 0, which means: autocompute
		void setDynamicBoundariesLayers(const uint numLayers);
		// get current value (NOTE: not yet autocomputed in problem constructor)
		uint getDynamicBoundariesLayers() { return m_numDynBoundLayers; }

		// callback for filtering out points before they become particles
		virtual void filterPoints(PointVect &fluidParts, PointVect &boundaryParts);
		// callback for initializing particles with custom values
		virtual void initializeParticles(BufferList &buffers, const uint numParticles);

	public:
		XProblem(GlobalData *);
		~XProblem(void);

		// initialize world size, ODE if necessary; public, since GPUSPH will call it
		bool initialize();

		int fill_parts();
		void copy_planes(PlaneList &planes);

		void copy_to_array(BufferList &buffers);
		void release_memory();

		uint suggestedDynamicBoundaryLayers();

		virtual void ODE_near_callback(void * data, dGeomID o1, dGeomID o2);

		// will probably implement a smart, general purpose one
		// void fillDeviceMap();
};

#endif	/* _XPROBLEM_H */
