/*  Copyright (c) 2014-2019 INGV, EDF, UniCT, JHU

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

#ifndef PROBLEM_API_H
#define PROBLEM_API_H

#include <string>
#include <vector>
#include <memory> // shared_ptr

// SIZE_MAX
#include <limits.h>


// Problem API version 1 is the first high-level Problem API, that used to be known as
// XProblem before GPUSPH version 5.

#define PROBLEM_API 1
#include "Problem.h"

// HDF5 and XYF file readers
#include "HDF5SphReader.h"
#include "XYZReader.h"

enum GeometryType {	GT_FLUID,
					GT_FIXED_BOUNDARY,
					GT_FIXEDBOUNDARY =
						GT_FIXED_BOUNDARY,
					GT_OPEN_BOUNDARY,
					GT_OPENBOUNDARY =
						GT_OPEN_BOUNDARY,
					GT_FLOATING_BODY,
					GT_MOVING_BODY,
					GT_PLANE,
					GT_DEM,
					GT_TESTPOINTS,
					GT_FREE_SURFACE,
					GT_DEFORMABLE_BODY,
					GT_FEA_RIGID_JOINT,
					GT_FEA_FLEXIBLE_JOINT,
					GT_FEA_FORCE,
					GT_FEA_WRITE
};

enum FillType {	FT_NOFILL,
				FT_SOLID,
				FT_SOLID_BORDERLESS,
				FT_BORDER,
				FT_UNFILL
};

enum IntersectionType {	IT_NONE,
						IT_INTERSECT,
						IT_SUBTRACT
};

// TODO currently we need to keep this in sync with the
// definitions in TopoCube.h, ideally this shouldn't be needed.
// We don't want to just do using TopographyFormat = TopoCube::Format
// for two reasons:
// 1. it requires the inclusion of TopoCube.h, which would make it an anomaly
//    compared to other geometries (for which no inclusion is needed)
// 2. we can't specify the DEM file format simply as DEM_FMT_*, but we would need to do
//    TopographyFormat::DEM_FMT_*, which is extremely verbose
// So for the time being this enum replicates TopoCube::Format explicitly
enum TopographyFormat {	DEM_FMT_ASCII,
						DEM_FMT_VTK,
						DEM_FMT_XYZ
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

using ObjectPtr = std::shared_ptr<Object>;

struct GeometryInfo {

	ObjectPtr ptr;

	GeometryType type;
	FillType fill_type;
	IntersectionType intersection_type;
	EraseOperation erase_operation;

	bool handle_collisions;
	bool handle_dynamics; // implies measure_forces
	bool measure_forces;
	bool fea; //object for finite element analysis
	bool enabled;
	bool is_dynamometer; // has a chrono constraint with force-torque measurement

	bool has_hdf5_file; // little redundant but clearer
	std::string hdf5_filename;
	HDF5SphReader *hdf5_reader;
	bool flip_normals; // for HF5 generated from STL files with wrong normals

	bool has_xyz_file;  // ditto
	std::string xyz_filename;
	XYZReader *xyz_reader;

	bool has_mesh_file; // ditto
	// STL mesh or topography file name
	std::string stl_filename;
	TopographyFormat dem_fmt; // file format for the DEM topography file

	// user-set inertia
	double custom_inertia[3];

	// user-set center of gravity
	double3 custom_cg;

	// aux vars to check if user set what he/she should
	bool mass_was_set;
	bool particle_mass_was_set;

	// flag to distinguish pressure/velocity open boundaries
	bool velocity_driven;

	// custom radius for unfill operations. NAN -> use dp
	double unfill_radius;

	GeometryInfo() {
		ptr = NULL;

		type = GT_FLUID;
		fill_type = FT_SOLID;
		intersection_type = IT_SUBTRACT;
		erase_operation = ET_ERASE_NOTHING;

		handle_collisions = false;
		handle_dynamics = false;
		measure_forces = false;

		fea = false;

		enabled = true;

		is_dynamometer = false;

		has_hdf5_file = false;
		hdf5_filename = "";
		hdf5_reader = NULL;
		flip_normals = false;

		has_xyz_file = false;
		xyz_filename = "";
		xyz_reader = NULL;

		has_mesh_file = false;
		stl_filename = "";
		dem_fmt = DEM_FMT_ASCII;

		custom_inertia[0] = NAN;
		custom_inertia[1] = NAN;
		custom_inertia[2] = NAN;

		custom_cg = make_double3(NAN);

		mass_was_set = false;
		particle_mass_was_set = false;

		velocity_driven = false;

		unfill_radius = NAN;
	}
};

typedef std::vector<GeometryInfo*> GeometryVector;

// GeometryID, aka index of the GeometryInfo in the GeometryVector
typedef size_t GeometryID;
#define INVALID_GEOMETRY	SIZE_MAX

template<>
class ProblemAPI<1> : public ProblemCore
{
	private:
		GeometryVector m_geometries;
		PointVect m_fluidParts;
		PointVect m_boundaryParts;
		PointVect m_testpointParts;
		//PointVect m_vertexParts;

		size_t m_numActiveGeometries;	// do NOT use it to iterate on m_geometries, since it lacks the deleted geoms
		size_t m_numForcesBodies;		// number of bodies with feedback enabled (includes floating)
		size_t m_numFloatingBodies;		// number of floating bodies (handled with Chrono)
		size_t m_numPlanes;				// number of plane geometries (Chrono and/or GPUSPH planes)
		size_t m_numOpenBoundaries;		// number of I/O geometries
		size_t m_numFEAObjects;		// chrono::fea elements or meshes 

		// extra margin to be added to computed world size
		double m_extra_world_margin;

		PositioningPolicy m_positioning;

		GeometryID m_dem_geometry; // the ID of the (only) DEM, if defined
		double m_dem_zmin_scale; // used to compute demzmin from ∆p
		double m_dem_dx_scale; // used to compute demdx from DEM resolution
		double m_dem_dy_scale; // used to compute demdy from DEM resolution

		// initialize Chrono FEA
		void initializeChronoFEA();
		// guess what
		void cleanupChronoFEA();

		// initialize Chrono
		void initializeChrono();
		// guess what
		void cleanupChrono();

		// wrapper with common operations for adding a geometry
		GeometryID addGeometry(const GeometryType otype, const FillType ftype, ObjectPtr obj_ptr,
			const char *hdf5_fname = NULL, const char *xyz_fname = NULL, const char *stl_fname = NULL);

		// check validity of given GeometryID
		bool validGeometry(GeometryID gid);

		//! Maximum height for a particle fall
		/*! This is used to autocompute the Lennard-Jones boundary coefficient dcoeff,
		 * and to autocompute the sound speed (together with m_maxParticleSpeed)
		 */
		double m_maxFall;

		//! Maximum expected velocity for a fluid particle
		/*! If unset, it will be auto-computed as the maximum fall velocity.
		 * Together with the maximum fall velocity, it is used to autocompute the speed of sound
		 * if not specified.
		 *
		 * \inpsection{c0_input_method, calculation}
		 * \label{FLUID_MAX_SPEED}
		 * \default{0}
		 */
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
			//const double side1, const double side2, const double side3, int nelsx, int nelsy, int nelsz);
			const double side1, const double side2, const double side3, int nelsx, int nelsy);
		GeometryID addBox(const GeometryType otype, const FillType ftype, const Point &origin,
			const double side1, const double side2, const double side3);
		GeometryID addCylinder(const GeometryType otype, const FillType ftype, const Point &origin,
			const double outer_radius, const double height);
		GeometryID addCylinder(const GeometryType otype, const FillType ftype, const Point &origin,
			const double outer_radius, const double inner_radius, const double height,
			//const uint nelst, const uint nelsc, const uint nelsh);
			const uint nelsh);
		GeometryID addCone(const GeometryType otype, const FillType ftype, const Point &origin,
			const double bottom_radius, const double top_radius, const double height);
		GeometryID addSphere(const GeometryType otype, const FillType ftype, const Point &origin,
			const double radius);
		GeometryID addTorus(const GeometryType otype, const FillType ftype, const Point &origin,
			const double major_radius, const double minor_radius);
		GeometryID addPlane(
			const double a_coeff, const double b_coeff, const double c_coeff, const double d_coeff,
			const FillType ftype = FT_NOFILL);
		GeometryID addSTLMesh(const GeometryType otype, const FillType ftype, const Point &origin,
			const char *fname);
		GeometryID addOBJMesh(const GeometryType otype, const FillType ftype, const Point &origin,
			const char *fname);
		GeometryID addHDF5File(const GeometryType otype, const Point &origin,
			const char *fname_hdf5, const char *fname_stl = NULL);
		GeometryID addTetFile(const GeometryType otype, const FillType ftype, const Point &origin,
			const char *nodes, const char *elems, const double z_frame);
		GeometryID addXYZFile(const GeometryType otype, const Point &origin,
			const char *fname_xyz, const char *fname_stl = NULL);

		//! Add a Digital Elevation Model (topography) from the given file (in the given format)
		/*! Topography is its own GeometryType (GT_DEM) and by default does not produce any particles
		 * (it is handled geometrically); note however that this needs ENABLE_DEM in the framework
		 * simulation flags.
		 * There can only be one “active” DEM in the simulation, but you can add other DEMs if they are
		 * only used for cutting (fill type FT_UNFILL).
		 * It is also possible to get additional geometries associated with the topography, see e.g.
		 * addDEMPlanes, addDEMFluidBox
		 */
		GeometryID addDEM(const char *fname_dem, const TopographyFormat dem_fmt = DEM_FMT_ASCII,
			const FillType fill_type = FT_NOFILL);
		GeometryID addDEM(std::string const& fname_dem, const TopographyFormat dem_fmt = DEM_FMT_ASCII,
			const FillType fill_type = FT_NOFILL)
		{ return addDEM(fname_dem.c_str(), dem_fmt, fill_type); }

		//! Add a box of fluid on top of the DEM, with the given fluid height
		/*! The fluid height is considered from the bottom of the DEM
		 */
		GeometryID addDEMFluidBox(double height, GeometryID dem_gid = INVALID_GEOMETRY);

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

		// method to set a custom center of gravity (and overwrite the precomputed one)
		void setCenterOfGravity(const GeometryID gid, const double3 cg);

		// methods for rotating an existing object
		void setOrientation(const GeometryID gid, const EulerParameters &ep);
		void rotate(const GeometryID gid, const EulerParameters ep); // DEPRECATED until we'll have a GPUSPH Quaternion class
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

		// set Young's modulus (only meaningful for deformable objects)
		void setYoungModulus(const GeometryID gid, const double youngModulus);

		// set Dynamometer (only for GT_FEA_RIGID_JOINT) 
		void setDynamometer(const GeometryID gid, const bool isDynamometer = true);

		// set Poisson ratio (only meaningful for deformable objects)
		void setPoissonRatio(const GeometryID gid, const double poissonRatio);

		// set Alpha damping (only meaningful for deformable objects)
		void setAlphaDamping(const GeometryID gid, const double alphaDamping);

		// set density (only meaningful for deformable objects)
		void setDensity(const GeometryID gid, const double density);

		// flag an open boundary as velocity driven; use with false to revert to pressure driven
		void setVelocityDriven(const GeometryID gid, bool isVelocityDriven = true);

		// set custom radius for unfill operations. NAN means: use dp
		void setUnfillRadius(const GeometryID gid, double unfillRadius);

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

		/*! Create a plane for each of the four sides of the specified DEM.
		 * The dem_gid refers to the GeometryID of the DEM to use as reference.
		 * It can be omitted, in which case the current DEM is used automatically.
		 * The planes will have the same filltype as the DEM.
		 */
		std::vector<GeometryID> addDEMPlanes(GeometryID gid = INVALID_GEOMETRY);
		std::vector<GeometryID> addDEMPlanes(GeometryID gid, FillType ftype);

		// world size will be increased by the given margin in each dimension and direction
		void addExtraWorldMargin(const double margin);

		// set m_waterLevel for automatic hydrostatic filling
		void setWaterLevel(double waterLevel) { m_waterLevel = waterLevel; }
		// set m_maxFall (former "H") for setting sspeed
		void setMaxFall(double maxFall) { m_maxFall = maxFall; }
		// set _expected_ max particle speed
		void setMaxParticleSpeed(double maxParticleSpeed) { m_maxParticleSpeed = maxParticleSpeed; }

		// Enable/disable automatic hydrostatic filling
		void enableHydrostaticFilling() { m_hydrostaticFilling = true; }
		void disableHydrostaticFilling()  { m_hydrostaticFilling = false; }

		//! demzmin will be computed as scale*m_deltap
		void setDEMZminScale(double scale);
		//! set demzmin to the given value
		void setDEMZmin(double demzmin);
		//! demdx and demdy will be computed as resolution/scale
		//! if scaley is not set, scalex will be used for both directions
		void setDEMNormalDisplacementScale(double scalex, double scaley=NAN);
		//! set demdx and demdy to the given values
		//! if demdy is not set, demdx will be used for both directions
		void setDEMNormalDisplacement(double demdx, double demdy=NAN);

		//! Compute DEM physical parameters.
		//! This is normally done automatically during the filling phase, but can be forced earlier
		//! if the user wants the information for other purposes
		void computeDEMphysparams();

		//! Get the inter-particle spacing based on the geometry type
		double preferredDeltaP(GeometryType type);

		// set number of layers for dynamic boundaries. Default is 0, which means: autocompute
		void setDynamicBoundariesLayers(const uint numLayers);
		// get current value (NOTE: not yet autocomputed in problem constructor)
		uint getDynamicBoundariesLayers() { return m_numDynBoundLayers; }

		//! callback for filtering out points before they become particles during
		//! GPUSPH internal pre-processing
		virtual void filterPoints(PointVect &fluidParts, PointVect &boundaryParts);

	public:
		ProblemAPI(GlobalData *);
		~ProblemAPI(void);

		// initialize world size, Chrono if necessary; public, since GPUSPH will call it
		bool initialize();

		int fill_parts(bool fill = true);
		void copy_planes(PlaneList &planes);

		void copy_to_array(BufferList &buffers);
		void release_memory();

		uint suggestedDynamicBoundaryLayers();

		// will probably implement a smart, general purpose one
		// void fillDeviceMap();
};
#endif


