#ifndef _XPROBLEM_H
#define	_XPROBLEM_H

#include <string>

// SIZE_MAX
#include <limits.h>

#include "Problem.h"

// HDF5 reader
#include "HDF5SphReader.h"

enum GeometryType {	GT_FLUID,
					GT_FIXED_BOUNDARY,
					GT_OPENBOUNDARY,
					GT_FLOATING_BODY,
					GT_MOVING_BODY,
					GT_PLANE
};

enum FillType {	FT_NOFILL,
				FT_SOLID,
				FT_SOLID_BORDERLESS,
				FT_BORDER
};

enum IntersectionType {	IT_INTERSECT,
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

	bool has_stl_file; // ditto
	std::string stl_filename;

	// user-set inertia
	double custom_inertia[3];

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

		has_stl_file = false;
		stl_filename = "";

		custom_inertia[0] = NAN;
		custom_inertia[1] = NAN;
		custom_inertia[2] = NAN;
	}
};

typedef std::vector<GeometryInfo*> GeometryVector;

// GeometryID, aka index of the GeometryInfo in the GeometryVector
typedef size_t GeometryID;
#define GEOMETRY_ERROR		SIZE_MAX

class XProblem: public Problem {
	private:
		GeometryVector m_geometries;
		PointVect m_fluidParts;
		PointVect m_boundaryParts;
		//PointVect m_vertexParts;

		size_t m_numActiveGeometries; // do NOT use it to iterate on m_geometries, since it lacks the deleted geoms
		size_t m_numPlanes;
		size_t m_numOpenBoundaries;
		size_t m_numFloatingBodies;		// number of floating bodies (handled with ODE)

		// extra margin to be added to computed world size
		double m_extra_world_margin;

		PositioningPolicy m_positioning;

		// initialize ODE
		void initializeODE();
		// guess what
		void cleanupODE();

		// wrapper with common operations for adding a geometry
		GeometryID addGeometry(const GeometryType otype, const FillType ftype, Object *obj_ptr,
			const char *hdf5_fname = NULL, const char *stl_fname = NULL);

		// check validity of given GeometryID
		bool validGeometry(GeometryID gid);

		// used for hydrostatic filling (absolute value)
		double m_waterLevel;
		// used to set LJ dcoeff and sound speed if m_maxParticleSpeed is unset
		double m_maxFall;
		// used to set sound of speed
		double m_maxParticleSpeed;

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

		// get read-only information
		const GeometryInfo* getGeometryInfo(GeometryID gid);

		// set the positioning policy for geometries added after the call
		void setPositioning(PositioningPolicy positioning);

		// define 6 planes delimiting the box with given corners; update word size and origin; write their IDs in planesIds
		void makeUniverseBox(const double3 corner1, const double3 corner2, GeometryID *planesIds = NULL);

		// world size will be increased by the given margin in each dimension and direction
		void addExtraWorldMargin(const double margin);

		// set m_waterLevel for automatic hydrostatic filling
		void setWaterLevel(double waterLevel) { m_waterLevel = waterLevel; }
		// set m_maxFall (former "H") for setting sspeed
		void setMaxFall(double maxFall) { m_maxFall = maxFall; }
		// set _expected_ max particle speed
		void setMaxParticleSpeed(double maxParticleSpeed) { m_maxParticleSpeed = maxParticleSpeed; }

		// callback for initializing particles with custom values
		virtual void initializeParticles(BufferList &buffers, const uint numParticles);

	public:
		XProblem(GlobalData *);
		~XProblem(void);

		// initialize world size, ODE if necessary; public, since GPUSPH will call it
		bool initialize();

		int fill_parts();
		uint fill_planes(void);
		void copy_planes(float4 *planes, float *planediv);

		void copy_to_array(BufferList &buffers);
		void release_memory();

		virtual void ODE_near_callback(void * data, dGeomID o1, dGeomID o2);

		// methods related to SA bounds
		virtual void init_keps(float*, float*, uint, particleinfo*, float4*, hashKey*);
		//virtual uint max_parts(uint);

		virtual void setboundconstants(
			const	PhysParams	*physparams,
			float3	const&		worldOrigin,
			uint3	const&		gridSize,
			float3	const&		cellSize);

		virtual void imposeBoundaryConditionHost(
					float4*			newVel,
					float4*			newEulerVel,
					float*			newTke,
					float*			newEpsilon,
			const	particleinfo*	info,
			const	float4*			oldPos,
					uint*			IOwaterdepth,
			const	float			t,
			const	uint			numParticles,
			const	uint			numObjects,
			const	uint			particleRangeEnd,
			const	hashKey*		particleHash);

		virtual void imposeForcedMovingObjects(
					float3	&gravityCenters,
					float3	&translations,
					float*	rotationMatrices,
			const	uint	ob,
			const	double	t,
			const	float	dt);

		// will probably implement a smart, general purpose one
		// void fillDeviceMap();
};

#endif	/* _XPROBLEM_H */
