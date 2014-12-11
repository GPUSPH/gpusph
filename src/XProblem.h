#ifndef _XPROBLEM_H
#define	_XPROBLEM_H

#include <string>

#include "Problem.h"

// HDF5 reader
//#include "HDF5SphReader.h"

enum GeometryType {	GT_FLUID,
					GT_FIXED_BOUNDARY,
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

struct GeometryInfo {

	Object* ptr;

	GeometryType type;
	FillType fill_type;
	IntersectionType intersection_type;
	EraseOperation erase_operation;

	bool handle_collisions;
	bool handle_dynamics;
	bool enabled;

	GeometryInfo() {
		ptr = NULL;

		type = GT_FLUID;
		fill_type = FT_SOLID;
		intersection_type = IT_SUBTRACT;
		erase_operation = ET_ERASE_NOTHING;

		handle_collisions = false;
		handle_dynamics = false;

		enabled = true;
	}
};

typedef std::vector<GeometryInfo*> GeometryVector;

// GeometryID, aka index of the GeometryInfo in the GeometryVector
typedef size_t GeometryID;

class XProblem: public Problem {
	private:
		GeometryVector m_geometries;
		PointVect m_fluidParts;
		PointVect m_boundaryParts;
		//PointVect m_vertexParts;

		size_t m_numActiveGeometries; // do NOT use it to iterate on m_geometries, since it lacks the deleted geoms
		size_t m_numRigidBodies; // equivalent to m_simparams.numODEbodies after bodies have been added
		size_t m_numPlanes;

		// extra margin to be added to computed world size
		double m_extra_world_margin;

		/*
		string			inputfile;
		double			world_w, world_l, world_h;			// world size (i.e. incl. margins and inlet box)
		double			box_w, box_l, box_h;	// size of the main box (excl. margins, no inlet box)
		HDF5SphReader	h5File;
		*/

		// initialize ODE
		void initializeODE();
		// guess what
		void cleanupODE();

		GeometryID addGeometry(const GeometryType otype, const FillType ftype, Object *obj_ptr);

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

		// method for deleting a geometry (actually disabling)
		void deleteGeometry(const GeometryID gid);

		// methods to enable/disable handling of dynamics/collisions for a specific geometry
		void enableDynamics(const GeometryID gid);
		void enableCollisions(const GeometryID gid);
		void disableDynamics(const GeometryID gid);
		void disableCollisions(const GeometryID gid);

		// methods for rotating an existing object
		void rotateGeometry(const GeometryID gid, const EulerParameters &ep);
		void rotateGeometry(const GeometryID gid, const dQuaternion quat);
		void rotateGeometry(const GeometryID gid, const double Xrot, const double Yrot, const double Zrot);

		// get and customize the unfilling policy
		IntersectionType getIntersectionType(const GeometryID gid) { return m_geometries[gid]->intersection_type; }
		EraseOperation getEraseOperation(const GeometryID gid) { return m_geometries[gid]->erase_operation; }
		void setIntersectionType(const GeometryID gid, const IntersectionType i_type);
		void setEraseOperation(const GeometryID gid, const EraseOperation e_operation);

		// set mass (only meaningful for floating objects)
		void setMass(const GeometryID gid, const double mass);
		double setMassByDensity(const GeometryID gid, const double density);

		// get read-only information
		const GeometryInfo* getGeometryInfo(GeometryID gid);

		// world size will be increased by the given margin in each dimension and direction
		void addExtraWorldMargin(const double margin);

	public:
		XProblem(const GlobalData *);
		~XProblem(void);

		// initialize world size, ODE if necessary
		// public, since GPUSPH will call it
		void initialize();

		int fill_parts();
		uint fill_planes(void);
		void copy_planes(float4 *planes, float *planediv);

		void copy_to_array(BufferList &buffers);
		void release_memory();

		void ODE_near_callback(void * data, dGeomID o1, dGeomID o2);

		/*
		void init_keps(float*, float*, uint, particleinfo*, float4*, hashKey*);
		uint max_parts(uint);

		void
		setboundconstants(
			const	PhysParams	*physparams,
			float3	const&		worldOrigin,
			uint3	const&		gridSize,
			float3	const&		cellSize);

		void
		imposeBoundaryConditionHost(
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

		void imposeForcedMovingObjects(
					float3*	gravityCenters,
					float3*	translations,
					float*	rotationMatrices,
			const	uint*	ODEobjectId,
			const	uint	numObjects,
			const	double	t,
			const	float	dt);

		void fillDeviceMap();
		*/
};

#endif	/* _XPROBLEM_H */
