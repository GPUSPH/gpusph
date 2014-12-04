#ifndef _XPROBLEM_H
#define	_XPROBLEM_H

#include <string>

#include "Problem.h"

// HDF5 reader
//#include "HDF5SphReader.h"

// Specialized Objects
//#include "Cube.h"
//#include "Sphere.h"
//#include "Cylinder.h"
//#include "STLMesh.h"

enum GeometryType {	GT_FLUID,
					GT_FIXED_BOUNDARY,
					GT_FLOATING_BODY,
					GT_MOVING_BODY
};

enum FillType {	FT_SOLID,
				FT_SOLID_BORDERLESS,
				FT_BORDER
};

struct GeometryInfo {

	Object* ptr;

	GeometryType type;
	FillType fill_type;

	bool handle_collisions;
	bool enabled;

	GeometryInfo() {
		ptr = NULL;
		type = GT_FLUID;
		fill_type = FT_SOLID;
		handle_collisions = false;
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
		/*string			inputfile;
		PointVect		test_points;
		double			world_w, world_l, world_h;			// world size (i.e. incl. margins and inlet box)
		double			box_w, box_l, box_h;	// size of the main box (excl. margins, no inlet box)
		double			initial_water_level;			// used for initial hydrostatic filling
		double			expected_final_water_level;		// used to set D constant
		HDF5SphReader	h5File;
		STLMesh		*container;
		STLMesh		*cube;
		dGeomID		m_box_planes[5];	// planes to model the main tank*/

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

		// methods for deletin a geometry (actually disabling)
		void deleteGeometry(const GeometryID gid);

		// methods for rotating an existing object
		void rotateGeometry(const GeometryID gid, const EulerParameters &ep);
		void rotateGeometry(const GeometryID gid, const dQuaternion quat);
		void rotateGeometry(const GeometryID gid, const double Xrot, const double Yrot, const double Zrot);

	public:
		XProblem(const GlobalData *);
		~XProblem(void);

		int fill_parts();
		void copy_to_array(BufferList &buffers);
		void release_memory();
		/*
		void init_keps(float*, float*, uint, particleinfo*, float4*, hashKey*);
		uint max_parts(uint);

		void ODE_near_callback(void * data, dGeomID o1, dGeomID o2);

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
