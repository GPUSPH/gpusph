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
/*
 * File:   Problem.h
 * Author: alexis
 *
 * Created on 13 juin 2008, 18:21
 */

#ifndef _PROBLEM_H
#define	_PROBLEM_H

#include <string>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>

#include "Options.h"
#include "Writer.h"
#include "particledefine.h"
#include "physparams.h"
#include "simparams.h"
#include "vector_math.h"
#include "Object.h"
#include "buffer.h"
#include "simframework.h"

#include "deprecation.h"

#include "ode/ode.h"

#define BLOCK_SIZE_IOBOUND	256

typedef std::vector<vertexinfo> VertexVect;

// forward declaration. If a class wants to actually use
// the callback writer it should include CallbackWriter.h
class CallbackWriter;

enum MovingBodyType {
	MB_ODE,
	MB_FORCES_MOVING,
	MB_MOVING
};

typedef struct KinematicData {
	double3			crot; ///< Center of rotation of the body
	double3			lvel; ///< Linear velocity
	double3			avel; ///< Angular velocity
	EulerParameters	orientation;

	KinematicData():
		crot(make_double3(0.0)),
		lvel(make_double3(0.0)),
		avel(make_double3(0.0)),
		orientation(EulerParameters())
	{};

	KinematicData(const KinematicData& kdata) {
		crot = kdata.crot;
		lvel = kdata.lvel;
		avel = kdata.avel;
		orientation = kdata.orientation;
	};

	KinematicData& operator = (const KinematicData& source) {
		crot = source.crot;
		lvel = source.lvel;
		avel = source.avel;
		orientation = source.orientation;
		return *this;
	};
} KinematicData;

typedef struct MovingBodyData {
	uint				index; ///< Sequential insertion index (NOTE: NOT index in the array)
	MovingBodyType		type;
	Object				*object;
	KinematicData		kdata;
	KinematicData		initial_kdata;

	MovingBodyData(): index(0), type(MB_MOVING), object(NULL), kdata(KinematicData()), initial_kdata(KinematicData()) {};

	MovingBodyData(const MovingBodyData& mbdata) {
		index = mbdata.index;
		type = mbdata.type;
		object = mbdata.object;
		kdata = mbdata.kdata;
		initial_kdata = mbdata.initial_kdata;
	};

	MovingBodyData& operator = (const MovingBodyData& source) {
		index = source.index;
		type = source.type;
		object = source.object;
		kdata = source.kdata;
		initial_kdata = source.initial_kdata;
		return *this;
	};
} MovingBodyData;

typedef std::vector<MovingBodyData *> MovingBodiesVect;

// not including GlobalData.h since it needs the complete definition of the Problem class
struct GlobalData;

using namespace std;

class Problem {
	private:
		string			m_problem_dir;
		WriterList		m_writers;

		const float		*m_dem;
		int				m_ncols, m_nrows;

	public:
		// used to set the preferred split axis; LONGEST_AXIS (default) uses the longest of the worldSize
		enum SplitAxis
		{
			LONGEST_AXIS,
			X_AXIS,
			Y_AXIS,
			Z_AXIS
		};

		dWorldID		m_ODEWorld;
		dSpaceID		m_ODESpace;
		dJointGroupID	m_ODEJointGroup;

		double3	m_size;			// Size of computational domain
		double3	m_origin;		// Origin of computational domain
		double3	m_cellsize;		// Size of grid cells
		uint3	m_gridsize;		// Number of grid cells along each axis
		double	m_deltap;		// Initial particle spacing

		const float*	get_dem() const { return m_dem; }
		int		get_dem_ncols() const { return m_ncols; }
		int		get_dem_nrows() const { return m_nrows; }
		void	set_dem(const float *dem, int ncols, int nrows) {
			m_dem = dem; m_ncols = ncols; m_nrows = nrows;
		}

		string	m_name;

		GlobalData	*gdata;
		const Options		*m_options;					// commodity pointer to gdata->clOptions

		SimParams			*m_simparams;				//< Simulation parameters. They only exist after SETUP_FRAMEWORK
		PhysParams			*m_physparams;				//< Physical parameters. They only exist after SETUP_FRAMEWORK

		SimFramework		*m_simframework;			// simulation framework

		// Set up the simulation framework. This must be done before the rest of the simulation parameters, and it sets
		// * SPH kernel
		// * SPH formulation
		// * viscosity model
		// * boundary model
		// * periodicity
		// * flags (see simflags.h)
		// Example usage with default parameters:
		/*
			SETUP_FRAMEWORK(
				kernel<WENDLAND>,
				formulation<SPH_F1>,
				viscosity<ARTVISC>,
				boundary<LJ_BOUNDARY>,
				periodicity<PERIODIC_NONE>,
				flags<ENABLE_DTADAPT>
			);
		*/

#define	SETUP_FRAMEWORK(...) do { \
	m_simframework = new CUDASimFramework< __VA_ARGS__ >(); \
	m_simparams = m_simframework->get_simparams(); \
	m_physparams = new PhysParams(); \
} while (0)

		// add a filter (MLS, SHEPARD), with given frequency
#define	addFilter(fltr, freq) m_simframework->addFilterEngine(fltr, freq)

		// add a post-processing filter, e.g.:
		// addPostProcess(CALC_PRIVATE);
		// addPostProcess(SURFACE_DETECTION); // simple surface detection
		// addPostProcess(SURFACE_DETECTION, BUFFER_NORMALS); // save normals too
#define	addPostProcess m_simframework->addPostProcessEngine

		MovingBodiesVect	m_bodies;			// array of moving objects
		KinematicData		*m_bodies_storage;				// kinematic data storage for bodie movement integration

		Problem(GlobalData *_gdata);

		// returns true if successful, false if should abort the simulation
		virtual bool initialize() { return true; };

		virtual ~Problem(void);

		/* a function to check if the (initial or fixed) timestep
		 * is compatible with the CFL conditions */
		virtual void check_dt();
		/* Find the minimum amount of maximum number of neighbors
		 * per particle based on the kernel and boundary choice,
		 * and compare against the user-set value (if any), or
		 * just set it by default */
		virtual void check_maxneibsnum();

		string const& create_problem_dir();

		Options const& get_options(void) const
		{
			return *m_options;
		}

		double3 const& get_worldorigin(void) const
		{
			return m_origin;
		};

		double3 const& get_worldsize(void) const
		{
			return m_size;
		};

		double3 const& get_cellsize(void) const
		{
			return m_cellsize;
		};

		uint3 const& get_gridsize(void) const
		{
			return m_gridsize;
		};

		float density(float, int) const;
		float density_for_pressure(float, int) const;

		float pressure(float, int) const;

		float soundspeed(float, int) const;

		string const& get_dirname(void) const
		{
			return m_problem_dir;
		}

		double set_deltap(const double dflt)
		{
			if (isfinite((double) m_options->deltap))
				m_deltap = m_options->deltap;
			else
				m_deltap = dflt;
			// also udate the smoothing length
			set_smoothing(m_simparams->sfactor);

			return m_deltap;
		}

		double get_deltap() const
		{ return m_deltap; }

		// Compute the ferrari coefficient based on a lengthscale
		void calculateFerrariCoefficient();

		/* set smoothing factor */
		double set_smoothing(const double smooth)
		{
			return m_simparams->set_smoothing(smooth, m_deltap);
		}

IGNORE_WARNINGS(deprecated-declarations)
		/* set kernel type and radius */
		// DEPRECATED, use set_kernel_radius instead
		double set_kernel(KernelType kernel, double radius=0) DEPRECATED
		{
			return m_simparams->set_kernel(kernel, radius);
		}
RESTORE_WARNINGS

		void set_kernel_radius(double radius)
		{
			m_simparams->set_kernel_radius(radius);
		}

		void set_grid_params(void);

		int3 calc_grid_pos(const Point&);

		uint calc_grid_hash(int3);

		void calc_localpos_and_hash(const Point&, const particleinfo&, float4&, hashKey&);

		const SimParams *get_simparams(void) const
		{
			return m_simparams;
		};

		SimParams *get_simparams(void)
		{
			return m_simparams;
		};

		const PhysParams *get_physparams(void) const
		{
			return m_physparams;
		};

		PhysParams *get_physparams(void)
		{
			return m_physparams;
		};

		// simple functions to add gages. the third component
		// is actually ignored
		void add_gage(double3 const& pt);

		inline
		void add_gage(double2 const& pt)
		{ add_gage(make_double3(pt.x, pt.y, 0.0)); }

		inline
		void add_gage(double x, double y, double z=0)
		{ add_gage(make_double3(x, y, z)); }

		// set the timer tick
		// DEPRECATED: use add_writer() with the frequency in seconds
		void set_timer_tick(double t) DEPRECATED;

		// add a new writer
		// by passing as argument the product of freq and the timer tick
		// DEPRECATED: use add_writer() with the frequency in seconds
		void add_writer(WriterType wt, int freq = 1) DEPRECATED_MSG("use add_writer(WriterType, double)");

		// add a new writer, with the given write frequency in (fractions of) seconds
		void add_writer(WriterType wt, double freq);

		// return the list of writers
		WriterList const& get_writers() const
		{ return m_writers; }

		// overridden in subclasses if they want explicit writes
		// beyond those controlled by the writer(s) periodic time
		virtual bool need_write(double) const;

		// overridden in subclasses if they want to write custom stuff
		// using the CALLBACKWRITER
		virtual void writer_callback(CallbackWriter *,
			uint numParts, BufferList const&, uint node_offset, double t,
			const bool testpoints) const;

		// is the simulation running at the given time?
		bool finished(double) const;

		virtual int fill_parts(void) = 0;
		// maximum number of particles that may be generated
		virtual uint max_parts(uint numParts);
		virtual uint fill_planes(void);
		virtual void copy_to_array(BufferList & ) = 0;
		virtual void copy_planes(float4*, float*);
		virtual void release_memory(void) = 0;

		/* moving boundary and gravity callbacks */
		virtual float3 g_callback(const float t) DEPRECATED;
		virtual float3 g_callback(const double t);

		/* ODE callbacks */
		virtual void ODE_near_callback(void * data, dGeomID o1, dGeomID o2)
		{
			cerr << "ERROR: you forget to implement ODE_near_callback in your problem.\n";
		}

		static void ODE_near_callback_wrapper(void * data, dGeomID o1, dGeomID o2)
		{
			Problem* problem = (Problem *) data;
			problem->ODE_near_callback(data, o1, o2);
		}

		void allocate_bodies_storage();
		void add_moving_body(Object *, const MovingBodyType);
		const MovingBodiesVect& get_mbvect() const
		{ return m_bodies; };

		MovingBodyData * get_mbdata(const uint);
		MovingBodyData * get_mbdata(const Object *);

		size_t	get_bodies_numparts(void);
		size_t	get_forces_bodies_numparts(void);
		size_t	get_body_numparts(const int);
		size_t	get_body_numparts(const Object *);

		void get_bodies_data(float3 * &, float * &, float3 * &, float3 * &);
		void get_bodies_cg(void);
		void set_body_cg(const double3&, MovingBodyData*);
		void set_body_cg(const uint, const double3&);
		void set_body_cg(const Object*, const double3&);
		void set_body_linearvel(const double3&, MovingBodyData*);
		void set_body_linearvel(const uint, const double3&);
		void set_body_linearvel(const Object*, const double3&);
		void set_body_angularvel(const double3&, MovingBodyData*);
		void set_body_angularvel(const uint, const double3&);
		void set_body_angularvel(const Object*, const double3&);

		/* This method can be overridden in problems when the object
		 * forces have to be altered in some way before being applied.
		 */
		virtual void
		bodies_forces_callback(const double t0, const double t1, const uint step, float3 *forces, float3 *torques);

		virtual void
		post_timestep_callback(const double t);

		virtual void
		moving_bodies_callback(const uint, Object*, const double, const double, const float3&,
								const float3&, const KinematicData &, KinematicData &,
								double3&, EulerParameters&);

		void bodies_timestep(const float3 *, const float3 *, const int,
							const double, const double, float3 * &, float3 * &,
							float * &, float3 * &, float3 * &);

		/*void restore_ODE_body(const uint, const float *gravity_center, const float *quaternion,
			const float *linvel, const float *angvel);*/

		virtual void init_keps(float*, float*, uint, particleinfo*, float4*, hashKey*);

		/* Initialize the particle volumes */
		virtual void init_volume(BufferList &, uint numParticles);

		virtual void
		setboundconstants(
			const	PhysParams	*physparams,
			float3	const&		worldOrigin,
			uint3	const&		gridSize,
			float3	const&		cellSize) {};

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

		// Partition the grid in numDevices parts - virtual to allow problem or topology-specific implementations
		virtual void fillDeviceMap();
		// partition by splitting the cells according to their linearized hash
		void fillDeviceMapByCellHash();
		// partition by splitting along an axis. Default: along the longest
		void fillDeviceMapByAxis(SplitAxis preferred_split_axis);
		// partition by coordinates satistfying an example equation
		void fillDeviceMapByEquation();
		// partition by cutting the domain in parallelepipeds
		void fillDeviceMapByRegularGrid();
		// partition by performing the specified number of cuts along the three cartesian axes
		void fillDeviceMapByAxesSplits(uint Xslices, uint Yslices, uint Zslices);

};
#endif
