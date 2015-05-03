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

#include "Options.h"
#include "Writer.h"
#include "particledefine.h"
#include "physparams.h"
#include "simparams.h"
#include "vector_math.h"
#include "Object.h"
#include "buffer.h"

#include "deprecation.h"

#include "ode/ode.h"

#define BLOCK_SIZE_IOBOUND	256

typedef std::vector<vertexinfo> VertexVect;

// not including GlobalData.h since it needs the complete definition of the Problem class
struct GlobalData;

using namespace std;

class Problem {
	private:
		string		m_problem_dir;
		WriterList	m_writers;

		const float	*m_dem;
		int			m_ncols, m_nrows;

		static uint		m_total_ODE_bodies;			///< Total number of rigid bodies used by ODE

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
		SimParams	m_simparams;
		PhysParams	m_physparams;
		MbCallBack	m_mbcallbackdata[MAXMOVINGBOUND];	// array of structure for moving boundary data
		int			m_mbnumber;							// number of moving boundaries

		Object		**m_ODE_bodies;						// array of floating ODE objects
		float4		m_mbdata[MAXMOVINGBOUND];			// moving boudary data to be provided to euler
		float3		m_bodies_cg[MAXBODIES];				// center of gravity of rigid bodies
		dQuaternion m_bodies_quaternion[MAXBODIES];		// orientation of the rigid bodies
		float3		m_bodies_trans[MAXBODIES];			// translation to apply between t and t + dt
		float3		m_bodies_linearvel[MAXBODIES];		// Linear velocity of rigid bodies
		float3		m_bodies_angularvel[MAXBODIES];		// Angular velocity of rigid bodies
		float		m_bodies_steprot[9*MAXBODIES];		// rotation to apply between t and t + dt
		uint		m_ODEobjectId[MAXBODIES];			// ODE object id

		Problem(GlobalData *_gdata);

		// returns true if successful, false if should abort the simulation
		virtual bool initialize() { return true; };

		virtual ~Problem(void);

		/* a function to check if the (initial or fixed) timestep
		 * is compatible with the CFL coditions */
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
			set_smoothing(m_simparams.sfactor);

			return m_deltap;
		}

		double get_deltap() const
		{ return m_deltap; }

		// Compute the ferrari coefficient based on a lengthscale
		void calculateFerrariCoefficient();

		/* set smoothing factor */
		double set_smoothing(const double smooth)
		{
			return m_simparams.set_smoothing(smooth, m_deltap);
		}

		/* set kernel type and radius */
		double set_kernel(KernelType kernel, double radius=0)
		{
			return m_simparams.set_kernel(kernel, radius);
		}

		void set_grid_params(void);

		int3 calc_grid_pos(const Point&);

		uint calc_grid_hash(int3);

		void calc_localpos_and_hash(const Point&, const particleinfo&, float4&, hashKey&);

		const SimParams *get_simparams(void) const
		{
			return &m_simparams;
		};

		SimParams *get_simparams(void)
		{
			return &m_simparams;
		};

		const PhysParams *get_physparams(void) const
		{
			return &m_physparams;
		};

		PhysParams *get_physparams(void)
		{
			return &m_physparams;
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
		virtual MbCallBack& mb_callback(const float t, const float dt, const int i) DEPRECATED;
		virtual float3 g_callback(const float t) DEPRECATED;

		virtual MbCallBack& mb_callback(const double t, const float dt, const int i);
		virtual float3 g_callback(const double t);

		float4* get_mbdata(const double t, const float dt, const bool forceupdate);


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

		void allocate_ODE_bodies(const uint);
		void add_ODE_body(Object* object);
		Object* get_ODE_body(const uint);
		Object const* const* get_ODE_bodies() const
		{ return m_ODE_bodies; }

		void get_ODE_bodies_data(float3 * &, float * &, float3 * &, float3 * &);
		float3* get_ODE_bodies_cg(void);
		dQuaternion *get_ODE_bodies_quaternion(void);
		float* get_ODE_bodies_steprot(void);
		float3* get_ODE_bodies_linearvel(void);
		float3* get_ODE_bodies_angularvel(void);

		/* This method can be overridden in problems when the object
		 * forces have to be altered in some way before being applied.
		 */
		virtual void
		object_forces_callback(double t, int step, float3 *forces, float3 *torques);

		void ODE_bodies_timestep(const float3 *, const float3 *, const int,
									const double, float3 * &, float3 * &, float * &,
									float3 * &, float3 * &);
		size_t	get_ODE_bodies_numparts(void) const;
		size_t	get_ODE_body_numparts(const int) const;

		void restore_ODE_body(const uint, const float *gravity_center, const float *quaternion,
			const float *linvel, const float *angvel);

		virtual void init_keps(float*, float*, uint, particleinfo*, float4*, hashKey*);

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
