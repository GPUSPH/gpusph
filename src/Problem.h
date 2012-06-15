/*  Copyright 2011 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

	Istituto de Nazionale di Geofisica e Vulcanologia
          Sezione di Catania, Catania, Italy

    Universita di Catania, Catania, Italy

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

#include "Options.h"
#include "RigidBody.h"
#include "particledefine.h"
#include "physparams.h"
#include "simparams.h"

#include "ode/ode.h"

// not including GlobalData.h since it needs the complete definition of the Problem class
struct GlobalData;

using namespace std;

class Problem {
	private:
		float	m_last_display_time;
		float	m_last_write_time;
		float	m_last_rbdata_write_time;
		float	m_last_screenshot_time;
		string	m_problem_dir;

		static int		m_total_ODE_bodies;			///< Total number of rigid bodies used by ODE

		const float*	m_dem;
		int		m_ncols, m_nrows;


	public:
		enum WriterType
		{
			TEXTWRITER,
			VTKWRITER,
			VTKLEGACYWRITER,
			CUSTOMTEXTWRITER,
			UDPWRITER
		};

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

		float3	m_size;			// Size of compuation domain
		float3	m_origin;		// Origin of compuatation domain
		float	m_deltap;		// Initial particle spacing

		// Min and max values used for display
		float	m_maxrho;
		float	m_minrho;
		float	m_maxvel;
		float	m_minvel;

		float		m_displayinterval;
		float		m_rbdata_writeinterval;
		int			m_writefreq;
		int			m_screenshotfreq;
		WriterType	m_writerType;
		FILE*		m_rbdatafile;

		const float*	get_dem() const { return m_dem; }
		int		get_dem_ncols() { return m_ncols; }
		int		get_dem_nrows() { return m_nrows; }
		void	set_dem(const float *dem, int ncols, int nrows) {
			m_dem = dem; m_ncols = ncols; m_nrows = nrows;
		}

		string	m_name;

		Options		m_options;
		SimParams	m_simparams;
		PhysParams	m_physparams;
		MbCallBack	m_mbcallbackdata[MAXMOVINGBOUND];	// array of structure for moving boundary data
		int			m_mbnumber;							// number of moving boundaries

		RigidBody	*m_bodies;							// array of RigidBody objects
		Object		**m_ODE_bodies;						// array of Objects tat are floating bodies
		float4		m_mbdata[MAXMOVINGBOUND];			// mb data to be provided by ParticleSystem to euler
		float3		m_bodies_cg[MAXBODIES];				// center of gravity of rigid bodies
		float3		m_bodies_trans[MAXBODIES];			// translation to apply between t and t + dt
		float		m_bodies_steprot[9*MAXBODIES];		// rotation to apply between t and t + dt

		Problem(const Options &options = Options());

		virtual ~Problem(void);

		Options get_options(void)
		{
			return m_options;
		}

		float3 const& get_worldorigin(void) const
		{
			return m_origin;
		};

		float3 const& get_worldsize(void) const
		{
			return m_size;
		};

		WriterType get_writertype(void)
		{
			return m_writerType;
		};

		float get_minrho(void) { return m_minrho; };

		float get_maxrho(void) { return m_maxrho; };

		float get_maxvel(void) { return m_maxvel; };

		float get_minvel(void) { return m_minvel; };

		float density(float, int);

		float pressure(float, int) const;

		float soundspeed(float, int);

		string get_dirname(void) const
		{
			return m_problem_dir.c_str();
		}

		float set_deltap(const float dflt)
		{
			if (isfinite((double) m_options.deltap))
				m_deltap = m_options.deltap;
			else
				m_deltap = dflt;
			return m_deltap;
		}

		// add an outlet min_[xyz], max_[xyz] with displacement direction
		// dir_[xyz]
		// Returns false if there isn't room for more outlets.
		bool add_outlet(
			float min_x, float min_y, float min_z,
			float max_x, float max_y, float max_z,
			float dir_x, float dir_y, float dir_z);

		// ditto, vector form
		inline bool add_outlet(float3 const& omin, float3 const& omax, float3 const& dir)
		{
			return add_outlet(omin.x, omin.y, omin.z,
				omax.x, omax.y, omax.z,
				dir.x, dir.y, dir.z);
		}

		// add an inlet min_[xyz], max_[xyz] with velocity components
		// vel_[xyzw]. Velocity components that should evolve naturally
		// should be set to NAN.
		// Returns false if there isn't room for more outlets.
		bool add_inlet(
			float min_x, float min_y, float min_z,
			float max_x, float max_y, float max_z,
			float vel_x, float vel_y, float vel_z, float vel_w);

		// ditto, vector form
		inline bool add_inlet(float3 const& omin, float3 const& omax, float4 const& vel)
		{
			return add_inlet(omin.x, omin.y, omin.z,
				omax.x, omax.y, omax.z,
				vel.x, vel.y, vel.z, vel.w);
		}

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

		string create_problem_dir();
		bool need_display(float);
		bool need_write(float);
		void mark_written(float t) { m_last_write_time = t; }
		bool need_write_rbdata(float);
		void write_rbdata(float);
		bool need_screenshot(float);
		// is the simulation running at the given time?
		bool finished(float);

		virtual int fill_parts(void) = 0;
		// maximum number of particles that may be generated
		virtual uint max_parts(uint numParts);
		virtual uint fill_planes(void);
		virtual void draw_boundary(float) = 0;
		virtual void draw_axis(void);
		/*virtual void draw_inlets(void);
		virtual void draw_outlets(void);*/

		virtual void copy_to_array(float4*, float4*, particleinfo*) = 0;
		virtual void copy_planes(float4*, float*);
		virtual void release_memory(void) = 0;
		virtual MbCallBack& mb_callback(const float, const float, const int);
		virtual float4* get_mbdata(const float, const float, const bool);
		virtual float3 g_callback(const float);
		virtual void ODE_near_callback(void * data, dGeomID o1, dGeomID o2)
		{
			cerr << "ERROR: you forget to implement ODE_near_callback in your problem.\n";
		}

		static void ODE_near_callback_wrapper(void * data, dGeomID o1, dGeomID o2)
		{
			Problem* problem = (Problem *) data;
			problem->ODE_near_callback(data, o1, o2);
		}

		// Partition the grid in numDevices parts - virtual to allow problem or topology-specific implementations
		virtual void fillDeviceMap(GlobalData* gdata);
		// partition by splitting the cells according to their linearized hash
		void fillDeviceMapByCellHash(GlobalData* gdata);
		// partition by splitting along an axis. Default: along the longest
		void fillDeviceMapByAxis(GlobalData* gdata, SplitAxis preferred_split_axis);
		// partition by coordinates satistfying an example equation
		void fillDeviceMapByEquation(GlobalData* gdata);
		// partition by cutting the domain in parallelepipeds
		void fillDeviceMapByRegularGrid(GlobalData* gdata);
		// partition by performing the specified number of cuts along the three cartesian axes
		void fillDeviceMapByAxesSplits(GlobalData* gdata, uint Xslices, uint Yslices, uint Zslices);

		void allocate_bodies(const int);
		void allocate_ODE_bodies(const int);
		void add_ODE_body(Object* object);
		Object* get_ODE_body(const int);
		RigidBody* get_body(const int);
		void get_rigidbodies_data(float3 * &, float * &);
		float3* get_rigidbodies_cg(void);
		float3* get_ODE_bodies_cg(void);
		float* get_rigidbodies_steprot(void);
		void rigidbodies_timestep(const float3 *, const float3 *, const int,
									const double, float3 * &, float3 * &, float * &);
		int	get_bodies_numparts(void);
		int	get_body_numparts(const int);
		int	get_ODE_bodies_numparts(void);
		int	get_ODE_body_numparts(const int);
};
#endif
