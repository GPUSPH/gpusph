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
/*
 * File:   ProblemCore.h
 * Author: alexis
 *
 * Created on 13 juin 2008, 18:21
 */

/*! \file
 * Core Problem class interface and related definitions
 */

/* \note
 * The sections to be used in the user interface are
 * defined in gpusphgui/SGPUSPH/resources/params.xml.
 * Please consult this file for the list of sections.
*/

#ifndef PROBLEMCORE_H
#define	PROBLEMCORE_H

#include <string>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

#include "Options.h"
#include "Writer.h"
#include "particledefine.h"
#include "physparams.h"
#include "simparams.h"
#include "vector_math.h"
#include "Object.h"
#include "MovingBody.h"

#include "buffer.h"
#include "simframework.h"
// #include "deprecation.h"

#include "chrono_select.opt"
#if USE_CHRONO == 1
#include "chrono/physics/ChSystem.h"
#endif

//#include "math.h"

#define BLOCK_SIZE_IOBOUND	256U

typedef std::vector<vertexinfo> VertexVect;

// forward declaration. If a class wants to actually use
// the callback writer it should include CallbackWriter.h
class CallbackWriter;

// not including GlobalData.h since it needs the complete definition of the ProblemCore class
struct GlobalData;

class ProblemCore
{
	private:
		std::string			m_problem_dir;
		WriterList		m_writers;

		const float		*m_dem;
		int				m_ncols, m_nrows;

		PhysParams			*m_physparams;				//< Physical parameters

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

#define	SETUP_FRAMEWORK(...) this->simframework() =  CUDASimFramework< __VA_ARGS__ >()

	public:
		// used to set the preferred split axis; LONGEST_AXIS (default) uses the longest of the worldSize
		enum SplitAxis
		{
			LONGEST_AXIS,
			X_AXIS,
			Y_AXIS,
			Z_AXIS
		};

#if USE_CHRONO == 1
		::chrono::ChSystem	*m_bodies_physical_system;	// Chrono physical system containing all solid bodies, contacts, ...
#else
		void *m_bodies_physical_system;
#endif

		/*! \inpsection{geometry}
		 * \label{SIZE}
		 * \default{1e9,1e9,1e9}
		 * \mandatory
     * TLT_SIZE
		 */
		double3	m_size;			// Size of the computational domain

		/*! \inpsection{geometry}
		 * \label{ORIGIN}
		 * \default{0,0,0}
		 * \mandatory
     * TLT_ORIGIN
		 */
		double3	m_origin;		// Origin of the computational domain

		double3	m_cellsize;		// Size of grid cells
		uint3	m_gridsize;		// Number of grid cells along each axis

		//! Number of particles that were placed outside of the domain
		//  during initialization.
		//  (Mutable because this is debug information that we collect during
		//   the execution of calc_localpos_and_hash(), which is const
		mutable size_t m_out_of_bounds_count;

		//! \inpsection{discretisation}
		//! \default{-1}
		//! \label{SPH_DR}
		//! TLT_SPH_DR
		double	m_deltap;		// Initial particle spacing

		/*!
		 * \inpsection{c0_input_method, calculation}
		 * \label{FLUID_WATER_LEVEL}
		 * \default{0}
		 * used for hydrostatic filling (absolute value) and to set the speed of sound
		 */
		double m_waterLevel;

		// enable hydrostatic filling already during fill - uses m_waterLevel
		bool m_hydrostaticFilling;

		const float*	get_dem() const { return m_dem; }
		int		get_dem_ncols() const { return m_ncols; }
		int		get_dem_nrows() const { return m_nrows; }
		void	set_dem(const float *dem, int ncols, int nrows) {
			m_dem = dem; m_ncols = ncols; m_nrows = nrows;
		}

		std::string	m_name;

		GlobalData	*gdata;
		const Options		*m_options;					// commodity pointer to gdata->clOptions

		MovingBodiesVect	m_bodies;			// array of moving objects
		KinematicData		*m_bodies_storage;				// kinematic data storage for bodie movement integration

		ProblemCore(GlobalData *_gdata);

		// returns true if successful, false if should abort the simulation
		virtual bool initialize();

		virtual ~ProblemCore(void);

		/*! a function to check if the (initial or fixed) timestep
		 * is compatible with the CFL conditions
		 */
		virtual void check_dt();
		/*! Find the minimum amount of maximum number of neighbors
		 * per particle based on the kernel and boundary choice,
		 * and compare against the user-set value (if any), or
		 * just set it by default
		 */
		virtual void check_neiblistsize();

		std::string const& create_problem_dir();

		Options const& get_options(void) const
		{ return *m_options; }

		template <typename T>
		T
		get_option(std::string const& key, T _default) const
		{ return m_options->get(key, _default); }

		std::string
		get_option(std::string const& key, const char * _default) const
		{ return m_options->get(key, std::string(_default)); }

		double3 const& get_worldorigin(void) const
		{ return m_origin; }

		double3 const& get_worldsize(void) const
		{ return m_size; }

		double3 const& get_cellsize(void) const
		{ return m_cellsize; }

		uint3 const& get_gridsize(void) const
		{ return m_gridsize; }

		//! Returns the at-rest (numerical) density
		/*! This function takes a fluid number like all fluid-related functions,
		 * even if the numerical at-rest density is always 0
		 */
		float atrest_density(int /* i */) const
		{ return 0; }

		//! Returns the at-rest physical density
		float atrest_physical_density(int i) const
		{ return physparams()->rho0[i]; }

		//! Compute the (numerical) hydrostatic density
		/*! Compute the hydrostatic density for depth h
		 * and fluid number i
		 */
		float hydrostatic_density(float h, int i) const;

		//! Inverse equation of state
		/*! Compute the (numerical) density needed to achieve pressure P
		 * for fluid number i
		 */
		float density_for_pressure(float P, int i) const;

		//! Return the physical density value
		/*! In preparation for the relative density work,
		 * provide a function that converts a used-density-value
		 * to a physical density value; currently this just returns
		 * the used density value itself, since we work with the physical
		 * density anyway.
		 */
		float physical_density(float numerical_density, int i) const;

		//! Return the numerical density value
		/*! In preparation for the relative density work,
		 * provide a function that converts a physical density to the
		 * value used internally; currently this just returns the used
		 * density value itself, since we work with the physical
		 * density anyway.
		 */
		float numerical_density(float physical_density, int i) const;

		float pressure(float, int) const;

		float soundspeed(float, int) const;

		std::string const& get_dirname(void) const
		{ return m_problem_dir; }

		double set_deltap(const double dflt)
		{
			if (std::isfinite(m_options->deltap))
				m_deltap = m_options->deltap;
			else
				m_deltap = dflt;
			// also udate the smoothing length
			set_smoothing(simparams()->sfactor);

			return m_deltap;
		}

		double get_deltap() const
		{ return m_deltap; }

		float3 const& set_gravity(const float3 gravity)
		{
			physparams()->gravity = gravity;
			return physparams()->gravity;
		}

		float3 set_gravity(const float gravityX, const float gravityY, const float gravityZ)
		{
			return set_gravity(make_float3(gravityX, gravityY, gravityZ));
		}

		float3 set_gravity(const float gravityZ);

		float3 const& get_gravity() const
		{ return physparams()->gravity; }
		float get_gravity_magnitude() const
		{ return length(get_gravity()); }

		float set_timestep(const float dt)
		{
			simparams()->dt = dt;
			return dt;
		}

		float get_timestep() const
		{ return simparams()->dt; }

		// Compute the density diffusion coefficient
		void calculateDensityDiffusionCoefficient();

		/* set smoothing factor */
		double set_smoothing(const double smooth)
		{ return simparams()->set_smoothing(smooth, m_deltap); }

		/* set the expansion factor for the neighbor list search:
		 * when building the neighbor list, particles will be
		 * added to the list if they are within alpha*influenceRadius
		 * rather than just influenceRadius
		 * Returns the new neighbor search radius
		 */
		double set_neiblist_expansion(double alpha)
		{ return simparams()->set_neiblist_expansion(alpha); }

		/// Resize the neighbors list
		/*! Sets the per-particle neighbors list size so that it has
		 * nonvertex elements for fluid and boundary neighbors, and vertex elements
		 * for vertex neighbors. Due to the end-of-list markers, the effective
		 * number of nonvertex and vertex elements that can be stored is effectively
		 * reduced by 1 compare to the values set here.
		 */
		void resize_neiblist
			( int nonvertex ///< size of the non-vertex area of the neighbor list
			, int vertex = 0 ///< size of the vertex area of the neighbor list
			)
		{
			simparams()->neibboundpos = nonvertex - 1;
			simparams()->neiblistsize = vertex + nonvertex;
		}

		void set_kernel_radius(double radius)
		{ simparams()->set_kernel_radius(radius); }

		void set_grid_params(void);

		/// Compute the uniform grid coordinates of a point, clamping to edges
		int3 calc_grid_pos(const Point&) const;
		/// Compute the uniform grid components of a vector
		int3 calc_grid_offset(double3 const& vec) const;
		/// Compute the local (fractional grid cell) components of a vector,
		/// given the vector and its grid offset
		double3 calc_local_offset(double3 const& vec, int3 const& gridOff) const;

		uint calc_grid_hash(int3) const;

		void calc_localpos_and_hash(const Point&, const particleinfo&, float4&, hashKey&) const;

		// convert a double3 point into a grid + local position
		void calc_grid_and_local_pos(double3 const& globalPos, int3 *gridPos, float3 *localPos) const;

		inline
		const SimFramework* simframework(void) const
		{ return m_simframework; }

		inline
		SimFramework*& simframework(void)
		{ return m_simframework; }
		// add a filter (MLS, SHEPARD), with given frequency
		inline AbstractFilterEngine*
		addFilter(FilterType filtertype, int frequency)
		{ return simframework()->addFilterEngine(filtertype, frequency); };

		// add a post-processing filter, e.g.:
		// addPostProcess(CALC_PRIVATE);
		// addPostProcess(SURFACE_DETECTION); // simple surface detection
		// addPostProcess(SURFACE_DETECTION, BUFFER_NORMALS); // save normals too
		inline AbstractPostProcessEngine*
		addPostProcess(PostProcessType pptype, flag_t options=NO_FLAGS)
		{ return simframework()->addPostProcessEngine(pptype, options); }

		// check if a post process engine is enabled
		inline AbstractPostProcessEngine*
		hasPostProcess(PostProcessType pptype)
		{ return simframework()->hasPostProcessEngine(pptype); }


		inline
		const SimParams *simparams(void) const
		{ return m_simframework->simparams(); }

		inline
		SimParams *simparams(void)
		{ return m_simframework->simparams(); }

		inline
		const PhysParams *physparams(void) const
		{ return m_physparams; }

		PhysParams *physparams(void)
		{
			if (!m_physparams)
				m_physparams = new PhysParams(m_simframework->simparams()->rheologytype);
			return m_physparams;
		}

		// wrappers for physparams functions
		size_t add_fluid(float rho)
		{ return physparams()->add_fluid(rho); }
		void set_density(size_t fluid_idx, float _rho0)
		{ return physparams()->set_density(fluid_idx, _rho0); }
		float get_density(size_t fluid_idx)
		{ return physparams()->get_density(fluid_idx); }
		void set_equation_of_state(size_t fluid_idx, float gamma, float c0)
		{ return physparams()->set_equation_of_state(fluid_idx, gamma, c0); }

		void set_interface_epsilon(float eps)
		{ return physparams()->set_interface_epsilon(eps); }

		void set_artificial_visc(float artvisc)
		{ return physparams()->set_artificial_visc(artvisc); }

		void set_kinematic_visc(size_t fluid_idx, float nu)
		{ return physparams()->set_kinematic_visc(fluid_idx, nu); }
		void set_dynamic_visc(size_t fluid_idx, float mu)
		{ return physparams()->set_dynamic_visc(fluid_idx, mu); }
		void set_bulk_visc(size_t fluid_idx, float zeta)
		{ return physparams()->set_bulk_visc(fluid_idx, zeta); }
		void set_consistency_index(size_t fluid_idx, float k)
		{ return physparams()->set_consistency_index(fluid_idx, k); }
		void set_yield_strength(size_t fluid_idx, float ys)
		{ return physparams()->set_yield_strength(fluid_idx, ys); }
		void set_visc_power_law(size_t fluid_idx, float n)
		{ return physparams()->set_visc_power_law(fluid_idx, n); }
		void set_visc_exponential_coeff(size_t fluid_idx, float n)
		{ return physparams()->set_visc_exponential_coeff(fluid_idx, n); }
		void set_limiting_kinvisc(float max_visc)
		{ return physparams()->set_limiting_kinvisc(max_visc); }
		void set_sinpsi(size_t fluid_idx, float sinpsivalue)
		{ return physparams()->set_sinpsi(fluid_idx, sinpsivalue); }
		void set_cohesion(size_t fluid_idx, float cohesionvalue)
		{ return physparams()->set_cohesion(fluid_idx, cohesionvalue); }

		float get_kinematic_visc(size_t fluid_idx) const
		{ return physparams()->get_kinematic_visc(fluid_idx); }
		float get_dynamic_visc(size_t fluid_idx) const
		{ return physparams()->get_dynamic_visc(fluid_idx); }
		float get_consistency_index(size_t fluid_idx) const
		{ return physparams()->get_consistency_index(fluid_idx); }
		float get_yield_strength(size_t fluid_idx) const
		{ return physparams()->get_yield_strength(fluid_idx); }
		float get_visc_power_law(size_t fluid_idx) const
		{ return physparams()->get_visc_power_law(fluid_idx); }
		float get_visc_exponential_coeff(size_t fluid_idx) const
		{ return physparams()->get_visc_exponential_coeff(fluid_idx); }
		float get_sinpsi(size_t fluid_idx) const
		{ return physparams()->get_sinpsi(fluid_idx); }
		float get_cohesion(size_t fluid_idx) const
		{ return physparams()->get_cohesion(fluid_idx); }


		//! Add a wave gage at the given pt.{x, y} coordinates.
		/*! The gage will be a smoothing wave gage with smoothing length gage_smoothing > 0,
		 *  or a nearest-neighbor gage if gage_smoothing = 0.
		 *  The .y coordinate will be ignored in 2D
		 */
		void add_gage(double2 const& pt, double gage_smoothing=0);

		inline void add_gage(double3 const& pt, double gage_smoothing=0)
		{ add_gage(make_double2(pt.x, pt.y), gage_smoothing); }
		inline void add_gage(double x, double y, double gage_smoothing)
		{ add_gage(make_double2(x, y), gage_smoothing); }

		//! Add a wave gage.
		/*! The meaning of the second parameter depends on the dimensionality:
		 * In 3D, it will be interpreted as the second coordinate for the wave gage position.
		 * In 2D, it will be interpreted as the gage smoothing length.
		 */
		void add_gage(double x, double y_or_gs);

		//! Add a nearest-neighbor gage in 2D
		void add_gage(double x);


		/// Define a plane with equation ax + by + cz + d
		plane_t implicit_plane(double4 const& p);

		inline
		plane_t implicit_plane(double a, double b, double c, double d)
		{ return implicit_plane(make_double4(a, b, c, d)); }

		plane_t make_plane(Point const& pt, Vector const& normal);

		// add a new writer, with the given write frequency in (fractions of) seconds
		void add_writer(WriterType wt, double freq);

		// return the list of writers
		WriterList const& get_writers() const
		{ return m_writers; }

		/*!
		 overridden in subclasses if they want explicit writes
		 beyond those controlled by the writer(s) periodic time
		 */
		virtual bool need_write(double) const;


		/*!
		 overridden in subclasses if they want to write custom stuff
		 using the CALLBACKWRITER
		 */
		virtual void writer_callback(CallbackWriter *,
			uint numParts, BufferList const&, uint node_offset, double t,
			const bool testpoints) const;

		//! is the simulation running at the given time?
		virtual bool finished(double) const;

		//!
		virtual int fill_parts(bool fill = true) = 0;
		//! maximum number of particles that may be generated
		//! @userfunc
		//! User function for setting the maximum number of particles with IO.
		//! Activate it with IO boundaries.
		virtual uint max_parts(uint numParts);
		//!
		virtual void copy_to_array(BufferList & ) = 0;
		//!
		virtual void release_memory(void) = 0;

		//! Print information about particles that were set out of bounds during init
		void show_out_of_bounds() const;

		//!
		virtual void copy_planes(PlaneList& planes);

		/*! moving boundary and gravity callbacks */
		//! @userfunc
		//! Variable gravity definition
		virtual float3 g_callback(const double t);

		void allocate_bodies_storage();
		void add_moving_body(Object *, const MovingBodyType);
		void restore_moving_body(const MovingBodyData &, const uint, const int, const int);
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

		void InitializeChrono(void);
		void FinalizeChrono(void);

		// callback for initializing joints between Chrono bodies
		virtual void initializeObjectJoints();

		//!
		/*! This method can be overridden in problems when the object
		 * forces have to be altered in some way before being applied.
		 */
		virtual void
		bodies_forces_callback(const double t0, const double t1, const uint step, float3 *forces, float3 *torques);

		virtual void
		post_timestep_callback(const double t);

		//! @userfunc
		//! @label{Prescribe objects' motion}
		virtual void
		moving_bodies_callback(const uint index, Object* object, const double t0, const double t1,
							const float3& force, const float3& torque, const KinematicData& initial_kdata,
							KinematicData& kdata, double3& dx, EulerParameters& dr);

		void bodies_timestep(const float3 *forces, const float3 *torques, const int step,
							const double dt, const double t,
							int3 * & cgGridPos, float3 * & cgPos,
							float3 * & trans, float * & steprot,
							float3 * & linearvel, float3 * & angularvel);

		/*! Initialize the particle volumes */
		virtual void init_volume(BufferList &, uint numParticles);

		/* Initialize the internal energy */
		virtual void init_internal_energy(BufferList &, uint numParticles);

		/* Initialize k and epsilon */
		virtual void init_keps(BufferList &, uint numParticles);

		/* Initialize eddy viscosity */
		virtual void init_turbvisc(BufferList &, uint numParticles);

		/* Initialize effective pressure */
		virtual void init_effpres(BufferList &, uint numParticles);

		//! @userfunc
		//! @label{Prescribe custom open boundary conditions}
		virtual void imposeBoundaryConditionHost(
			BufferList&		bufwrite,
			const BufferList&	bufread,
					uint*			IOwaterdepth,
			const	float			t,
			const	uint			numParticles,
			const	uint			numOpenBoundaries,
			const	uint			particleRangeEnd);

		virtual void imposeForcedMovingObjects(
					float3	&gravityCenters,
					float3	&translations,
					float*	rotationMatrices,
			const	uint	ob,
			const	double	t,
			const	float	dt);

		/// Problem-specific implementation of CALC_PRIVATE
		/*! A problem requesting the CALC_PRIVATE post-processing filter
		 * MUST override this
		 */
		//! @userfunc
		//! @label{Custom user function}
		virtual void calcPrivate(flag_t options,
			BufferList const& bufread,
			BufferList & bufwrite,
			uint numParticles,
			uint particleRangeEnd,
			uint deviceIndex,
			const GlobalData * const gdata);

		/// Get the name to give to the private buffer(s)
		/*! A problem requesting the CALC_PRIVATE post-processing filter
		 * can override this if they want to provide a meaningful name
		 * for the BUFFER_PRIVATE ( and ...2 and ...4 variant, if used)
		 * buffer(s).
		 */
		virtual std::string get_private_name(flag_t buffer) const;

		//! Partition the grid in numDevices parts - virtual to allow problem or topology-specific implementations
		//! @userfunc
		//! @label{User function for the mutli-GPU domain splitting. Activate it if you have chosen a splitting option.}
		virtual void fillDeviceMap();
		// partition by splitting the cells according to their linearized hash
		void fillDeviceMapByCellHash();
		// partition by splitting along an axis. Default: along the longest

		/** @defpsubsection{split_axis, SPLIT_AXIS}
		 * @inpsection{domain_splitting}
		 * @default{x}
		 * @values{x,y,z}
		 * TLT_SPLIT_AXIS
		 */
		void fillDeviceMapByAxis(SplitAxis preferred_split_axis);
		// like fillDeviceMapByAxis(), but splits are proportional to the contained fluid particles
		void fillDeviceMapByAxisBalanced(SplitAxis preferred_split_axis);
		// partition by coordinates satistfying an example equation
		void fillDeviceMapByEquation();
		// partition by cutting the domain in parallelepipeds
		void fillDeviceMapByRegularGrid();
		// partition by performing the specified number of cuts along the three cartesian axes
		void fillDeviceMapByAxesSplits(uint Xslices, uint Yslices, uint Zslices);

		void PlaneCut(PointVect&, const double, const double, const double, const double);

		//! @userfunc
		//! callback for initializing particles with custom values
		virtual void initializeParticles(BufferList &buffers, const uint numParticles);
		//! callback for resetting the buffer values after resuming from a repack file
		virtual void resetBuffers(BufferList &buffers, const uint numParticles);
		void printBody(const uint bid);

};
#endif
