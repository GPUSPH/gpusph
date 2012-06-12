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

#include "Options.h"
#include "RigidBody.h"
#include "particledefine.h"


using namespace std;

class Problem {
	private:
		float	m_last_display_time;
		float	m_last_write_time;
		float	m_last_rbdata_write_time;
		float	m_last_screenshot_time;
		string	m_problem_dir;

	public:
		enum WriterType
		{
			TEXTWRITER,
			VTKWRITER,
			VTKLEGACYWRITER,
			CUSTOMTEXTWRITER
		};

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

		float*	m_dem;
		int		m_ncols, m_nrows;

		string	m_name;

		Options		m_options;
		SimParams	m_simparams;
		PhysParams	m_physparams;
		MbCallBack	m_mbcallbackdata[MAXMOVINGBOUND];	// array of structure for moving boundary data
		int			m_mbnumber;							// number of moving boundaries

		RigidBody	*m_bodies;							// array of RigidBody objects
		float4		m_mbdata[MAXMOVINGBOUND];			// mb data to be provided by ParticleSystem to euler
		float3		m_bodies_cg[MAXBODIES];				// center of gravity of rigid bodies
		float3		m_bodies_trans[MAXBODIES];			// translation to apply between t and t + dt
		float		m_bodies_steprot[9*MAXBODIES];		// rotation to apply between t and t + dt

		Problem(const Options &options = Options());

		~Problem(void);

		Options get_options(void)
		{
			return m_options;
		}

		float3 get_worldorigin(void)
		{
			return m_origin;
		};

		float3 get_worldsize(void)
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
			if (isfinite(m_options.deltap))
				m_deltap = m_options.deltap;
			else
				m_deltap = dflt;
			return m_deltap;
		}

		SimParams &get_simparams(void)
		{
			SimParams &simparams = m_simparams;
			return simparams;
		};

		const PhysParams &get_physparams(void) const
		{
			return m_physparams;
		};

		string create_problem_dir();
		bool need_display(float);
		bool need_write(float);
		bool need_write_rbdata(float);
		void write_rbdata(float);
		bool need_screenshot(float);
		// is the simulation running at the given time?
		bool finished(float);

		virtual int fill_parts(void) = 0;
		virtual uint fill_planes(void);
		virtual void draw_boundary(float) = 0;
		virtual void draw_axis(void);
		virtual void copy_to_array(float4*, float4*, particleinfo*) = 0;
		virtual void copy_planes(float4*, float*);
		virtual void release_memory(void) = 0;
		virtual MbCallBack& mb_callback(const float, const float, const int);
		virtual float4* get_mbdata(const float, const float, const bool);
		virtual float3 g_callback(const float);

		void allocate_bodies(const int);
		RigidBody* get_body(const int);
		void get_rigidbodies_data(float3 * &, float * &);
		float3* get_rigidbodies_cg(void);
		float* get_rigidbodies_steprot(void);
		void rigidbodies_timestep(const float3 *, const float3 *, const int, 
									const double, float3 * &, float3 * &, float * &);
		int	get_bodies_numparts(void);
		int	get_body_numparts(const int);
};
#endif
