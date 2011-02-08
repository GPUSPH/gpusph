/*
 * File:   Problem.h
 * Author: alexis
 *
 * Created on 13 juin 2008, 18:21
 */

#ifndef _PROBLEM_H
#define	_PROBLEM_H

#include <iostream>
#include <fstream>
#include <string>

#include "Options.h"
#include "particledefine.h"

using namespace std;

class Problem {
	private:
		float	m_last_display_time;
		float	m_last_write_time;
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

		float	m_displayinterval;
		int		m_writefreq;
		int		m_screenshotfreq;
		WriterType	m_writerType;

		float*	m_dem;
		int		m_ncols, m_nrows;

		string	m_name;

		Options		m_options;
		SimParams	m_simparams;
		PhysParams	m_physparams;
		MbCallBack	m_mbcallback;
		int			m_mbnumber;		// number of moving boundaries
		float4*		m_mbdata;		// mb data provided by problem to euler

		Problem(const Options &options = Options())
		{
			m_options = options;
			m_last_display_time = 0.0;
			m_last_write_time = 0.0;
			m_last_screenshot_time = 0.0;
			m_mbnumber = 0;
			m_mbdata = NULL;
		};

		~Problem(void)
		{
			if (m_mbdata)
				delete [] m_mbdata;
		};

		void allocate_mbdata(void)
		{
			m_mbdata = new float4[m_mbnumber];
		};

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

		PhysParams &get_physparams(void)
		{
			PhysParams &physparams = m_physparams;
			return physparams;
		};

//		void skipComment(ifstream &);
//		int read_init(char *fname);

		string create_problem_dir();
		bool need_display(float);
		bool need_write(float);
		bool need_screenshot(float);
		// is the simulation running at the given time?
		bool finished(float);

		virtual int fill_parts(void) = 0;
		virtual uint fill_planes(void);
		virtual void draw_boundary(float) = 0;
		virtual void copy_to_array(float4*, float4*, particleinfo*) = 0;
		virtual void copy_planes(float4*, float*);
		virtual void release_memory(void) = 0;
		virtual MbCallBack& mb_callback(const float, const float);
		virtual float4* get_mbdata(const float, const float);
};
#endif
