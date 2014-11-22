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

#include <sstream>
#include <stdexcept>
#include <string>
#include <cmath>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "Problem.h"
#include "vector_math.h"
#include "vector_print.h"
#include "utils.h"

// here we need the complete definition of the GlobalData struct
#include "GlobalData.h"

// COORD1, COORD2, COORD3
#include "linearization.h"

uint Problem::m_total_ODE_bodies = 0;

Problem::Problem(const GlobalData *_gdata)
{
	gdata = _gdata;
	m_options = gdata->clOptions;
	m_mbnumber = 0;
	m_rbdatafile = NULL;
	m_rbdata_writeinterval = 0;
	memset(m_mbcallbackdata, 0, MAXMOVINGBOUND*sizeof(float4));
	m_ODE_bodies = NULL;
	m_problem_dir = m_options->dir;
}


Problem::~Problem(void)
{
	if (m_ODE_bodies)
		delete [] m_ODE_bodies;
	if (m_rbdatafile != NULL) {
        fclose(m_rbdatafile);
    }
}

void
Problem::check_dt(void)
{
	float dt_from_sspeed = INFINITY;
	for (uint f = 0 ; f < m_physparams.numFluids; ++f) {
		float sspeed = m_physparams.sscoeff[f];
		dt_from_sspeed = fmin(dt_from_sspeed, m_simparams.slength/sspeed);
	}
	dt_from_sspeed *= m_simparams.dtadaptfactor;

	float dt_from_gravity = sqrt(m_simparams.slength/length(m_physparams.gravity));
	dt_from_gravity *= m_simparams.dtadaptfactor;

	float dt_from_visc = NAN;
	if (m_simparams.visctype != ARTVISC) {
		dt_from_visc = m_simparams.slength*m_simparams.slength/m_physparams.kinematicvisc;
		dt_from_visc *= 0.125f; // TODO this should be configurable
	}

	float cfl_dt = fminf(dt_from_sspeed, fminf(dt_from_gravity, dt_from_visc));

	if (m_simparams.dt > cfl_dt) {
		fprintf(stderr, "WARNING: dt %g bigger than %g imposed by CFL conditions (sspeed: %g, gravity: %g, viscosity: %g)\n",
			m_simparams.dt, cfl_dt,
			dt_from_sspeed, dt_from_gravity, dt_from_visc);
	} else if (!m_simparams.dt) { // dt wasn't set
			m_simparams.dt = cfl_dt;
			printf("setting dt = %g from CFL conditions (soundspeed: %g, gravity: %g, viscosity: %g)\n",
				m_simparams.dt,
				dt_from_sspeed, dt_from_gravity, dt_from_visc);
	} else {
			printf("dt = %g (CFL conditions from soundspeed: %g, from gravity %g, from viscosity %g)\n",
				m_simparams.dt,
				dt_from_sspeed, dt_from_gravity, dt_from_visc);
	}

}

void
Problem::check_maxneibsnum(void)
{
	// kernel radius times smoothing factor, rounded to the next integer
	double r = m_simparams.sfactor*m_simparams.kernelradius;
	r = ceil(r);

	// volumes are computed using a coefficient which is sligthly more than π
#define PI_PLUS_EPS 3.2
	double vol = 4*PI_PLUS_EPS*r*r*r/3;
	// and rounded up
	vol = ceil(vol);

	// maxneibsnum is obtained rounding up the volume to the next
	// multiple of 32
	uint maxneibsnum = round_up((uint)vol, 32U);

	// with semi-analytical boundaries, boundary particles
	// are doubled, so we expand by a factor of 1.5,
	// again rounding up
	if (m_simparams.boundarytype == SA_BOUNDARY)
		maxneibsnum = round_up(3*maxneibsnum/2, 32U);

	// more in general, it's possible to have different particle densities for the
	// boundaries even with other boundary conditions. we do not have a universal
	// parameter that marks the inter-particle distance for boundary particles,
	// although we know that r0 is normally used for this too.
	// TODO FIXME when the double meaning of r0 as inter-particle distance for
	// boundary particles and as fluid-boundary distance is split into separate
	// variables, the inter-particle distance should be used in the next formula

	// The formula we use is based on the following:
	// 1. a half-sphere has (3/2) pi r^3 particle
	// 2. a full circle has pi (r/q)^2 particles, if q is the ratio beween
	//   the inter-particle distance on the full circle and the inter-particle
	//   distance used in the fluid
	// * the number of neighbors that are seen by a particle which is near
	//   a boundary plane with q*dp interparticle-distance is augmented the number
	//   in 2. over the number in 1., giving (3/2) (1/q)^2 (1/r)
	// * of course this does not affect the entire neighborhood, but only the part
	//   which is close to a boundary, which we estimate to be at most 2/3rds of
	//   the neighborhood, which cancels with the (3/2) factor
	//   TODO check if we should assume 7/8ths instead (particle near vertex
	//   only has 1/8th of a sphere in the fluid, the rest is all boundaries).
	double qq = m_deltap/m_physparams.r0; // 1/q
	// double ratio = fmax((21*qq*qq)/(16*r), 1.0); // if we assume 7/8
	double ratio = fmax((qq*qq)/r, 1.0); // only use this if it gives us _more_ particles
	// increase maxneibsnum as appropriate
	maxneibsnum = (uint)ceil(ratio*maxneibsnum);
	// round up to multiple of 32
	maxneibsnum = round_up(maxneibsnum, 32U);

	// if the maxneibsnum was user-set, check against computed minimum
	if (m_simparams.maxneibsnum) {
		if (m_simparams.maxneibsnum < maxneibsnum) {
			fprintf(stderr, "WARNING: problem-set max neibs num too low! %u < %u\n",
				m_simparams.maxneibsnum, maxneibsnum);
		} else {
			printf("Using problem-set max neibs num %u (safe computed value was %u)\n",
				m_simparams.maxneibsnum, maxneibsnum);
		}
	} else {
		printf("Using computed max neibs num %u\n", maxneibsnum);
		m_simparams.maxneibsnum = maxneibsnum;
	}
}


float
Problem::density(float h, int i) const
{
	float density = m_physparams.rho0[i];

	if (h > 0) {
		//float g = length(m_physparams.gravity);
		float g = abs(m_physparams.gravity.z);
		density = m_physparams.rho0[i]*pow(g*m_physparams.rho0[i]*h/m_physparams.bcoeff[i] + 1,
				1/m_physparams.gammacoeff[i]);
		}
	return density;
}

float
Problem::soundspeed(float rho, int i) const
{
	return m_physparams.sscoeff[i]*pow(rho/m_physparams.rho0[i], m_physparams.sspowercoeff[i]);
}


float
Problem::pressure(float rho, int i) const
{
	return m_physparams.bcoeff[i]*(pow(rho/m_physparams.rho0[i], m_physparams.gammacoeff[i]) - 1);
}

void
Problem::add_gage(double3 const& pt)
{
	m_simparams.gage.push_back(pt);
}

std::string const&
Problem::create_problem_dir(void)
{
	// if no data save directory was specified, default to a name
	// composed of problem name followed by date and time
	if (m_problem_dir.empty()) {
		time_t  rawtime;
		char	time_str[18];

		time(&rawtime);
		strftime(time_str, 18, "_%Y-%m-%dT%Hh%M", localtime(&rawtime));
		time_str[17] = '\0';
		// if "./tests/" doesn't exist yet...
		mkdir("./tests/", S_IRWXU | S_IRWXG | S_IRWXO);
		m_problem_dir = "./tests/" + m_name + std::string(time_str);
	}

	// TODO it should be possible to specify a directory with %-like
	// replaceable strings, such as %{problem} => problem name,
	// %{time} => launch time, etc.

	mkdir(m_problem_dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);

	if (m_rbdata_writeinterval) {
		string rbdata_filename = m_problem_dir + "/rbdata.txt";
		m_rbdatafile = fopen(rbdata_filename.c_str(), "w");

		if (m_rbdatafile == NULL) {
			stringstream ss;
			ss << "Cannot open rigid bodies data file " << rbdata_filename;
			throw runtime_error(ss.str());
			}
	}
	return m_problem_dir;
}

void
Problem::set_timer_tick(float t)
{
	Writer::SetTimerTick(t);
}

void
Problem::add_writer(WriterType wt, int freq)
{
	m_writers.push_back(make_pair(wt, freq));
}

// override in problems where you want to save
// at specific times regardless of standard conditions
bool
Problem::need_write(double t) const
{
	return false;
}

bool
Problem::need_write_rbdata(double t) const
{
	if (m_rbdata_writeinterval == 0)
		return false;

	if (t - m_last_rbdata_write_time >= m_rbdata_writeinterval) {
		return true;
	}

	return false;
}


void
Problem::write_rbdata(double t)
{
	if (m_simparams.numODEbodies) {
		if (need_write_rbdata(t)) {
			for (uint i = 1; i < m_simparams.numODEbodies; i++) {
				const dReal* quat = dBodyGetQuaternion(m_ODE_bodies[i]->m_ODEBody);
				const dReal* cg = dBodyGetPosition(m_ODE_bodies[i]->m_ODEBody);
				fprintf(m_rbdatafile, "%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", i, t, cg[0],
						cg[1], cg[2], quat[0], quat[1], quat[2], quat[3]);
			}
		}
	}
	m_last_rbdata_write_time = t;
}

// is the simulation finished at the given time?
bool
Problem::finished(double t) const
{
	double tend(m_simparams.tend);
	return tend && (t > tend);
}


MbCallBack&
Problem::mb_callback(const float t, const float dt, const int i)
{
	return m_mbcallbackdata[i];
};


float3
Problem::g_callback(const float t)
{
	return make_float3(0.0);
}

// Fill the device map with "devnums" (*global* device ids) in range [0..numDevices[.
// Default algorithm: split along the longest axis
void Problem::fillDeviceMap()
{
	fillDeviceMapByAxis(LONGEST_AXIS);
}

// partition by splitting the cells according to their linearized hash.
void Problem::fillDeviceMapByCellHash()
{
	uint cells_per_device = gdata->nGridCells / gdata->totDevices;
	for (uint i=0; i < gdata->nGridCells; i++)
		gdata->s_hDeviceMap[i] = min( i/cells_per_device, gdata->totDevices-1);
}

// partition by splitting along the specified axis
void Problem::fillDeviceMapByAxis(SplitAxis preferred_split_axis)
{
	// select the longest axis
	if (preferred_split_axis == LONGEST_AXIS) {
		if (	gdata->worldSize.x >= gdata->worldSize.y &&
				gdata->worldSize.x >= gdata->worldSize.z)
			preferred_split_axis = X_AXIS;
		else
		if (	gdata->worldSize.y >= gdata->worldSize.z)
			preferred_split_axis = Y_AXIS;
		else
			preferred_split_axis = Z_AXIS;
	}
	uint cells_per_longest_axis = 0;
	switch (preferred_split_axis) {
		case X_AXIS:
			cells_per_longest_axis = gdata->gridSize.x;
			break;
		case Y_AXIS:
			cells_per_longest_axis = gdata->gridSize.y;
			break;
		case Z_AXIS:
			cells_per_longest_axis = gdata->gridSize.z;
			break;
	}
	uint cells_per_device_per_longest_axis = (uint)round(cells_per_longest_axis / (float)gdata->totDevices);
	/*
	printf("Splitting domain along axis %s, %u cells per part\n",
		(preferred_split_axis == X_AXIS ? "X" : (preferred_split_axis == Y_AXIS ? "Y" : "Z") ), cells_per_device_per_longest_axis);
	*/
	for (uint cx = 0; cx < gdata->gridSize.x; cx++)
		for (uint cy = 0; cy < gdata->gridSize.y; cy++)
			for (uint cz = 0; cz < gdata->gridSize.z; cz++) {
				uint axis_coordinate;
				switch (preferred_split_axis) {
					case X_AXIS: axis_coordinate = cx; break;
					case Y_AXIS: axis_coordinate = cy; break;
					case Z_AXIS: axis_coordinate = cz; break;
				}
				// everything is just a preparation for the following line
				uchar dstDevice = axis_coordinate / cells_per_device_per_longest_axis;
				// handle the case when cells_per_longest_axis multiplies cells_per_longest_axis
				dstDevice = min(dstDevice, gdata->totDevices - 1);
				// compute cell address
				uint cellLinearHash = gdata->calcGridHashHost(cx, cy, cz);
				// assign it
				gdata->s_hDeviceMap[cellLinearHash] = dstDevice;
			}
}

void Problem::fillDeviceMapByEquation()
{
	// 1st equation: diagonal plane. (x+y+z)=coeff
	//uint longest_grid_size = max ( max( gdata->gridSize.x, gdata->gridSize.y), gdata->gridSize.z );
	uint coeff = (gdata->gridSize.x + gdata->gridSize.y + gdata->gridSize.z) / gdata->totDevices;
	// 2nd equation: sphere. Sqrt(cx²+cy²+cz²)=radius
	uint diagonal = (uint) sqrt(	gdata->gridSize.x * gdata->gridSize.x +
									gdata->gridSize.y * gdata->gridSize.y +
									gdata->gridSize.z * gdata->gridSize.z) / 2;
	uint radius_part = diagonal /  gdata->totDevices;
	for (uint cx = 0; cx < gdata->gridSize.x; cx++)
		for (uint cy = 0; cy < gdata->gridSize.y; cy++)
			for (uint cz = 0; cz < gdata->gridSize.z; cz++) {
				uint dstDevice;
				// 1st equation: rough oblique plane split --
				dstDevice = (cx + cy + cz) / coeff;
				// -- end of 1st eq.
				// 2nd equation: spheres --
				//uint distance_from_origin = (uint) sqrt( cx * cx + cy * cy + cz * cz);
				// comparing directly the square would be more efficient but could require long uints
				//dstDevice = distance_from_origin / radius_part;
				// -- end of 2nd eq.
				// handle special cases at the edge
				dstDevice = min(dstDevice, gdata->totDevices - 1);
				// compute cell address
				uint cellLinearHash = gdata->calcGridHashHost(cx, cy, cz);
				// assign it
				gdata->s_hDeviceMap[cellLinearHash] = (uchar)dstDevice;
			}
}

// Partition by performing the splitting the domain in the specified number of slices for each axis.
// Values must be > 0. The number of devices will be the product of the input values.
// This is not meant to be called directly by a problem since the number of splits (and thus the devices)
// would be hardocded. A wrapper method (like fillDeviceMapByRegularGrid) can provide an algorithm to
// properly factorize a given number of GPUs in 2 or 3 values.
void Problem::fillDeviceMapByAxesSplits(uint Xslices, uint Yslices, uint Zslices)
{
	// is any of these zero?
	if (Xslices * Yslices * Zslices == 0)
		printf("WARNING: fillDeviceMapByAxesSplits() called with zero values, using 1 instead");

	if (Xslices == 0) Xslices = 1;
	if (Yslices == 0) Yslices = 1;
	if (Zslices == 0) Zslices = 1;

	// divide and round
	uint devSizeCellsX = (gdata->gridSize.x + Xslices - 1) / Xslices ;
	uint devSizeCellsY = (gdata->gridSize.y + Yslices - 1) / Yslices ;
	uint devSizeCellsZ = (gdata->gridSize.z + Zslices - 1) / Zslices ;

	// iterate on all cells
	for (uint cx = 0; cx < gdata->gridSize.x; cx++)
			for (uint cy = 0; cy < gdata->gridSize.y; cy++)
				for (uint cz = 0; cz < gdata->gridSize.z; cz++) {

				// where are we in the 3D grid of devices?
				uint whichDevCoordX = (cx / devSizeCellsX);
				uint whichDevCoordY = (cy / devSizeCellsY);
				uint whichDevCoordZ = (cz / devSizeCellsZ);

				// round if needed
				if (whichDevCoordX == Xslices) whichDevCoordX--;
				if (whichDevCoordY == Yslices) whichDevCoordY--;
				if (whichDevCoordZ == Zslices) whichDevCoordZ--;

				// compute dest device
				uint dstDevice = whichDevCoordZ * Yslices * Xslices + whichDevCoordY * Xslices + whichDevCoordX;
				// compute cell address
				uint cellLinearHash = gdata->calcGridHashHost(cx, cy, cz);
				// assign it
				gdata->s_hDeviceMap[cellLinearHash] = (uchar)dstDevice;
			}
}

// Wrapper for fillDeviceMapByAxesSplits() computing the number of cuts along each axis.
// WARNING: assumes the total number of devices is divided by a combination of 2, 3 and 5
void Problem::fillDeviceMapByRegularGrid()
{
	float Xsize = gdata->worldSize.x;
	float Ysize = gdata->worldSize.y;
	float Zsize = gdata->worldSize.z;
	uint cutsX = 1;
	uint cutsY = 1;
	uint cutsZ = 1;
	uint remaining_factors = gdata->totDevices;

	// define the product of non-zero cuts to keep track of current number of parallelepipeds
//#define NZ_PRODUCT	((cutsX > 0? cutsX : 1) * (cutsY > 0? cutsY : 1) * (cutsZ > 0? cutsZ : 1))

	while (cutsX * cutsY * cutsZ < gdata->totDevices) {
		uint factor = 1;
		// choose the highest factor among 2, 3 and 5 which divides remaining_factors
		if (remaining_factors % 5 == 0) factor = 5; else
		if (remaining_factors % 3 == 0) factor = 3; else
		if (remaining_factors % 2 == 0) factor = 2; else {
			factor = remaining_factors;
			printf("WARNING: splitting by regular grid but %u is not divided by 2,3,5!\n", remaining_factors);
		}
		// choose the longest axis to split along
		if (Xsize >= Ysize && Xsize >= Zsize) {
			Xsize /= factor;
			cutsX *= factor;
		} else
		if (Ysize >= Xsize && Ysize >= Zsize) {
			Ysize /= factor;
			cutsY *= factor;
		} else {
			Zsize /= factor;
			cutsZ *= factor;
		}
	}

	// should always hold, but double check for bugs
	if (cutsX * cutsY * cutsZ != gdata->totDevices)
		printf("WARNING: splitting by regular grid but final distribution (%u, %u, %u) does not produce %u parallelepipeds!\n",
			cutsX, cutsY, cutsZ, gdata->totDevices);

	fillDeviceMapByAxesSplits(cutsX, cutsY, cutsZ);
}

void
Problem::allocate_ODE_bodies(const uint i)
{
	m_simparams.numODEbodies = i;
	m_ODE_bodies = new Object *[i];
}


Object*
Problem::get_ODE_body(const uint i)
{
	if (i >= m_simparams.numODEbodies) {
		stringstream ss;
		ss << "get_ODE_body: body number " << i << " >= numbodies";
		throw runtime_error(ss.str());
	}
	return m_ODE_bodies[i];
}


void
Problem::add_ODE_body(Object* object)
{
	if (m_total_ODE_bodies >= m_simparams.numODEbodies) {
		stringstream ss;
		ss << "add_ODE_body: body number " << m_total_ODE_bodies << " >= numbodies";
		throw runtime_error(ss.str());
	}
	m_ODE_bodies[m_total_ODE_bodies] = object;
	m_total_ODE_bodies++;
}


int
Problem::get_ODE_bodies_numparts(void) const
{
	int total_parts = 0;
	for (uint i = 0; i < m_simparams.numODEbodies; i++) {
		total_parts += m_ODE_bodies[i]->GetParts().size();
	}

	return total_parts;
}


int
Problem::get_ODE_body_numparts(const int i) const
{
	if (!m_simparams.numODEbodies)
		return 0;

	return m_ODE_bodies[i]->GetParts().size();
}


void
Problem::get_ODE_bodies_data(float3 * & cg, float * & steprot)
{
	cg = m_bodies_cg;
	steprot = m_bodies_steprot;
}


float3*
Problem::get_ODE_bodies_cg(void)
{
	for (uint i = 0; i < m_simparams.numODEbodies; i++)  {
		m_bodies_cg[i] = make_float3(dBodyGetPosition(m_ODE_bodies[i]->m_ODEBody));
	}

	return m_bodies_cg;
}


float*
Problem::get_ODE_bodies_steprot(void)
{
	return m_bodies_steprot;
}


void
Problem::ODE_bodies_timestep(const float3 *force, const float3 *torque, const int step,
		const double dt, float3 * & cg, float3 * & trans, float * & steprot)
{
	dReal prev_quat[MAXBODIES][4];
	for (uint i = 0; i < m_total_ODE_bodies; i++)  {
		const dReal* quat = dBodyGetQuaternion(m_ODE_bodies[i]->m_ODEBody);
		prev_quat[i][0] = quat[0];
		prev_quat[i][1] = quat[1];
		prev_quat[i][2] = quat[2];
		prev_quat[i][3] = quat[3];
		dBodyAddForce(m_ODE_bodies[i]->m_ODEBody, force[i].x, force[i].y, force[i].z);
		dBodyAddTorque(m_ODE_bodies[i]->m_ODEBody, torque[i].x, torque[i].y, torque[i].z);
	}

	dSpaceCollide(m_ODESpace, (void *) this, &ODE_near_callback_wrapper);
	dWorldStep(m_ODEWorld, dt);
	if (m_ODEJointGroup)
		dJointGroupEmpty(m_ODEJointGroup);

	for (uint i = 0; i < m_simparams.numODEbodies; i++)  {
		float3 new_cg = make_float3(dBodyGetPosition(m_ODE_bodies[i]->m_ODEBody));
		m_bodies_trans[i] = new_cg - m_bodies_cg[i];
		m_bodies_cg[i] = new_cg;
		const dReal *new_quat = dBodyGetQuaternion(m_ODE_bodies[i]->m_ODEBody);
		dQuaternion step_quat;
		dMatrix3 R;
		dQMultiply2 (step_quat, new_quat, prev_quat[i]);
		dQtoR (step_quat, R);
		float *base_addr = m_bodies_steprot + 9*i;
		base_addr[0] = R[0];
		base_addr[1] = R[1];
		base_addr[2] = R[2]; // Skipp R[3]
		base_addr[3] = R[4];
		base_addr[4] = R[5];
		base_addr[5] = R[6]; // Skipp R[7]
		base_addr[6] = R[8];
		base_addr[7] = R[9];
		base_addr[8] = R[10];
	}
	cg = m_bodies_cg;
	steprot = m_bodies_steprot;
	trans = m_bodies_trans;
}

// Number of planes
uint
Problem::fill_planes(void)
{
	return 0;
}


// Copy planes for upload
void
Problem::copy_planes(float4*, float*)
{
	return;
}


float4*
Problem::get_mbdata(const float t, const float dt, const bool forceupdate)
{
	bool needupdate = false;

	for (int i=0; i < m_mbnumber; i++) {
		MbCallBack& mbcallbackdata = mb_callback(t, dt, i);
		float4 data = make_float4(0.0f);

		switch(mbcallbackdata.type) {
			case PISTONPART:
				data.x = mbcallbackdata.vel.x;
				break;

			case PADDLEPART:
				data.x = mbcallbackdata.origin.x;
				data.y = mbcallbackdata.origin.z;
				data.z = mbcallbackdata.dthetadt;
				break;

			case GATEPART:
				data.x = mbcallbackdata.vel.x;
				data.y = mbcallbackdata.vel.y;
				data.z = mbcallbackdata.vel.z;
				break;
		}
		if (m_mbdata[i].x != data.x || m_mbdata[i].y != data.y ||
			m_mbdata[i].z != data.z ||m_mbdata[i].w != data.w) {
			m_mbdata[i] = data;
			needupdate = true;
			}
	}

	if (needupdate || forceupdate)
		return m_mbdata;

	return NULL;
}

/*! Compute grid and cell size from the kernel influence radius
 * The number of cell is obtained as the ratio between the domain size and the
 * influence radius, rounded down to the closest integer.
 * The reason for rounding down is that we want the cell size to be no smaller
 * than the influence radius, to guarantee that all neighbors of a particle are
 * found at most one cell away in each direction.
 */
void
Problem::set_grid_params(void)
{
	double influenceRadius = m_simparams.kernelradius*m_simparams.slength;
	// with semi-analytical boundaries, we want a cell size which is
	// deltap/2 + the usual influence radius
	double cellSide = influenceRadius;
	if (m_simparams.boundarytype == SA_BOUNDARY)
		cellSide += m_deltap/2.0f;

	m_gridsize.x = floor(m_size.x / cellSide);
	m_gridsize.y = floor(m_size.y / cellSide);
	m_gridsize.z = floor(m_size.z / cellSide);

	// While trying to run a simulation at very low resolution, the user might
	// set a deltap so large that cellSide is bigger than m_size.{x,y,z}, resulting
	// in a corresponding gridsize of 0. Check for this case (by checking if any
	// of the gridsize components are zero) and throw.

	if (!m_gridsize.x || !m_gridsize.y || !m_gridsize.z) {
		stringstream ss;
		ss << "resolution " << m_simparams.slength << " is too low! Resulting grid size would be "
			<< m_gridsize;
		throw runtime_error(ss.str());
	}

	m_cellsize.x = m_size.x / m_gridsize.x;
	m_cellsize.y = m_size.y / m_gridsize.y;
	m_cellsize.z = m_size.z / m_gridsize.z;

	/*
	printf("set_grid_params\t:\n");
	printf("Domain size\t: (%f, %f, %f)\n", m_size.x, m_size.y, m_size.z);
	*/
	printf("Influence radius / expected cell side\t: %g, %g\n", influenceRadius, cellSide);
	/*
	printf("Grid   size\t: (%d, %d, %d)\n", m_gridsize.x, m_gridsize.y, m_gridsize.z);
	printf("Cell   size\t: (%f, %f, %f)\n", m_cellsize.x, m_cellsize.y, m_cellsize.z);
	printf("       delta\t: (%.2f%%, %.2f%%, %.2f%%)\n",
		(m_cellsize.x - cellSide)*100/cellSide,
		(m_cellsize.y - cellSide)*100/cellSide,
		(m_cellsize.z - cellSide)*100/cellSide);
	*/
}


// Compute position in uniform grid (clamping to edges)
int3
Problem::calc_grid_pos(const Point&	pos)
{
	int3 gridPos;
	gridPos.x = floor((pos(0) - m_origin.x) / m_cellsize.x);
	gridPos.y = floor((pos(1) - m_origin.y) / m_cellsize.y);
	gridPos.z = floor((pos(2) - m_origin.z) / m_cellsize.z);
	gridPos.x = min(max(0, gridPos.x), m_gridsize.x-1);
	gridPos.y = min(max(0, gridPos.y), m_gridsize.y-1);
	gridPos.z = min(max(0, gridPos.z), m_gridsize.z-1);

	return gridPos;
}


// Compute address in grid from position
uint
Problem::calc_grid_hash(int3 gridPos)
{
	return gridPos.COORD3 * m_gridsize.COORD2 * m_gridsize.COORD1 + gridPos.COORD2 * m_gridsize.COORD1 + gridPos.COORD1;
}


void
Problem::calc_localpos_and_hash(const Point& pos, const particleinfo& info, float4& localpos, hashKey& hash)
{
	int3 gridPos = calc_grid_pos(pos);

	// automatically choose between long hash (cellHash + particleId) and short hash (cellHash)
	hash = makeParticleHash( calc_grid_hash(gridPos), info );

	localpos.x = float(pos(0) - m_origin.x - (gridPos.x + 0.5)*m_cellsize.x);
	localpos.y = float(pos(1) - m_origin.y - (gridPos.y + 0.5)*m_cellsize.y);
	localpos.z = float(pos(2) - m_origin.z - (gridPos.z + 0.5)*m_cellsize.z);
	localpos.w = float(pos(3));
}

void
Problem::init_keps(float* k, float* e, uint numpart, particleinfo* info)
{
	const float Lm = fmax(2*m_deltap, 1e-5f);
	const float k0 = pow(0.002f*m_physparams.sscoeff[0], 2);
	const float e0 = 0.16f*pow(k0, 1.5f)/Lm;

	for (uint i = 0; i < numpart; i++) {
		k[i] = k0;
		e[i] = e0;
	}
}
