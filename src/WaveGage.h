/*  Copyright (c) 2021 INGV, EDF, UniCT, JHU

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

/*! \file
 * Wave gages
 */

#ifndef WAVE_GAGE_H
#define WAVE_GAGE_H

#include <cmath>
#include <cfloat>
#include <stdexcept>

#include "device_runtime_api.h"

//! The base wavegage class
/*! WaveGages work by looking at neighboring SURFACE particles and interpolating their z coordinates.
 *  The specific details about how the interpolation is done depend on the choice of gage.
 *  Currently we provide SmoothingWaveGage that uses a Shepard-corrected Wendland kernel
 *  (in one dimension less than the world dimensionality),
 *  and NearestNeighborWaveGage, that takes the value from the closest surface particle.
 */

class WaveGage {
protected:
	double depth;
public:
	WaveGage() : depth(0) {}
	virtual ~WaveGage() {}

	// add contribution from a particle at global position gpos
	virtual void add_particle(double4 const& gpos) = 0;

	// reset the wavegage
	virtual void initialize() { depth = 0; }
	// finalize the wavegage (e.g. normalize depth)
	// does nothing by default
	virtual void finalize() {}

	// get the gage depth
	double get_depth() const { return depth; }

	// get a 3D representation of the gage
	// (in 2D the z component should be 0)
	virtual double3 get_3D_pos() const = 0;
};

// Common part of dimension-dependent wavegages
template<int dim>
class WaveGagePos : public WaveGage
{
private:
	// WaveGage position: two components, even if the second component will be ignored
	// if dim == 2
	const double2 pos;

protected:
	using WaveGage::depth; // expose depth to derived classes

	double height(double4 const& gpos)
	{
		switch (dim) {
		case 3: return gpos.z;
		case 2: return gpos.y;
		}
		return NAN; // shouldn't happen
	}

	double distance(double4 const& gpos)
	{
		const double dx = pos.x - gpos.x;
		if (dim == 2) return fabs(dx); // 2D case: distance on the x axis

		const double dy = pos.y - gpos.y;
		return sqrt(dx*dx+dy*dy); // 3D case: distance on the xy plane
	}

public:
	double3 get_3D_pos() const override {
		return make_double3(pos.x,
			dim == 2 ? this->get_depth() : pos.y,
			dim == 3 ? this->get_depth() : 0);
	}

	WaveGagePos(double2 const& pos_) :
		WaveGage(),
		pos(pos_)
	{
		// Sanity check: no y component in 2D
		if (dim == 2 && pos.y != 0)
			throw std::invalid_argument("invalid gage specification (non-zero y pos in 2D)");
	}

};

// WaveGage implementing smoothing. templatized on the dimensionality of the world
// (the wavegage itself will use one less dimensions for the position and for the
// smoothing kernel, obviously)
template<int dim>
class SmoothingWaveGage : public WaveGagePos<dim>
{
	// Smoothing length for the kernel
	const double slength;
	// Influence radius for the kernel
	const double influenceradius;
	// Normalization factor for the kernel
	const double W_norm;

	// Smoothing kernel. Currently only WENDLAND is supported
	double W(double r)
	{
		const double q = r/slength;
		double t = 1 - q/2;
		t *= t;
		t *= t; // t = (1-q/2)^4
		return t*(2*q+1)*W_norm;
	}

	// Cumulative Shepard normalization factor
	double shepard_w;
public:
	SmoothingWaveGage(double2 const& pos, double h_) :
		WaveGagePos<dim>(pos),
		slength(h_),
		influenceradius(2.0*h_), // TODO kernel radius may be different for non-WENDLAND kernels
		W_norm(
			dim == 2 ? 3/(2*h_) : // 1D coefficient if world is 2D
			dim == 3 ? 7/(4*M_PI*h_*h_) : // 2D coefficient if world is 3D
			NAN) // this shouldn't happen
	{
		// Sanity check: W_norm should be finite:
		if (!std::isfinite(W_norm))
			throw std::invalid_argument("invalid gage specification (non-finite W norm)");
	};

	void add_particle(double4 const& gpos) override
	{
		const double r = this->distance(gpos);
		if (r >= influenceradius) return;
		double w = W(r);

		this->depth += this->height(gpos)*w;

		shepard_w += w;
	}

	void initialize() override
	{
		WaveGage::initialize();
		shepard_w = 0;
	}

	void finalize() override
	{
		if (shepard_w)
			this->depth /= shepard_w;
	}
};

// WaveGage implementing nearest-neighbor research. templatized on the dimensionality of the world
// (the wavegage itself will use one less dimensions for the position and for the
// smoothing kernel, obviously)
template<int dim>
class NearestNeighborWaveGage : public WaveGagePos<dim>
{
	// Minimum distance so far
	double min_dist;
public:
	NearestNeighborWaveGage(double2 const& pos) :
		WaveGagePos<dim>(pos)
	{};

	void add_particle(double4 const& gpos) override
	{
		const double r = this->distance(gpos);
		if (r >= min_dist) return;

		this->depth = this->height(gpos);

		min_dist = r;
	}

	void initialize() override
	{
		WaveGage::initialize();
		min_dist = DBL_MAX;
	}
};


#endif

