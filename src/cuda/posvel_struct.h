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

//! \file Introduce the split representation of the float4 for position, velocity
//! and boundary elements
//! NOTE: all types here should have an imposed aligned of 16 bytes or nvcc may produce
//! incorrect load/store operations when restoring to register spliis.

#ifndef POSVEL_STRUCT_H
#define POSVEL_STRUCT_H

/*
 * Generic split
 */
struct __builtin_align__(16) space_w_t
{
	float3 xyz;
	float w;

	__host__ __device__
	space_w_t() = default;

	__host__ __device__
	space_w_t(float4 const& v) :
		xyz{v.x, v.y, v.z},
		w(v.w)
	{}
};

inline __host__ __device__
space_w_t operator-(float3 const& xyz, space_w_t const& other)
{
	space_w_t ret;
	ret.xyz = xyz - other.xyz;
	ret.w = other.w;
	return ret;
}


/*
 * Position and mass
 */

//! Position is stored as a float4 with pos in the first three components and mass in the last
//! TODO this should be dimensionality-aware, at least for '.5' dimensions
struct __builtin_align__(16) pos_mass
{
	float3 pos;
	float mass;

	__host__ __device__
	pos_mass(float4 const& pm4) :
		pos{pm4.x, pm4.y, pm4.z},
		mass(pm4.w)
	{}

	//! convert back to float4
	//! this is implicit because the pos_mass <> float4 conversion
	//! is roundtrip-safe (
	__host__ __device__
	operator float4() {
		return make_float4(pos, mass);
	}
};

inline __host__ __device__ void
disable_particle(pos_mass& pdata)
{ pdata.mass = NAN; }

//! Just like pos_mass, but the spatial member is called relPos
//! We don't provide a constructor for this, since this is assumed to be constructed
//! by difference
struct __builtin_align__(16) relPos_mass
{
	float3 relPos;
	float mass;
};

//! Add something to the space component
inline __host__ __device__
relPos_mass operator+(relPos_mass const& pm, float3 const& delta)
{
	relPos_mass ret(pm);
	ret.relPos += delta;
	return ret;
}

inline __host__ __device__
relPos_mass operator-(float3 const& pos, pos_mass const& neib_pm)
{
	relPos_mass ret;
	ret.relPos = pos - neib_pm.pos;
	ret.mass = neib_pm.mass;
	return ret;
}

inline __host__ __device__
//! central - neib => relPos + mass of the neib
relPos_mass operator-(pos_mass const& pm, pos_mass const& neib_pm)
{
	return pm.pos - neib_pm;
}


template<typename PosMass>
inline __host__ __device__
bool constexpr is_active(PosMass const& pdata)
{ return ::isfinite(pdata.mass); }

template<typename PosMass>
inline __host__ __device__
bool constexpr is_inactive(PosMass const& pdata)
{ return !is_active(pdata); }

/*
 * Velocity and density
 */

//! Velocity is stored as a float4 with vel in the first three components and rhotilde in the last
//! TODO this should be dimensionality-aware, at least for '.5' dimensions
struct __builtin_align__(16) vel_rho
{
	float3 vel;
	float rhotilde; // be clear that this is the relative density difference

	__host__ __device__
	vel_rho(float4 const& pm4) :
		vel{pm4.x, pm4.y, pm4.z},
		rhotilde(pm4.w)
	{}

	//! convert back to float4
	//! this is implicit because the pos_mass <> float4 conversion
	//! is roundtrip-safe (
	__host__ __device__
	operator float4() {
		return make_float4(vel, rhotilde);
	}

	__host__ __device__ __forceinline__
	vel_rho& operator+=(vel_rho const& other)
	{
		vel += other.vel;
		rhotilde += other.rhotilde;
		return *this;
	}

	__host__ __device__ __forceinline__
	vel_rho& operator/=(float div)
	{
		vel /= div;
		rhotilde /= div;
		return *this;
	}
};

//! Just like vel_rho, but the spatial member is called relVel
//! We don't provide a constructor for this, since this is assumed to be constructed
//! by difference
struct __builtin_align__(16) relVel_rho
{
	float3 relVel;
	float rhotilde;
};

inline __host__ __device__
relVel_rho operator-(float3 const& vel, vel_rho const& neib_pm)
{
	relVel_rho ret;
	ret.relVel = vel - neib_pm.vel;
	ret.rhotilde = neib_pm.rhotilde;
	return ret;
}

//! central - neib => relVel + rhotilde of the neib
inline __host__ __device__
relVel_rho operator-(vel_rho const& pm, vel_rho const& neib_pm)
{
	return pm.vel - neib_pm;
}

/*
 * Boundary elements
 */

struct __builtin_align__(16) belem_t
{
	float3 normal;
	float  area;

	__host__ __device__
	belem_t(float4 const& b) :
		normal{b.x, b.y, b.z},
		area(b.w)
	{}
};

static_assert(sizeof(space_w_t) == sizeof(float4), "incosistent xyz/w split");
static_assert(sizeof(pos_mass) == sizeof(float4), "incosistent pos/mass split");
static_assert(sizeof(vel_rho) == sizeof(float4), "incosistent vel/rho split");
static_assert(sizeof(belem_t) == sizeof(float4), "incosistent normal/area split");

#endif // _BUILDNEIBS_PARAMS_H

