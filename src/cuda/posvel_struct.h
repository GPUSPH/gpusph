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

//! \file Introduce the split representation of the float4 for position and velocity

#ifndef POSVEL_STRUCT_H
#define POSVEL_STRUCT_H

//! Position is stored as a float4 with pos in the first three components and mass in the last
//! TODO this should be dimensionality-aware, at least for '.5' dimensions
struct pos_mass
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

inline __host__ __device__
bool constexpr is_active(pos_mass const& pdata)
{ return ::isfinite(pdata.mass); }

inline __host__ __device__
bool constexpr is_inactive(pos_mass const& pdata)
{ return !is_active(pdata); }

#endif // _BUILDNEIBS_PARAMS_H

