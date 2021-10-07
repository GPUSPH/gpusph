/*  Copyright (c) 2011-2017 INGV, EDF, UniCT, JHU

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

/*! \file This file defines a bridge class between EulerParameter
 *  and the ChQuaternion class. Doing it with a bridge class such as this
 *  allows us to include and use EulerParameters without having to include Chrono headers
 */
#ifndef EULER_PARAMETERS_QUATERNION_H
#define EULER_PARAMETERS_QUATERNION_H

#include "EulerParameters.h"

#include "chrono_select.opt"
#if USE_CHRONO
#include "chrono/core/ChQuaternion.h"
#endif

//! Bridge class that can be used to convert to/from EulerParameter and ChQuaternion
class EulerParametersQuaternion
{

protected:
	double		m_ep[4];			///< Values of Euler parameters

	// We want EulerParameters to construct directly from us, to simplify syntax
	friend class EulerParameters;

public:
	EulerParametersQuaternion(EulerParameters const& ep)
	{ m_ep[0] = ep(0); m_ep[1] = ep(1); m_ep[2] = ep(2); m_ep[3] = ep(3); }

#if USE_CHRONO
	EulerParametersQuaternion(::chrono::ChQuaternion<> const& q)
	{ m_ep[0] = q.e0(); m_ep[1] = q.e1(); m_ep[2] = q.e2(); m_ep[3] = q.e3(); }

	// Conversion operator to ChQuaternion
	//operator EulerParameters() const { return EulerParameters(m_ep); }
	operator ::chrono::ChQuaternion<>() const { return ::chrono::ChQuaternion<>(m_ep[0], m_ep[1], m_ep[2], m_ep[3]); }
#endif

};

#endif

