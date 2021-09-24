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

//! 24-bit integer multiplication on 1.x was faster than simple integer multiplication,
//! so wrap the multiplication in a macro.
//! NOTE: we could probably just get rid of the macro at this point, since AFAIK
//! there are no more devices around where 24-bit multiplication is faster than 32-bit.
#if CUDA_BACKEND_ENABLED && (__COMPUTE__ < 20)
	#define INTMUL(x,y) __mul24(x,y)
#else
	#define INTMUL(x,y) (x)*(y)
#endif


