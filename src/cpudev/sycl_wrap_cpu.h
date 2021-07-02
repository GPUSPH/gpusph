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

#ifndef SYCL_WRAP_CPUDEV
#define SYCL_WRAP_CPUDEV

#include <stddef.h>

#include "cpp11_missing.h"

/*
 * SYCL-style wrappers
 *
 * The following structure, CUDA kernel template and host function template are
 * a simple gimmick that allow the definition of a SYCL-style calling convention
 */

//! Empty structures from which the current CUDA thread can be extracted using get_id()
//! This is effectively a simplified version of cl::sycl::item<1>, where the dimension
//! parameter to item.get_id(d) has been suppressed since we always assume one dimension only.
class simple_work_item
{
	unsigned m_id;
public:
	simple_work_item(unsigned i) : m_id(i) {}

	unsigned get_id() { return m_id; }
};

//! Wrapper kernel invocation function template
template<typename KernelClass>
void execute_kernel(KernelClass const& k, size_t numBlocks, size_t threadsPerBlock, size_t shMemSize = 0)
{
	// TODO UINT_MAX overflow check
	// (should never happen anyway since is an unsigned int at the moment anyway
	const size_t gws = numBlocks*threadsPerBlock;
#pragma omp parallel for
	for (unsigned i = 0; i < gws; ++i) {
		k(simple_work_item(i));
	}
}

#define COPY_TO_SYMBOL(dst, src, count) memcpy(&(dst), &(src), count*sizeof(src))

#define COPY_FROM_SYMBOL(dst, src, count) memcpy(&(dst), &(src), count*sizeof(dst))

// same thing
#define COPY_FROM_DEVICE(dst, src, count) memcpy(dst, src, count*sizeof(*(dst)))
#define COPY_TO_DEVICE(dst, src, count) memcpy(dst, src, count*sizeof(*(dst)))

#endif

