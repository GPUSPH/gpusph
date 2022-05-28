/*  Copyright (c) 2011-2018 INGV, EDF, UniCT, JHU

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
 * Macros to mark functions as deprecated
 */

#ifndef _DEPRECATION_H
#define _DEPRECATION_H

// Since GPUSPH 3.0 we will try to preserve API between minor version
// of the software, in the sense that a problem written for GPUSPH M.0
// should be expected to also compile and run on GPUSPH M.x, up to the
// next major release.
// To this end, when a new (better) API is offered for some features,
// it should be carried side-by-side with the older, deprecated one.
// All the user-visible functions and variables of the deprecated API
// should still be offered, marked with the DEPRECRATED attribute,
// so that the user will know at compile time that a new API is being
// offered and that they should migrate.
// Deprecated functions and class methods should also print a message
// on stderr on (first) usage, so that even on execution the user
// is reminded to upgrade to the new API.

/* Mark a function deprecated */
#define GPUSPH_DEPRECATED __attribute__((deprecated))

/* Mark a function deprecated, explaining what to do instead,
 */

#define GPUSPH_DEPRECATED_MSG(str) __attribute__((deprecated(str)))

/* Mark an enum value as deprecated. In GCC this is only supported
 * from version 6 onwards, so for older version the macro doesn't do anything
 */

#if  __clang_major__ > 2 || __GNUC__ > 5
#define GPUSPH_DEPRECATED_VALUE(str) __attribute__((deprecated(str)))
#else
#define GPUSPH_DEPRECATED_VALUE(str) /* unsupported */
#endif

/* For the functions that provide compatibility between the deprecated
 * and new APIs, we want to avoid getting deprecation warnings,
 * since they are only for (all the) other uses of the deprecated APIs.
 * To achieve this, we would wrap the compatibility code in this way:

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
compatibility_code(calling, deprecated, functions);
and_or_assigning_to = obsolete_variables;
#pragma GCC diagnostic pop

 * which is only supported fully since GCC 4.6, and partially since 4.2.
 *
 * To have cleaner compiles with older versions of GCC (yes, some users have
 * reported using versions older than 4.2 (!)), we wrap these diagnostic
 * functions in a macro that does nothing before GCC 4.6, which we
 * use as cut-off for properly supported clean compilation.
 *
 * See also
 * http://dbp-consulting.com/tutorials/SuppressingGCCWarnings.html
 * for a more complete solution.
 */

// auxiliary macros to assemble the pragmas
#define GCC_DIAG_STR(s) #s
#define GCC_DIAG_JOINSTR(x,y) GCC_DIAG_STR(x ## y)
#define GCC_DIAG_DO_PRAGMA(x) _Pragma (#x)
#if __clang__ > 0
#define GCC_DIAG_PRAGMA(x) GCC_DIAG_DO_PRAGMA(clang diagnostic x)
#else
#define GCC_DIAG_PRAGMA(x) GCC_DIAG_DO_PRAGMA(GCC diagnostic x)
#endif

// the macros we will use in the code: IGNORE_WARNINGS and RESTORE_WARNINGS
#define IGNORE_WARNINGS(str) \
	GCC_DIAG_PRAGMA(push) \
	GCC_DIAG_PRAGMA(ignored GCC_DIAG_JOINSTR(-W, str))
#define RESTORE_WARNINGS \
	GCC_DIAG_PRAGMA(pop)
#endif
