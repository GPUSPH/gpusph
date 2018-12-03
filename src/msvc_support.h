#ifndef _MSVC_SUPPORT
#define _MSVC_SUPPORT

#ifdef _MSC_VER
#include <direct.h>
#define mkdir(name, mode) _mkdir(name)
#endif

#endif
