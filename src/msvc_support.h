#ifdef _MSC_VER

#ifndef _MSVC_SUPPORT
#define _MSVC_SUPPORT

#include <direct.h>
#define mkdir(name, mode) _mkdir(name)

#include <process.h> //_getpid()
#define getpid _getpid

// not is not defined and will be replaced by !
#define not !

#endif // _MSVC_SUPPORT

#endif
