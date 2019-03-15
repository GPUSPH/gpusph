#ifdef _MSC_VER

#ifndef _MSVC_SUPPORT
#define _MSVC_SUPPORT

// Define the NOMINMAX symbol at the top of your source, before you include any headers,
// to prevents Visual C++ defining min and max as macros somewhere in windows.h,
// to avoid interferences with the corresponding standard functions.
#define NOMINMAX 

#include <direct.h>
#define mkdir(name, mode) _mkdir(name)

#include <process.h> //_getpid()
#define getpid _getpid

// not is not defined and will be replaced by !
#define not !

// strncasecmp is not defined, but we have _strnicmp
#define strncasecmp _strnicmp

#endif // _MSVC_SUPPORT

#endif
