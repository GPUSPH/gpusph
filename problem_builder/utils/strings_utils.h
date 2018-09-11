#ifndef _STRINGS_UTILS_H_
#define _STRINGS_UTILS_H_

#include <algorithm>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#define SYSERROR()  GetLastError()
#else
#include <errno.h>
#define SYSERROR()  errno
#endif

/*! The path to the user problems sources directory relative to the destination
 * directory.
 */
#define USER_DIR "src/problems/user"

std::string getFileContent (const char*);

//! Find an indent before the given substring.
std::string getIndent( const std::string& theStr, const std::string& theSubstr );

void replaceAll(std::string& str, const std::string& from, const std::string& to);

std::string stringToUpper(std::string strToConvert);

std::string stringToMacro(std::string strToConvert);

/*!
 * Check if the string is the name like basename_index.
 * @param theStr the input string to check
 * @param theBaseName the output base name if found, otherwise the copy of the
 *  input string
 * @param theIndex the index if found, otherwise -1
 * @return true if the input string is like basename_index
 */
bool isIndexedSection( const std::string& theStr, std::string& theBaseName,
    int& theIndex );

namespace patch
{
	template < typename T > std::string to_string( const T& n )
	{
		std::ostringstream stm ;
		stm << n ;
		return stm.str() ;
	}
}


#endif
