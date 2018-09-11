// Read an INI file into easy-to-access name/value pairs.

// inih and INIReader are released under the New BSD license (see LICENSE.txt).
// Go to the project home page for more info:
//
// http://code.google.com/p/inih/

#ifndef __INIREADER_H__
#define __INIREADER_H__

#include <map>
#include <string>

//! Section content: a map of names to values.
typedef std::map< std::string, std::string > IniSection;
typedef std::map< std::string, std::string >::iterator IniSection_Iter;
//! Map of a section name to its content
typedef std::map< std::string, IniSection > IniSections;
typedef std::map< std::string, IniSection >::iterator IniSections_Iter;
//! Map of index to indexed section content.
typedef std::map< int, IniSection > IndexedSection;
typedef std::map< int, IniSection >::iterator IndexedSection_Iter;
//! Map of indexed sections base names to their content.
typedef std::map< std::string, IndexedSection > IndexedSections;
typedef std::map< std::string, IndexedSection >::iterator IndexedSections_Iter;


// Read an INI file into easy-to-access name/value pairs. (Note that I've gone
// for simplicity here rather than speed, but it should be pretty decent.)
class INIReader
{
public:
    // Construct INIReader and parse given filename. See ini.h for more info
    // about the parsing.
    INIReader(std::string filename);

    // Return the result of ini_parse(), i.e., 0 on success, line number of
    // first error on parse error, or -1 on file open error.
    int ParseError();

    // Get a string value from INI file, returning default_value if not found.
    std::string Get(std::string section, std::string name,
                    std::string default_value);

    // Get an integer (long) value from INI file, returning default_value if
    // not found or not a valid integer (decimal "1234", "-1234", or hex "0x4d2").
    long GetInteger(std::string section, std::string name, long default_value);

    // Get a real (floating point double) value from INI file, returning
    // default_value if not found or not a valid floating point value
    // according to strtod().
    double GetReal(std::string section, std::string name, double default_value);

    // Get a boolean value from INI file, returning default_value if not found or if
    // not a valid true/false value. Valid true values are "true", "yes", "on", "1",
    // and valid false values are "false", "no", "off", "0" (not case sensitive).
    bool GetBoolean(std::string section, std::string name, bool default_value);

    // Get pure sections from INI file.
    IniSections GetSections() const;

    // Get sections named like <name>_<i> (i = 0,1,...).
    IndexedSections GetIndexedSections() const;

private:
    int _error;
    std::map<std::string, std::string> _values;
    IniSections mySections; //! Pure sections
    IndexedSections myIndexedSections; //! Indexed sections (named like <name>_<i>)
    static std::string MakeKey(std::string section, std::string name);
    static int ValueHandler(void* user, const char* section, const char* name,
                            const char* value);
};

#endif  // __INIREADER_H__
