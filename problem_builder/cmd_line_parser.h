// Copyright (C) 2017 EDF R&D

#ifndef CMDLINEPARSER_H_
#define CMDLINEPARSER_H_

#include <string>

//! The parser of a command line.
class CmdLineParser
{
public:
  CmdLineParser( int argc, char **argv );
  CmdLineParser( int argc, const char **argv );
  virtual ~CmdLineParser();

  /*!
   * Get a value of the given option or zero pointer if not found.
   * @param theOption the option key
   * @return the option value or empty string if not found
   */
  char* getOption( const std::string & theOption ) const;

  /*!
   * Check if the option is present in the command line.
   * @param theOption the option key
   * @return true if the option is found
   */
  bool hasOption( const std::string& theOption ) const;

private:
  char** myBegin; //!< Start of command line arguments array
  char** myEnd; //!< End of command line arguments array
};

#endif /* CMDLINEPARSER_H_ */
