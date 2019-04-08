// Copyright (C) 2017 EDF R&D

#include "cmd_line_parser.h"
#include <algorithm>

CmdLineParser::CmdLineParser( int argc, char **argv )
{
  myBegin = argv;
  myEnd = myBegin + argc;
}

CmdLineParser::CmdLineParser( int argc, const char **argv )
{
  myBegin = const_cast<char**>(argv);
  myEnd = myBegin + argc;
}

CmdLineParser::~CmdLineParser()
{
}

char* CmdLineParser::getOption( const std::string & theOption ) const
{
  char ** itr = std::find( myBegin, myEnd, theOption );
  if ( itr != myEnd && ++itr != myEnd )
  {
    return *itr;
  }
  return 0;
}

bool CmdLineParser::hasOption( const std::string& theOption ) const
{
  return std::find( myBegin, myEnd, theOption ) != myEnd;
}
