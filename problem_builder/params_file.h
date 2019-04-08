// Copyright (C) 2017 EDF R&D

#ifndef PARAMS_FILE_H_
#define PARAMS_FILE_H_

#include "ini/cpp/INIReader.h"

#define GENERATION_TIME_STR "// Generation time:    "
#define GPUSPH_PREFIX "GPUSPH_"
#define GPUSPH_SUFFIX "__"

//! Parameters values file writer.
class ParamsFile
{
public:
  ParamsFile( const INIReader& theConfig );
  virtual ~ParamsFile();

  int write( const char* theParamsFile );

private:
  const INIReader* myConfig;
};

#endif /* PARAMS_FILE_H_ */
