#ifndef _GENERATE_CUDA_FILE_H_
#define _GENERATE_CUDA_FILE_H_

#include "gpusph_options.h"
#include "utils/strings_utils.h"
#include "utils/vector_math.h"
#include <string>
#include <list>

//! Place holder for a directive to include a generated parameters file.
#define GPUSPH_INCLUDE_PARAMS "GPUSPH_INCLUDE_PARAMS"
//! Placeholder for user functions insertion.
#define GPUSPH_USER_FUNCTIONS "GPUSPH_USER_FUNCTIONS"
//! User functions section name.
#define UF_SUBDIR "user_functions"
//! The inserted user function beginning mark prefix.
#define UF_BEGIN_COMMENT "// BEGIN USER FUNCTION FILE: "
//! The inserted user function ending mark prefix.
#define UF_END_COMMENT "// END USER FUNCTION FILE: "

//! Problem.cu and Problem.h files writer.
class CudaFile
{
public:
  /*!
   * Constructor.
   * @param theConfig the configuration from ini file
   * @param theOptions the specific GPUSPH options
   * @param theTemplatesDir the directory to search templates
   */
  CudaFile( const INIReader& theConfig, GPUSPHOptions theOptions,
      const std::string& theTemplatesDir = "" );
  virtual ~CudaFile();

  /*! Create Problem.cu file from the template.
   * @param theTemplate the .cu file template
   * @param theDestinationDir the destination directory
   * @return 0 if succeeded, otherwise error code
   */
  int write( const std::string& theTemplate,
      const std::string& theDestinationDir );

  /*! Create Problem.h file from the template.
   * @param theTemplate the .h file template
   * @param theDestinationDir the destination directory
   * @return 0 if succeeded, otherwise error code
   */
  int writeHeaderFile ( const std::string& theGenericHeader,
      const std::string& theDestinationDir );

  /*!
   * Read user functions from .cpp files and insert into the template code.
   * @param theModifiedStr the template
   * @return 0 if succeeded, otherwise error code
   */
  int replaceUserFunctions( std::string& theModifiedStr ) const;
  /*!
   * Read user functions declarations from .h files and insert into the template
   * code.
   * @param theModifiedStr the template
   * @return 0 if succeeded, otherwise error code
   */
  int replaceUserFunctionsHeader( std::string& theModifiedStr ) const;
  /*!
   * Get the list of user functions names from user_functions section from ini.
   * @return the list of functions names
   */
  std::list< std::string > getUserFunctions() const;

private:
  const INIReader* myConfig; //!< Generic ini file content
  GPUSPHOptions mySetUp; //!< Specific GPUSPH options from the ini file
  const std::string myTemplatesDir; //!< The path to the templates folder
  std::string myCudaFile; //!< The result cuda file content
};
#endif
