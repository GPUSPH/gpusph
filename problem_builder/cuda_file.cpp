#include "cuda_file.h"
#include "params_file.h"

CudaFile::CudaFile( const INIReader& theConfig, GPUSPHOptions theOptions,
    const std::string& theTemplatesDir )
    : myConfig( &theConfig ), mySetUp( theOptions ),
        myTemplatesDir( theTemplatesDir )
{
}

CudaFile::~CudaFile()
{
}

int CudaFile::write( const std::string& genericCudaFile,
    const std::string& theDestinationDir )
{

	// Set the new cuda file name
	// ---------------------------------------------------------------
	std::string CudaFname = theDestinationDir;
	if ( !CudaFname.empty() && ( *( CudaFname.end() - 1 ) ) != '/' )
	{
	  CudaFname += "/";
	}
	CudaFname += USER_DIR;
	CudaFname += "/" + mySetUp.ProblemName + ".cu";
	std::ofstream new_cuda_file(CudaFname.c_str());

  if ( !new_cuda_file.is_open() )
  {
    std::cerr << "Failed to open file : \"" << CudaFname << "\"\n"
        << strerror( SYSERROR() ) << std::endl;
      return -1;
  }


	// Copy genericCudaFile into modifiedCudaFile
	// ---------------------------------------------------------------
	std::string myCudaFile = genericCudaFile;

	// Replace all the necessary place holders in the generic file
	// ---------------------------------------------------------------
	// Problem name
	replaceAll(myCudaFile, std::string("GenericProblem"), mySetUp.ProblemName);

	// Replace user functions placeholcer by user functions.
	replaceUserFunctions( myCudaFile );

	// Write the new cuda file
	// ---------------------------------------------------------------
	new_cuda_file << myCudaFile;

	// Close it
	new_cuda_file.close();
	return 0;
}

int CudaFile::replaceUserFunctions( std::string& theModifiedStr ) const
{
  std::string aStr;
  std::list< std::string > aFunctions = getUserFunctions();
  std::list< std::string >::iterator aFunc = aFunctions.begin();
  replaceAll( theModifiedStr, std::string( "#define " ) + GPUSPH_USER_FUNCTIONS,
      "" );
  // Find indent
  std::string anIndent = getIndent( theModifiedStr, GPUSPH_USER_FUNCTIONS );

  for ( ; aFunc != aFunctions.end(); aFunc++ )
  {
    // Find <TemplatesDir>/user_functions/<Func>.cpp file
    std::string aFilePath = std::string( UF_SUBDIR ) + "/" + ( *aFunc )
        + ".cpp";
    std::string aFuncFile = myTemplatesDir + "/" + aFilePath;
    std::string aContent = getFileContent( aFuncFile.c_str() ) + "\n";
    if ( aContent.size() > 0 )
    {
      if ( aStr.size() > 0 )
        aStr += anIndent;
      aStr += UF_BEGIN_COMMENT + aFilePath + "\n";
      aStr += anIndent + getFileContent( aFuncFile.c_str() ) + "\n";
      aStr += anIndent + UF_END_COMMENT + aFilePath + "\n";
    }
  }
  replaceAll( theModifiedStr, GPUSPH_USER_FUNCTIONS, aStr );
  return 0;
}

int CudaFile::writeHeaderFile ( const std::string& theGenericHeader,
    const std::string& theDestinationDir )
{
  // Set the new header file name
  std::string HeaderFname =  theDestinationDir;
  if ( !HeaderFname.empty() && ( *( HeaderFname.end() - 1 ) ) != '/' )
  {
    HeaderFname += "/";
  }
  HeaderFname += USER_DIR;
  HeaderFname += "/" + mySetUp.ProblemName + ".h";
  std::ofstream new_header_file(HeaderFname.c_str());

  if ( !new_header_file.is_open() )
  {
    std::cerr << "Failed to open file : \"" << HeaderFname << "\"\n"
        << strerror( SYSERROR() ) << std::endl;
      return -1;
  }

  // Copy genericHeader into modifiedHeaderFile and replace
  // all necessary items in the latter

  // Change the problem name
  std::string modifiedHeaderFile = theGenericHeader;

  replaceAll( modifiedHeaderFile,
      std::string( "#define " ) + GPUSPH_INCLUDE_PARAMS, "" );
  replaceAll( modifiedHeaderFile, GPUSPH_INCLUDE_PARAMS,
      "#include \"params/" + mySetUp.ProblemName + "_Params.h\"" );

  // Write user functions declarations.
  replaceUserFunctionsHeader(modifiedHeaderFile);

  // Write the new header file
  new_header_file << modifiedHeaderFile;

  // Close it
  new_header_file.close();
  return 0;
}

std::list< std::string > CudaFile::getUserFunctions() const
{
  std::list< std::string > userFunctionsList;
  IniSections sectionList = myConfig->GetSections();
  IniSections_Iter section = sectionList.begin();
  for ( ; section != sectionList.end(); section++ )
  {
    // Find user_functions section.
    if ( section->first == "user_functions" )
    {
      IniSection_Iter aPar = section->second.begin();
      for ( ; aPar != section->second.end(); aPar++ )
      {
        if ( aPar->second == "enable" )
        {
					std::cout << aPar->first << std::endl;
          userFunctionsList.push_back( aPar->first );
        }
      }
    }
    if ( section->first == "domain_splitting" )
		{
			std::string device_map_function = "fillDeviceMap";
			std::cout << device_map_function << std::endl;
			userFunctionsList.push_back( device_map_function );
		}
  }
  IndexedSections sequenceList = myConfig->GetIndexedSections();
  IndexedSections_Iter sequence = sequenceList.begin();
  for ( ; sequence != sequenceList.end(); sequence++ )
	{
		if ( sequence->first == "special_boundary" )
		{
			IndexedSection_Iter section = sequence->second.begin();
			for ( ; section != sequence->second.end(); section++ )
			{

				IniSection_Iter aPar = section->second.begin();
				for ( ; aPar != section->second.end(); aPar++ )
				{
					if ( aPar->first == "open_bnd_type" )
					{
						std::string max_parts_function = "max_parts";
						userFunctionsList.push_back( max_parts_function );
						break;
					}
				}
				break;
			}
		}
	}
  return userFunctionsList;
}

int CudaFile::replaceUserFunctionsHeader(
    std::string& theModifiedStr ) const
{
  std::string aDecl;
  std::list< std::string > aFunctions = getUserFunctions();
  std::list< std::string >::iterator aFunc = aFunctions.begin();
  replaceAll( theModifiedStr, std::string( "#define " ) + GPUSPH_USER_FUNCTIONS,
      "" );
  // Find indent
  std::string anIndent = "\n" + getIndent( theModifiedStr, GPUSPH_USER_FUNCTIONS );

  for ( ; aFunc != aFunctions.end(); aFunc++ )
  {
    // Find <TemplatesDir>/user_functions/<Func>.h file
    std::string aFilePath = std::string( UF_SUBDIR ) + "/" + ( *aFunc )
        + ".h";
    std::string aFuncFile = myTemplatesDir + "/" + aFilePath;
    std::string aContent = getFileContent( aFuncFile.c_str() ) + "\n";
    if ( aContent.size() > 0 )
    {
      if ( aDecl.size() > 0 )
        aDecl += anIndent;
      aDecl += UF_BEGIN_COMMENT + aFilePath;
      aDecl += anIndent + aContent;
      aDecl += anIndent + UF_END_COMMENT + aFilePath;
    }
  }
  replaceAll( theModifiedStr, GPUSPH_USER_FUNCTIONS, aDecl );
  return 0;
}
