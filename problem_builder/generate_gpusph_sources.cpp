#include "gpusph_options.h"
#include "cmd_line_parser.h"
#include "cuda_file.h"
#include "params_file.h"
#include "utils/return.h"
#include "utils/strings_utils.h"

#include <sys/stat.h>

//! Permissions for directories to be created.
#define DIR_PERMISSIONS S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH

using namespace std;

int main( int argc, char**argv )
{

  cout << endl;
  cout << "\t**************************************" << endl;
  cout << "\t*                                    *" << endl;
  cout << "\t*       GPUSPH CODE GENERATOR        *" << endl;
  cout << "\t*                                    *" << endl;
  cout << "\t**************************************" << endl;
  cout << "\t* Version     : 1.0                  *" << endl;
  cout << "\t* Date        : 23.10.2017           *" << endl;
  cout << "\t* Author      : Agnès Leroy          *" << endl;
  cout << "\t* Contributors: Roman Kozlov         *" << endl;
  cout << "\t**************************************" << endl;
  cout << endl;

  // Check that the program was correctly called
  if ( argc == 1 )
  {
    cout << "No configuration file specified." << endl;
    cout
        << "Correct use: build_problem filename [-t templates_path] [-d destination_path]"
        << endl;
    cout << "Example use: build_problem box.ini" << endl;
    return NO_FILE;
  }
  else if ( argc > 3 )
  {
    cout << "Ignoring additional arguments." << endl;
  }

  string ConfigFname = argv[ 1 ];
  INIReader config( ConfigFname );
  // Return an error if the configuration file cannot be read
  if ( config.ParseError() < 0 )
  {
    std::cout << "Can't load configuration file " << ConfigFname << endl;
    ;
    return CANT_READ_CONFIG;
  }

  // Build the simulation setup
  GPUSPHOptions SetUp;

  // Get the options from the configuration file
  SetUp.getGeneralOptions( config );

  // Parse command line.
  CmdLineParser anArgs( argc, argv );
  // Get templates directory option.
  const char* aTemplatesDir = "";
  if ( anArgs.hasOption( "-t" ) )
  {
    aTemplatesDir = anArgs.getOption( "-t" );
  }
  cout << "Templates directory: " << aTemplatesDir << endl;

  // Get destination directory option.
  const char* aDestDir = "";
  if ( anArgs.hasOption( "-d" ) )
  {
    aDestDir = anArgs.getOption( "-d" );
    if ( mkdir( aDestDir, DIR_PERMISSIONS ) && SYSERROR() != EEXIST )
    {
      cout << "Can't create the destination directory: " << aDestDir << endl;
      cout << "mkdir: " << strerror( SYSERROR() ) << endl;
      return CANT_CREATE_DEST_DIR;
    }
  }

  // Create <dest_dir>/src/problems/user/params directory if not exists
  std::string aUserDir = aDestDir;
  if ( !aUserDir.empty() && ( *( aUserDir.end() - 1 ) ) != '/' )
  {
    aUserDir += "/";
  }
  aUserDir += USER_DIR;
  aUserDir += "/params";
  if ( system( ( "mkdir -p " + aUserDir ).c_str() ) && SYSERROR() != EEXIST )
  {
    cout << "Can't create the destination user directory: " << aUserDir << endl;
    cout << "mkdir: " << strerror( SYSERROR() ) << endl;
    return CANT_CREATE_DEST_DIR;
  }

  cout << "Destination directory: " << aDestDir << endl;
  cout << "User problems directory: " << aUserDir << endl;

  // Generate the params.h file
  string aParamsFname( aUserDir + "/" );
  aParamsFname += SetUp.ProblemName + "_Params.h";
  ParamsFile aParams( config );
  if ( aParams.write( aParamsFname.c_str() ) )
  {
    cout << "Can't create the parameters file " << aParamsFname << endl;
    cout << strerror( SYSERROR() ) << endl;
    return CANT_CREATE_HEADER;
  }
  cout << "New parameters file created: " << aParamsFname << endl;

  CudaFile aCudaFile( config, SetUp, aTemplatesDir );
  // Generate the .h file
  string headerFile = getFileContent( "src/problems/GenericProblem.h" );
  if ( !headerFile.empty() )
  {
    if ( aCudaFile.writeHeaderFile( headerFile, aDestDir ) )
    {
      cout << "Can't create the new header file src/problems/user/"
          << SetUp.ProblemName << ".h" << endl;
      return CANT_CREATE_HEADER;
    }
    cout << "New header file created: src/problems/user/" << SetUp.ProblemName
        << ".h" << endl;
  }
  else
  {
    cout << "Generic header file is empty... check the path" << endl;
    exit( 1 );
  }
  // Generate the .cu file
  string cudaFile = getFileContent( "src/problems/GenericProblem.cu" );
  if ( !cudaFile.empty() )
  {
    if ( aCudaFile.write( cudaFile, aDestDir ) )
    {
      cout << "Can't create the new cuda file created in src/problems/user/"
          << SetUp.ProblemName << ".cu" << endl;
      return CANT_CREATE_CPP;
    }
    cout << "New cuda file created in src/problems/user/" << SetUp.ProblemName
        << ".cu" << endl;
  }
  else
  {
    cout << "Generic cuda file is empty... check the path" << endl;
    exit( 1 );
  }
  return 0;
}

