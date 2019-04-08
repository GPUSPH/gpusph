#include "gtest/gtest.h"
#include "cmd_line_parser.h"
#include "utils/strings_utils.h"
#include "ini/cpp/INIReader.h"
#include "params_file.h"
#include "cuda_file.h"

using namespace ::testing;

TEST(TestProblemBuilder, CmdLineParser)
{
  const char* anArgs[] = { "arg0", "-key1", "val1", "-key2", "val2" };
  const int anArgc = 5;

  CmdLineParser anOpts( anArgc, &anArgs[ 0 ] );
  EXPECT_FALSE( anOpts.hasOption( "" ) );
  EXPECT_TRUE( anOpts.hasOption( "-key1" ) );
  EXPECT_TRUE( anOpts.hasOption( "-key2" ) );
  EXPECT_FALSE( anOpts.hasOption( "key1" ) );
  EXPECT_FALSE( anOpts.hasOption( "-key" ) );
  EXPECT_FALSE( anOpts.hasOption( "-key3" ) );
  EXPECT_STREQ( NULL, anOpts.getOption( "" ) );
  EXPECT_STREQ( "val1", anOpts.getOption( "-key1" ) );
  EXPECT_STREQ( "val2", anOpts.getOption( "-key2" ) );
  EXPECT_STREQ( NULL, anOpts.getOption( "key1" ) );
  EXPECT_STREQ( NULL, anOpts.getOption( "-key" ) );
  EXPECT_STREQ( NULL, anOpts.getOption( "-key3" ) );
}

TEST(TestProblemBuilder, IsIndexedSection)
{
  std::string aBase;
  int anIndex;

  EXPECT_TRUE( isIndexedSection( "section_231", aBase, anIndex ) );
  EXPECT_STREQ( "section", aBase.c_str() );
  EXPECT_EQ( 231, anIndex );

  EXPECT_TRUE( isIndexedSection( "sect_ion_1", aBase, anIndex ) );
  EXPECT_STREQ( "sect_ion", aBase.c_str() );
  EXPECT_EQ( 1, anIndex );

  EXPECT_TRUE( isIndexedSection( "sect_2ion_1", aBase, anIndex ) );
  EXPECT_STREQ( "sect_2ion", aBase.c_str() );
  EXPECT_EQ( 1, anIndex );

  EXPECT_FALSE( isIndexedSection( "sect_ion_s1", aBase, anIndex ) );
  EXPECT_STREQ( "sect_ion_s1", aBase.c_str() );
  EXPECT_EQ( -1, anIndex );

  EXPECT_FALSE( isIndexedSection( "_", aBase, anIndex ) );
  EXPECT_STREQ( "_", aBase.c_str() );
  EXPECT_EQ( -1, anIndex );

  EXPECT_FALSE( isIndexedSection( "s_", aBase, anIndex ) );
  EXPECT_STREQ( "s_", aBase.c_str() );
  EXPECT_EQ( -1, anIndex );

  EXPECT_FALSE( isIndexedSection( "sec1", aBase, anIndex ) );
  EXPECT_STREQ( "sec1", aBase.c_str() );
  EXPECT_EQ( -1, anIndex );

  EXPECT_FALSE( isIndexedSection( "s1", aBase, anIndex ) );
  EXPECT_STREQ( "s1", aBase.c_str() );
  EXPECT_EQ( -1, anIndex );

  EXPECT_FALSE( isIndexedSection( "", aBase, anIndex ) );
  EXPECT_STREQ( "", aBase.c_str() );
  EXPECT_EQ( -1, anIndex );
}

TEST(TestProblemBuilder, IniReader)
{
  // 4 simple sections: section1, section2, a_b, user_functions
  // 2 indexed sections: fluid(fluid_1,fluid_2,fluid_3) and gage(gage_1,gage_2)
  const char* aFname = "problem_builder/test/data/test1.ini";
  INIReader aReader( aFname );
  // Return an error if the configuration file cannot be read
  EXPECT_EQ( 0, aReader.ParseError() ) << "Can't read ini file: "
      << strerror( SYSERROR() );
  EXPECT_EQ( 4, aReader.GetSections().size() );
  EXPECT_EQ( 2, aReader.GetIndexedSections().size() );

  EXPECT_EQ( 2, aReader.GetSections()[ "section1" ].size() );
  EXPECT_EQ( 2, aReader.GetSections()[ "section2" ].size() );
  EXPECT_EQ( 2, aReader.GetSections()[ "a_b" ].size() );

  EXPECT_EQ( 3, aReader.GetIndexedSections()[ "fluid" ].size() );
  EXPECT_EQ( 2, aReader.GetIndexedSections()[ "gage" ].size() );

  EXPECT_EQ( 2, aReader.GetIndexedSections()[ "fluid" ][ 0 ].size() );
  EXPECT_EQ( 2, aReader.GetIndexedSections()[ "fluid" ][ 1 ].size() );
  EXPECT_EQ( 2, aReader.GetIndexedSections()[ "fluid" ][ 2 ].size() );

  EXPECT_EQ( 2, aReader.GetIndexedSections()[ "gage" ][ 0 ].size() );
  EXPECT_EQ( 2, aReader.GetIndexedSections()[ "gage" ][ 1 ].size() );

  EXPECT_STREQ( "val11",
      aReader.GetSections()[ "section1" ][ "par11" ].c_str() );
  EXPECT_STREQ( "val12",
      aReader.GetSections()[ "section1" ][ "par12" ].c_str() );

  EXPECT_STREQ( "val21",
      aReader.GetSections()[ "section2" ][ "par21" ].c_str() );
  EXPECT_STREQ( "val22",
      aReader.GetSections()[ "section2" ][ "par22" ].c_str() );

  EXPECT_STREQ( "b", aReader.GetSections()[ "a_b" ][ "a" ].c_str() );
  EXPECT_STREQ( "0.3/f:", aReader.GetSections()[ "a_b" ][ "b" ].c_str() );

  EXPECT_STREQ( "fval11",
      aReader.GetIndexedSections()[ "fluid" ][ 0 ][ "fpar1" ].c_str() );
  EXPECT_STREQ( "fval12",
      aReader.GetIndexedSections()[ "fluid" ][ 0 ][ "fpar2" ].c_str() );

  EXPECT_STREQ( "fval21",
      aReader.GetIndexedSections()[ "fluid" ][ 1 ][ "fpar1" ].c_str() );
  EXPECT_STREQ( "fval22",
      aReader.GetIndexedSections()[ "fluid" ][ 1 ][ "fpar2" ].c_str() );
}

std::string removeHeader( const std::string& theStr )
{
  std::string aRes = theStr;
  size_t aPos = aRes.find( GENERATION_TIME_STR );
  if ( aPos != std::string::npos )
  {
    aRes = aRes.substr( aPos );
    aPos = aRes.find( "\n" );
    if ( aPos != std::string::npos )
    {
      aRes = aRes.substr( aPos );
    }
  }
  return aRes;
}

TEST(TestProblemBuilder, WriteParams)
{
  const char* anOutFile = "problem_builder/test/data/params.txt";
  const char* aSampleFile = "problem_builder/test/data/sample_params.txt";
  // 3 simple sections: section1, section2, a_b
  // 2 indexed sections: fluid(fluid_1,fluid_2,fluid_3) and gage(gage_1,gage_2)
  const char* aFname = "problem_builder/test/data/test1.ini";
  INIReader aReader( aFname );
  EXPECT_EQ( 0, aReader.ParseError() ) << strerror( SYSERROR() );

  ParamsFile aParamsFile( aReader );


  EXPECT_EQ( 0, aParamsFile.write( anOutFile ) ) << "Can't write params file: "
      << strerror( SYSERROR() );

  std::string aRes = getFileContent( anOutFile );
  aRes = removeHeader( aRes );
  std::string aSample = getFileContent( aSampleFile );
  aSample = removeHeader( aSample );

  EXPECT_STREQ( aSample.c_str(), aRes.c_str() );
}

TEST(TestProblemBuilder, GetUserFunctions)
{
  // func1=enable, check_dt=disable, g_callback=enable
  const char* aFname = "problem_builder/test/data/test1.ini";
  INIReader aReader( aFname );
  // Return an error if the configuration file cannot be read
  EXPECT_EQ( 0, aReader.ParseError() ) << "Can't read ini file: "
      << strerror( SYSERROR() );
  CudaFile aCudaFile( aReader, GPUSPHOptions() );
  std::list< std::string > aFunctions = aCudaFile.getUserFunctions();
  EXPECT_EQ( 2, aFunctions.size() );
  if ( aFunctions.size() == 2 )
  {
    EXPECT_STREQ( "func1", aFunctions.front().c_str() );
    EXPECT_STREQ( "g_callback", aFunctions.back().c_str() );
  }
}

TEST(TestProblemBuilder, ReplaceFunctionsHeader)
{
  // func1=enable, check_dt=disable, g_callback=enable
  const char* aGoodDecl =
      UF_BEGIN_COMMENT"user_functions/func1.h\n"
      "virtual void func1();\n\n\n"
      UF_END_COMMENT"user_functions/func1.h\n"
      UF_BEGIN_COMMENT"user_functions/g_callback.h\n"
      "virtual void g_callback();\n\n\n"
      UF_END_COMMENT"user_functions/g_callback.h";
  const char* aFname = "problem_builder/test/data/test1.ini";
  const char* aTemplDir = "problem_builder/test/data";
  INIReader aReader( aFname );
  // Return an error if the configuration file cannot be read
  EXPECT_EQ( 0, aReader.ParseError() ) << "Can't read ini file: "
      << strerror( SYSERROR() );
  CudaFile aCudaFile( aReader, GPUSPHOptions(), aTemplDir );
  std::string aHeader( GPUSPH_USER_FUNCTIONS );
  EXPECT_EQ( 0, aCudaFile.replaceUserFunctionsHeader( aHeader ) );
  EXPECT_STREQ( aGoodDecl, aHeader.c_str() );
}

TEST(TestProblemBuilder, ReplaceFunctionsHeaderWithIndent)
{
#define SOME_PREFIX "Some prefix\n"
  // func1=enable, check_dt=disable, g_callback=enable
  const char* aGoodDecl = SOME_PREFIX
      "\t\t" UF_BEGIN_COMMENT "user_functions/func1.h\n"
      "\t\tvirtual void func1();\n\n\n"
      "\t\t" UF_END_COMMENT "user_functions/func1.h\n"
      "\t\t" UF_BEGIN_COMMENT "user_functions/g_callback.h\n"
      "\t\tvirtual void g_callback();\n\n\n"
      "\t\t" UF_END_COMMENT "user_functions/g_callback.h";
  const char* aFname = "problem_builder/test/data/test1.ini";
  const char* aTemplDir = "problem_builder/test/data";
  INIReader aReader( aFname );
  // Return an error if the configuration file cannot be read
  EXPECT_EQ( 0, aReader.ParseError() ) << "Can't read ini file: "
      << strerror( SYSERROR() );
  CudaFile aCudaFile( aReader, GPUSPHOptions(), aTemplDir );
  std::string aHeader( GPUSPH_USER_FUNCTIONS );
  aHeader = SOME_PREFIX"\t\t" + aHeader;
  EXPECT_EQ( 0, aCudaFile.replaceUserFunctionsHeader( aHeader ) );
  EXPECT_STREQ( aGoodDecl, aHeader.c_str() );
}

TEST(TestProblemBuilder, ReplaceFunctions)
{
  // func1=enable, check_dt=disable, g_callback=enable
  const char* aGoodRes = SOME_PREFIX
      "\t\t" UF_BEGIN_COMMENT "user_functions/func1.cpp\n"
      "\t\tvoid func1(){}\n\n"
      "\t\t" UF_END_COMMENT "user_functions/func1.cpp\n"
      "\t\t" UF_BEGIN_COMMENT "user_functions/g_callback.cpp\n"
      "\t\tvoid g_callback(){}\n\n"
      "\t\t" UF_END_COMMENT "user_functions/g_callback.cpp\n";
  const char* aFname = "problem_builder/test/data/test1.ini";
  const char* aTemplDir = "problem_builder/test/data";
  INIReader aReader( aFname );
  // Return an error if the configuration file cannot be read
  EXPECT_EQ( 0, aReader.ParseError() ) << "Can't read ini file: "
      << strerror( SYSERROR() );
  CudaFile aCudaFile( aReader, GPUSPHOptions(), aTemplDir );
  std::string aSrc( GPUSPH_USER_FUNCTIONS );
  aSrc = SOME_PREFIX "\t\t" + aSrc;
  EXPECT_EQ( 0, aCudaFile.replaceUserFunctions( aSrc ) );
  EXPECT_STREQ( aGoodRes, aSrc.c_str() );
}

/*
TEST(TestProblemBuilder,  ReplaceSimulationFramework)
{
  // func1=enable, check_dt=disable, g_callback=enable
  const char* aGoodRes = "  SETUP_FRAMEWORK(\n"
      "    kernel<WENDLAND>,\n"
      "    formulation<SPH_F1>,\n"
      "    viscosity<DYNAMICVISC>,\n"
      "    boundary<SA_BOUNDARY>,\n"
      "    periodicity<PERIODICITY>,\n"
      "    flags<ENABLE_DENSITY_SUM |\n\t\t\tENABLE_WATER_DEPTH>\n"
      "  )";
  std::string aSrc = "  SETUP_FRAMEWORK(\n"
      "    kernel<KERNEL_TYPE>,\n"
      "    formulation<SPH_FORMULATION>,\n"
      "    viscosity<VISCOSITY_TYPE>,\n"
      "    boundary<BOUNDARY_TYPE>,\n"
      "    periodicity<PERIODICITY>,\n"
      "    flags<FLAGS_LIST>\n"
      "  )";

  const char* aFname = "problem_builder/test/data/test1.ini";
  const char* aTemplDir = "problem_builder/test/data";
  INIReader aReader( aFname );
  // Return an error if the configuration file cannot be read
  EXPECT_EQ( 0, aReader.ParseError() ) << "Can't read ini file: "
      << strerror( SYSERROR() );

  // Build the simulation setup
  GPUSPHOptions SetUp;

  // Get the options from the configuration file
  SetUp.getGeneralOptions( aReader );
  SetUp.getSPHOptions( aReader );
//  SetUp.getPhysicalOptions( aReader );

  CudaFile aCudaFile( aReader, SetUp, aTemplDir );

  aCudaFile.replaceSimulationFramework( aSrc );
  EXPECT_STREQ( aGoodRes, aSrc.c_str() );
}
*/

#define GPUSPH_n_m__ 5
#define GPUSPH_x_y__ teststr
#define GPUSPH_geometry_fluid_file__ aFile
#define PQUOTE(x) #x
#define PVAL(s,p) GPUSPH_##s##_##p##__
#define _PSTR(theValue) PQUOTE(theValue)
#define PSTR(s,p) _PSTR(PVAL(s,p))
#define ISDEFINED(s,p) strcmp(PSTR(s,p),"GPUSPH_"#s"_"#p"__")
#define PVALS(s,p) GPUSPH_##s##_##p##_VALS__
#define PSTRVALS(s,p) GPUSPH_##s##_##p##_STRVALS__

#define GPUSPH_special_boundary_collisions_file_VALS__ NAN,aCollisionsFileName
#define GPUSPH_special_boundary_collisions_file_STRVALS__ NULL,"aCollisionsFileName"
#define GPUSPH_special_boundary_density_VALS__ NAN,5

enum SplitAxis
{
  LONGEST_AXIS,
  X_AXIS,
  Y_AXIS,
  Z_AXIS
};
/* Periodic boundary */
enum Periodicity {
  PERIODIC_NONE = 0,
  PERIODIC_X   = 1,
  PERIODIC_Y   = PERIODIC_X << 1,
  PERIODIC_XY  = PERIODIC_X | PERIODIC_Y,
  PERIODIC_Z   = PERIODIC_Y << 1,
  PERIODIC_XZ  = PERIODIC_X | PERIODIC_Z,
  PERIODIC_YZ  = PERIODIC_Y | PERIODIC_Z,
  PERIODIC_XYZ = PERIODIC_X | PERIODIC_Y | PERIODIC_Z,
};

#define x_AXIS X_AXIS
#define y_AXIS Y_AXIS
#define z_AXIS Z_AXIS
#define GPUSPH_domain_splitting_split_axis__ x
#define GPUSPH_domain_splitting_split_axis_x

#define ISENUM_EQ(s,p,v) defined( GPUSPH_##s##_##p##_##v )
#if (ISENUM_EQ( domain_splitting, split_axis, x ))
int bb;
#else
int bb;
#endif

#define __SPLIT_AXIS(a) a##_AXIS
#define _SPLIT_AXIS(a) __SPLIT_AXIS(a)
#define SPLIT_AXIS _SPLIT_AXIS( PVAL( domain_splitting, split_axis ) )

#define GPUSPH_periodicity_periodicity_x__ false
#define GPUSPH_periodicity_periodicity_y__ true
#define GPUSPH_periodicity_periodicity_z__ false

#define periodicity_x_true PERIODIC_X
#define periodicity_x_false PERIODIC_NONE
#define periodicity_y_true PERIODIC_Y
#define periodicity_y_false PERIODIC_NONE
#define periodicity_z_true PERIODIC_Z
#define periodicity_z_false PERIODIC_NONE
#define __PERIODIC(x,v) periodicity##_##x##_##v
#define _PERIODIC(x,v) __PERIODIC(x,v)
#define PERIODIC(x) _PERIODIC(x, PVAL(periodicity, periodicity_##x ))
#define PERIODICITY (Periodicity)(PERIODIC(x) | PERIODIC(y) | PERIODIC(z))

#define TENABLE_INLET_OUTLET   1
#define TENABLE_WATER_DEPTH    (TENABLE_INLET_OUTLET << 1)
#define TFLAG_INLET_OUTLET 0
#define TFLAG_DENSITY_SUM 0

#define TFLAG_LIST TFLAG_INLET_OUTLET | TFLAG_DENSITY_SUM | TENABLE_WATER_DEPTH

#define FLAG_1 TENABLE_INLET_OUTLET
#define FLAG_2 0

#define FLAG_LIST1 FLAG_1 | FLAG_2

#define IS_DEF(x,y) defined(x##_##y)

#if IS_DEF(FLAG,1)
int aa;
#else
int aa;
#endif

TEST(TestProblemBuilder, Macros)
{
  EXPECT_EQ( TENABLE_WATER_DEPTH, TFLAG_LIST );
  EXPECT_NE( TENABLE_INLET_OUTLET | TENABLE_WATER_DEPTH, TFLAG_LIST );
  EXPECT_EQ( TENABLE_INLET_OUTLET, FLAG_LIST1 );

  const char* aTstStr = PSTR(x,y);
  const char* aFluidFile = PSTR( geometry, fluid_file );
  EXPECT_STREQ( "teststr", aTstStr );
  EXPECT_STREQ( "aFile", aFluidFile );
  EXPECT_TRUE( ISDEFINED(x,y) );
  EXPECT_TRUE( ISDEFINED( geometry, fluid_file ) );
  EXPECT_FALSE( ISDEFINED(z,y) );

  const char* aStr = NULL;
  if ( ISDEFINED(z,y) )
  {
    aStr = PSTR(z,y);
  }

  EXPECT_EQ( NULL, aStr );

  aStr = NULL;
  if ( ISDEFINED( geometry, fluid_file ) )
  {
    aStr = PSTR( geometry, fluid_file );
  }

  EXPECT_EQ( "aFile", aStr );

  int n = 0;
  if ( ISDEFINED( n, m ) )
  {
    n = PVAL( n, m );
  }

  EXPECT_EQ( 5, n );

  const char* aCollisionsFiles[] =
      { PSTRVALS( special_boundary, collisions_file ) };
  double aDensity[] = { PVALS( special_boundary, density ) };
  for ( int i = 0; i < 2; i++ )
  {
    const char* aCollisionsFile = NULL;
    if( aCollisionsFiles[ i ] )
    {
      aCollisionsFile = aCollisionsFiles[ i ];
    }
    switch(i)
    {
    case 0:
      {
        EXPECT_EQ( NULL, aCollisionsFile );
        EXPECT_FALSE( aDensity[ i ] > 0 );
        break;
      }
    case 1:
      {
        EXPECT_STREQ( "aCollisionsFileName", aCollisionsFile );
        EXPECT_TRUE( aDensity[ i ] > 0 );
        EXPECT_EQ( 5, aDensity[ i ] );
        break;
      }
    }
  }

  EXPECT_EQ( X_AXIS, SPLIT_AXIS );
  EXPECT_NE( Y_AXIS, SPLIT_AXIS );

  EXPECT_EQ( PERIODIC_Y, PERIODICITY );
  EXPECT_NE( PERIODIC_X, PERIODICITY );
  EXPECT_NE( PERIODIC_Z, PERIODICITY );
}
