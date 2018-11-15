# ===================================================== #
#                                                       #
#    #    ##   #  #   #                           #     #
#   # #  #  #  #  #  # #                          #     #
#  ##### #  #  #  # #####  ##  ###  #  #  ## ###  ###   #
#  #   # #  #  #  # #   # #  # #  # #  # #   #  # #  #  #
#  #   # #  #  #  # #   # #  # #  # #  #   # #  # #  #  #
#  #   #  ## #  ##  #   #  ### ###   ### ##  ###  #  #  #
#                            # #             #          #
#                          ##  #             #          #
#                                                       #
# ===================================================== #
#                                                       #
#   Authors:                                            #
#       - Jose Luis Cercos-Pita                         #
#       - Leo Miguel Gonzalez                           #
#       - Antonio Souto-Iglesias                        #
#                                                       #
# ===================================================== #


# ===================================================== #
# General CMake options                                 #
# ===================================================== #
cmake_minimum_required(VERSION 3.9.0 FATAL_ERROR)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cMake")

# Set the version from git repo.
# See cMake/MunkeiVersionFromGit.cmake, which is a slightly modified version of
# https://github.com/Munkei/munkei-cmake/blob/master/doc/MunkeiVersionFromGit.md
include( MunkeiVersionFromGit )
version_from_git(
  LOG          OFF
  INCLUDE_HASH OFF
)

project(GPUSPH
  VERSION ${VERSION}
  LANGUAGES C;CXX;CUDA
)
add_definitions(-DGPUSPH_VERSION=\"${VERSION}\" -DUSE_CMAKE)

# -std=c++11
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CUDA_STANDARD 11)

# No Debug/Release output paths
set(DEBUG_MAIN_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(RELEASE_MAIN_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# Some useful commands comaptibility
set(PLATFORM_CP cp)
set(PLATFORM_MK mkdir -p)
if(APPLE)
    set(PLATFORM_SED sed -i "" -e)
else(APPLE)
    set(PLATFORM_SED sed -i -e)
endif(APPLE)


# ===================================================== #
# Output directories for install target                 #
# ===================================================== #
SET(CMAKE_INSTALL_PREFIX
    "/usr"
    CACHE PATH
    "Installation root directory")
SET(INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
if(APPLE)
    set(BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif(APPLE)

SET(CMAKE_INSTALL_DATADIR
    share/gpusph
    CACHE PATH
    "Output directory for data and resource files")
SET(CMAKE_INSTALL_INCLUDEDIR
    include/gpusph
    CACHE PATH
    "Output directory for header files")
SET(CMAKE_INSTALL_DOCDIR
    share/doc/gpusph
    CACHE PATH
    "Output directory for documentation and license files")
SET(CMAKE_INSTALL_BINDIR
    bin
    CACHE PATH
    "Output directory for binary files")
SET(CMAKE_INSTALL_LIBDIR
    lib
    CACHE PATH
    "Output directory for library files")

# used as compiler defines
SET(RESOURCEDIR "${CMAKE_INSTALL_DATADIR}")
SET(DOCDIR "${CMAKE_INSTALL_DOCDIR}")

# ===================================================== #
# Build options                                         #
# ===================================================== #
IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE "Release")
ENDIF(NOT CMAKE_BUILD_TYPE)
SET(GPUSPH_PROBLEM "DamBreak3D" CACHE STRING "Problem in src/problems folder to be considered")
OPTION(GPUSPH_USE_MPI "MPI support" ON)
OPTION(GPUSPH_USE_HDF5 "HDF5 files support" ON)
OPTION(GPUSPH_USE_CHRONO "CHRONO Multi-physics Simulation Engine support" OFF)
OPTION(GPUSPH_FASTMATH "Apply --use_fast_math CUDA option" ON)
SET(LINEARIZATION_VALUES xyz xzy yxz yzx zxy zyx)
SET(GPUSPH_LINEARIZATION yzx CACHE STRING "Linearization alternative")
SET_PROPERTY(CACHE GPUSPH_LINEARIZATION PROPERTY STRINGS ${LINEARIZATION_VALUES})

# ===================================================== #
# Search the packages                                   #
# ===================================================== #
# AWK
FIND_PACKAGE(AWK REQUIRED)
IF(NOT AWK_FOUND)
    MESSAGE(FATAL_ERROR "AWK not found")
ENDIF(NOT AWK_FOUND)

# CUDA
FIND_PACKAGE(CUDA REQUIRED)
IF(NOT CUDA_FOUND)
    MESSAGE(FATAL_ERROR "CUDA not found")
ENDIF(NOT CUDA_FOUND)

IF(CUDA_VERSION_MAJOR LESS 7)
    MESSAGE(FATAL_ERROR "CUDA is too old (${CUDA_VERSION} < 7.0)")
ENDIF(CUDA_VERSION_MAJOR LESS 7)

# MPI
IF(GPUSPH_USE_MPI)
    FIND_PACKAGE(MPI)
    IF(NOT MPI_CXX_FOUND)
        MESSAGE(FATAL_ERROR "MPI (CXX) not found, but GPUSPH_USE_MPI is ON. Install MPI or set GPUSPH_USE_MPI=OFF")
    ENDIF(NOT MPI_CXX_FOUND)

    ADD_DEFINITIONS(-DUSE_MPI=1)
ELSE(GPUSPH_USE_MPI)
    ADD_DEFINITIONS(-DUSE_MPI=0)
ENDIF(GPUSPH_USE_MPI)

# HDF5
IF(GPUSPH_USE_HDF5)
    FIND_PACKAGE(HDF5)
    IF(NOT HDF5_FOUND)
        MESSAGE(FATAL_ERROR "HDF5 not found, but GPUSPH_USE_HDF5 is ON. Install HDF5 or set GPUSPH_USE_HDF5=OFF")
    ENDIF(NOT HDF5_FOUND)

    ADD_DEFINITIONS(-DUSE_HDF5=1)

    IF(GPUSPH_USE_MPI AND NOT HDF5_IS_PARALLEL)
        MESSAGE(WARNING "MPI support is enabled. However installed HDF5 does not support it.")
    ENDIF(GPUSPH_USE_MPI AND NOT HDF5_IS_PARALLEL)
ELSE(GPUSPH_USE_HDF5)
    ADD_DEFINITIONS(-DUSE_HDF5=0)
ENDIF(GPUSPH_USE_HDF5)

# CHRONO
IF(GPUSPH_USE_CHRONO)
    FIND_PACKAGE(CHRONO)
    IF(NOT CHRONO_FOUND)
        MESSAGE(FATAL_ERROR "CHRONO not found, but GPUSPH_USE_CHRONO is ON. Install CHRONO or set GPUSPH_USE_CHRONO=OFF")
    ENDIF(NOT CHRONO_FOUND)

    ADD_DEFINITIONS(-DUSE_CHRONO=1)
ELSE(GPUSPH_USE_CHRONO)
    ADD_DEFINITIONS(-DUSE_CHRONO=0)
ENDIF(GPUSPH_USE_CHRONO)

# ===================================================== #
# CUDA compute capabilities                             #
# ===================================================== #
EXECUTE_PROCESS( COMMAND ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc -Wno-deprecated-gpu-targets -o ${CMAKE_BINARY_DIR}/list-cuda-cc ${CMAKE_CURRENT_SOURCE_DIR}/scripts/list-cuda-cc.cc
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/scripts/
                 RESULT_VARIABLE CUDA_CC_HAS_ERROR
                 OUTPUT_VARIABLE CUDA_CC
                 ERROR_VARIABLE CUDA_CC_ERROR )
IF(CUDA_CC_HAS_ERROR EQUAL 0)
    EXECUTE_PROCESS( COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/define-cuda-cc.sh  
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/
                     RESULT_VARIABLE CUDA_CC_HAS_ERROR
                     OUTPUT_VARIABLE CUDA_CC
                     ERROR_VARIABLE CUDA_CC_ERROR )
ENDIF(CUDA_CC_HAS_ERROR EQUAL 0)

IF(NOT CUDA_CC_HAS_ERROR EQUAL 0)
    MESSAGE(WARNING "Failed detecting the CUDA compute capabilities:")
    MESSAGE(WARNING "${CUDA_CC_ERROR}")
    SET(CUDA_CC "20")
ENDIF(NOT CUDA_CC_HAS_ERROR EQUAL 0)
STRING(REGEX REPLACE "\n$" "" CUDA_CC "${CUDA_CC}")
SET(GPUSPH_COMPUTE_CAPABILITIES "${CUDA_CC}" CACHE STRING "11, 12, 13, 20, 21, 30, 35, etc. Left empty for automatic detection")
MARK_AS_ADVANCED(GPUSPH_COMPUTE_CAPABILITIES)

ADD_DEFINITIONS(-DCOMPUTE=${GPUSPH_COMPUTE_CAPABILITIES})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__COMPUTE__=${GPUSPH_COMPUTE_CAPABILITIES}")
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D__COMPUTE__=${GPUSPH_COMPUTE_CAPABILITIES} -D__NVCC_VERSION__=${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR} -arch=sm_${GPUSPH_COMPUTE_CAPABILITIES}")

# ===================================================== #
# Other options                                         #
# ===================================================== #
IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
    ADD_DEFINITIONS(-D_DEBUG_)
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info")
ELSE(CMAKE_BUILD_TYPE MATCHES DEBUG)
    # Took from former Makefile. Prolly not required at all
    REMOVE_DEFINITIONS(-D_DEBUG_)
ENDIF(CMAKE_BUILD_TYPE MATCHES DEBUG)

STRING(SUBSTRING ${GPUSPH_LINEARIZATION} 0 1 GPUSPH_LINEARIZATION_WORD1)
STRING(SUBSTRING ${GPUSPH_LINEARIZATION} 1 1 GPUSPH_LINEARIZATION_WORD2)
STRING(SUBSTRING ${GPUSPH_LINEARIZATION} 2 1 GPUSPH_LINEARIZATION_WORD3)
ADD_DEFINITIONS(-DLINEARIZATION=\"${GPUSPH_LINEARIZATION}\" 
                -DCOORD1=${GPUSPH_LINEARIZATION_WORD1} 
                -DCOORD2=${GPUSPH_LINEARIZATION_WORD2} 
                -DCOORD3=${GPUSPH_LINEARIZATION_WORD3} )

IF(GPUSPH_FASTMATH)
    ADD_DEFINITIONS(-DFASTMATH=1)
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
ELSE(GPUSPH_FASTMATH)
    ADD_DEFINITIONS(-DFASTMATH=0)
ENDIF(GPUSPH_FASTMATH)

IF(CMAKE_VERBOSE_MAKEFILE)
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")    
ENDIF(CMAKE_VERBOSE_MAKEFILE)

# ===================================================== #
# problem.h                                             #
# ===================================================== #
CONFIGURE_FILE(problem.h.cmake ${CMAKE_CURRENT_BINARY_DIR}/problem.h)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

# ===================================================== #
# Parse debugflags.def                                  #
# ===================================================== #
EXECUTE_PROCESS( COMMAND ${AWK_EXECUTABLE} -f ${CMAKE_CURRENT_SOURCE_DIR}/scripts/parse-debugflags.awk ${CMAKE_CURRENT_SOURCE_DIR}/src/debugflags.def
                 WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/
                 RESULT_VARIABLE PARSE_DEBUG_FLAGS_HAS_ERROR
                 OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/parse-debugflags.h
                 ERROR_VARIABLE PARSE_DEBUG_FLAGS_ERROR )
IF(NOT PARSE_DEBUG_FLAGS_HAS_ERROR EQUAL 0)
    MESSAGE(FATAL_ERROR "Failure generating parse-debugflags.h\n${PARSE_DEBUG_FLAGS_ERROR}")
ENDIF(NOT PARSE_DEBUG_FLAGS_HAS_ERROR EQUAL 0)
EXECUTE_PROCESS( COMMAND ${AWK_EXECUTABLE} -f ${CMAKE_CURRENT_SOURCE_DIR}/scripts/describe-debugflags.awk ${CMAKE_CURRENT_SOURCE_DIR}/src/debugflags.def
                 WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/
                 RESULT_VARIABLE PARSE_DEBUG_FLAGS_HAS_ERROR
                 OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/describe-debugflags.h
                 ERROR_VARIABLE PARSE_DEBUG_FLAGS_ERROR )
IF(NOT PARSE_DEBUG_FLAGS_HAS_ERROR EQUAL 0)
    MESSAGE(FATAL_ERROR "Failure generating describe-debugflags.h\n${PARSE_DEBUG_FLAGS_ERROR}")
ENDIF(NOT PARSE_DEBUG_FLAGS_HAS_ERROR EQUAL 0)

# ===================================================== #
# Compiling time!                                       #
# ===================================================== #
ADD_SUBDIRECTORY(src)
