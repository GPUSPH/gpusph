# ===================================================== #
# Dependencies                                          #
# ===================================================== #
IF(GPUSPH_USE_MPI)
    SET(OPTIONAL_INCLUDE_PATH ${OPTIONAL_INCLUDE_PATH} ${MPI_INCLUDE_PATH})
    SET(OPTIONAL_LIBS ${OPTIONAL_LIBS} ${MPI_LIBRARIES})
ENDIF(GPUSPH_USE_MPI)

IF(GPUSPH_USE_HDF5)
    SET(OPTIONAL_INCLUDE_PATH ${OPTIONAL_INCLUDE_PATH} ${HDF5_INCLUDE_DIRS})
    SET(OPTIONAL_LIBS ${OPTIONAL_LIBS} ${HDF5_LIBRARIES})
ENDIF(GPUSPH_USE_HDF5)

# CHRONO
IF(GPUSPH_USE_CHRONO)
    SET(OPTIONAL_INCLUDE_PATH ${OPTIONAL_INCLUDE_PATH} ${CHRONO_INCLUDE_DIRS})
    SET(OPTIONAL_LIBS ${OPTIONAL_LIBS} ${CHRONO_LIBRARIES})
ENDIF(GPUSPH_USE_CHRONO)

# ===================================================== #
# Include & Link                                        #
# ===================================================== #
INCLUDE_DIRECTORIES(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/cuda
    ${CMAKE_SOURCE_DIR}/src/geometries
    ${CMAKE_SOURCE_DIR}/src/problems
    ${CMAKE_SOURCE_DIR}/src/writers
    ${CUDA_INCLUDE_DIRS}
    ${OPTIONAL_INCLUDE_PATH}
)

SET(DEP_LIBS 
    ${CUDA_LIBRARIES}
    ${OPTIONAL_LIBS}
)

# ===================================================== #
# Sources to compile                                    #
# ===================================================== #
FILE(GLOB PROBLEM_CXX_SRCS
     "*.cc"
)

ADD_SUBDIRECTORY(cuda)
ADD_SUBDIRECTORY(geometries)
ADD_SUBDIRECTORY(writers)

SET(PROBLEM_CUDA_SRCS "${CMAKE_SOURCE_DIR}/src/problems/${GPUSPH_PROBLEM}.cu")

# ===================================================== #
# Target                                                #
# ===================================================== #
SOURCE_GROUP("GPUSPH" FILES ${PROBLEM_CXX_SRCS} ${PROBLEM_CUDA_SRCS} )
ADD_EXECUTABLE(GPUSPH ${PROBLEM_CXX_SRCS} ${PROBLEM_CUDA_SRCS})
SET_PROPERTY(TARGET GPUSPH
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(GPUSPH ${DEP_LIBS})

# ===================================================== #
# Install App                                           #
# ===================================================== #
SET_TARGET_PROPERTIES(GPUSPH PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})
SET_TARGET_PROPERTIES(GPUSPH PROPERTIES INSTALL_RPATH ${INSTALL_RPATH})

INSTALL(TARGETS GPUSPH
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
