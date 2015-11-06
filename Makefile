# GPUSPH Makefile
#
# Notes:
# - When adding a target, comment it with "# target: name - desc" and help
#   will always be up-to-date
# - When adding an option, comment it with "# option: name - desc" and help
#   will always be up-to-date
# - When adding an overridable setting, document it with "# override: name - desc" and help
#   will always be up-to-date
# - Makefile is assumed to be GNU (see http://www.gnu.org/software/make/manual/)
# - Source C++ files have extension .cc (NOT .cpp)
# - C++ Headers have extension .h
# - CUDA C++ files have extension .cu
# - CUDA C++ headers have extension .cuh


# Cached configuration. All settings that should be persistent across compilation
# (until changed) should be stored here.
sinclude Makefile.conf

# Include, if present, a local Makefile.
# This can be used by the user to set additional include paths
# (INCPATH = ....)
# library search paths
# (LIBPATH = ...)
# libaries
# (LIBS = ...)
# and general flags
# CPPFLAGS, CXXFLAGS, CUFLAGS, LDFLAGS,
sinclude Makefile.local

# need for some substitutions
comma:=,
empty:=
space:=$(empty) $(empty)

# GPUSPH version
GPUSPH_VERSION=$(shell git describe --tags --dirty=+custom 2> /dev/null | sed -e 's/-\([0-9]\+\)/+\1/' -e 's/-g/-/' 2> /dev/null)

ifeq ($(GPUSPH_VERSION), $(empty))
$(warning Unable to determine GPUSPH version)
GPUSPH_VERSION=unknown-version
endif

# system information
platform=$(shell uname -s 2>/dev/null)
platform_lcase=$(shell uname -s 2>/dev/null | tr '[:upper:]' '[:lower:]')
arch=$(shell uname -m)

# sed syntax differs a bit
ifeq ($(platform), Darwin)
	SED_COMMAND=sed -i "" -e
else # Linux
	SED_COMMAND=sed -i -e
endif


# option: target_arch - if set to 32, force compilation for 32 bit architecture
ifeq ($(target_arch), 32)
	arch=i686
endif

# name of the top-level Makefile (this file)
MFILE_NAME = $(firstword $(MAKEFILE_LIST))
MAKEFILE = $(CURDIR)/$(MFILE_NAME)

DBG_SFX=_dbg
ifeq ($(DBG), 1)
	TARGET_SFX=$(DBG_SFX)
else
	TARGET_SFX=
endif

# directories: binary, objects, sources, expanded sources
DISTDIR = dist/$(platform_lcase)/$(arch)
OBJDIR = build
SRCDIR = src
EXPDIR = $(SRCDIR)/expanded
SCRIPTSDIR = scripts
DOCSDIR = docs
OPTSDIR = options

# target binary
TARGETNAME := GPUSPH$(TARGET_SFX)
TARGET := $(DISTDIR)/$(TARGETNAME)

# binary to list compute capabilities of installed devices
LIST_CUDA_CC=$(SCRIPTSDIR)/list-cuda-cc


# --------------- File lists

# makedepend will generate dependencies in these file
GPUDEPS = $(MAKEFILE).gpu
CPUDEPS = $(MAKEFILE).cpu

# all files under $(SRCDIR), needed by tags files
ALLSRCFILES = $(shell find $(SRCDIR) -type f)

# .cc source files (CPU)
MPICXXFILES = $(SRCDIR)/NetworkManager.cc
ifeq ($(USE_HDF5),2)
	MPICXXFILES += $(SRCDIR)/HDF5SphReader.cc
endif

PROBLEM_DIR=$(SRCDIR)/problems
USER_PROBLEM_DIR=$(SRCDIR)/problems/user

SRCSUBS=$(sort $(filter %/,$(wildcard $(SRCDIR)/*/)))
SRCSUBS:=$(SRCSUBS:/=)
OBJSUBS=$(patsubst $(SRCDIR)/%,$(OBJDIR)/%,$(SRCSUBS) $(USER_PROBLEM_DIR))

# list of problems
PROBLEM_LIST = $(foreach adir, $(PROBLEM_DIR) $(USER_PROBLEM_DIR), \
	$(notdir $(basename $(wildcard $(adir)/*.h))))

# only one problem is active at a time, this is the list of all other problems
INACTIVE_PROBLEMS = $(filter-out $(PROBLEM),$(PROBLEM_LIST))
# we don't want to build inactive problems, so we will filter them out
# from the sources list
PROBLEM_FILTER = $(foreach adir, $(PROBLEM_DIR) $(USER_PROBLEM_DIR), \
	$(patsubst %,$(adir)/%.cc,$(INACTIVE_PROBLEMS)) \
	$(patsubst %,$(adir)/%.cu,$(INACTIVE_PROBLEMS)) \
	$(patsubst %,$(adir)/%_BC.cu,$(INACTIVE_PROBLEMS)))

# list of problem source files
PROBLEM_SRCS = $(foreach adir, $(PROBLEM_DIR) $(USER_PROBLEM_DIR), \
	$(filter \
		$(adir)/$(PROBLEM).cc \
		$(adir)/$(PROBLEM).cu \
		$(adir)/$(PROBLEM)_BC.cu,\
		$(wildcard $(adir)/*)))

# list of .cc files, exclusing MPI sources and disabled problems
CCFILES = $(filter-out $(PROBLEM_FILTER),\
	  $(filter-out $(MPICXXFILES),\
	  $(foreach adir, $(SRCDIR) $(SRCSUBS),\
	  $(wildcard $(adir)/*.cc))))

# GPU source files: we only directly compile the current problem (if it's CUDA)
# and cudautil.cu, everything else gets in by nested includes
CUFILES = $(SRCDIR)/cuda/cudautil.cu \
	  $(filter %.cu,$(PROBLEM_SRCS))

# headers
HEADERS = $(foreach adir, $(SRCDIR) $(SRCSUBS),$(wildcard $(adir)/*.h))

# object files via filename replacement
MPICXXOBJS = $(patsubst %.cc,$(OBJDIR)/%.o,$(notdir $(MPICXXFILES)))
CCOBJS = $(patsubst $(SRCDIR)/%.cc,$(OBJDIR)/%.o,$(CCFILES))
CUOBJS = $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(CUFILES))

OBJS = $(CCOBJS) $(MPICXXOBJS) $(CUOBJS)

# data files needed by some problems
EXTRA_PROBLEM_FILES ?=
# TestTopo uses this DEM:
EXTRA_PROBLEM_FILES += half_wave0.1m.txt

# --------------- Locate and set up compilers and flags

# override: CUDA_INSTALL_PATH - where CUDA is installed
# override:                     defaults /usr/local/cuda,
# override:                     validity is checked by looking for bin/nvcc under it,
# override:                     /usr is always tried as a last resort
CUDA_INSTALL_PATH ?= /usr/local/cuda

# We check the validity of the path by looking for bin/nvcc under it.
# if not found, we look into /usr, and finally abort
ifeq ($(wildcard $(CUDA_INSTALL_PATH)/bin/nvcc),)
	CUDA_INSTALL_PATH = /usr
	# check again
	ifeq ($(wildcard $(CUDA_INSTALL_PATH)/bin/nvcc),)
$(error Could not find CUDA, please set CUDA_INSTALL_PATH)
	endif
endif

# Here follow experimental CUDA installation detection. These work if CUDA binaries are in
# the current PATH (i.e. when using Netbeans without system PATH set, don't work).
# CUDA_INSTALL_PATH=$(shell which nvcc | sed "s/\/bin\/nvcc//")
# CUDA_INSTALL_PATH=$(shell which nvcc | head -c -10)
# CUDA_INSTALL_PATH=$(shell echo $PATH | sed "s/:/\n/g" | grep "cuda/bin" | sed "s/\/bin//g" |  head -n 1)
# ld-based CUDA location: more robust but problematic for Mac OS
#CUDA_INSTALL_PATH=$(shell \
#	dirname `ldconfig -p | grep libcudart | a$4}' | head -n 1` | head -c -5)

# nvcc info
NVCC=$(CUDA_INSTALL_PATH)/bin/nvcc
NVCC_VER=$(shell $(NVCC) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
versions_tmp  := $(subst ., ,$(NVCC_VER))
CUDA_MAJOR := $(firstword  $(versions_tmp))
CUDA_MINOR := $(lastword  $(versions_tmp))

# Some paths depend on whether we are on CUDA 5 or higher.
# CUDA_PRE_5 will be 0 if we are on CUDA 5 or higher, nonzero otherwise
# (please only test against 0, I'm not sure it will be a specific nonzero value)
# NOTE: the test is reversed because test returns 0 for true (shell-like)
CUDA_PRE_5=$(shell test $(CUDA_MAJOR) -ge 5; echo $$?)

# override: CUDA_SDK_PATH - location for the CUDA SDK samples
# override:                 defaults to $(CUDA_INSTALL_PATH)/samples for CUDA 5 or higher,
# override:                             /usr/local/cudasdk for older versions of CUDA.
ifeq ($(CUDA_PRE_5), 0)
	CUDA_SDK_PATH ?= $(CUDA_INSTALL_PATH)/samples
else
	CUDA_SDK_PATH ?= /usr/local/cudasdk
endif

# CXX is the host compiler. nvcc doesn't really allow you to use any compiler though:
# it only supports gcc (on both Linux and Darwin). Since CUDA 5.5 it also
# supports clang on Darwin 10.9, but only if the ccbin name actually contains the string
# 'clang'.
#
# The solution to our problem is the following:
# * we do not change anything, unless CXX is set to c++ (generic)
# * if CXX is generic, we set it to g++, _unless_ we are on Darwin, the CUDA
#   version is at least 5.5 and clang++ is executable, in which case we set it to /usr/bin/clang++
# There are cases in which this might fail, but we'll fix it when we actually come across them
WE_USE_CLANG=0

# override: CXX - the host C++ compiler.
# override:       defaults to g++, except on Darwin
# override:       where clang++ is used if available
ifeq ($(CXX),c++)
	CXX = g++
	ifeq ($(platform), Darwin)
		versions_tmp:=$(shell [ -x /usr/bin/clang++ -a $(CUDA_MAJOR)$(CUDA_MINOR) -ge 55 ] ; echo $$?)
		ifeq ($(versions_tmp),0)
			CXX = /usr/bin/clang++
			WE_USE_CLANG=1
		endif
	endif
endif

# Force nvcc to use the same host compiler that we selected
# Note that this requires the compiler to be supported by
# nvcc.
NVCC += -ccbin=$(CXX)


# files to store last compile options: problem, dbg, compute, fastmath, MPI usage
PROBLEM_SELECT_OPTFILE=$(OPTSDIR)/problem_select.opt
DBG_SELECT_OPTFILE=$(OPTSDIR)/dbg_select.opt
COMPUTE_SELECT_OPTFILE=$(OPTSDIR)/compute_select.opt
FASTMATH_SELECT_OPTFILE=$(OPTSDIR)/fastmath_select.opt
MPI_SELECT_OPTFILE=$(OPTSDIR)/mpi_select.opt
HDF5_SELECT_OPTFILE=$(OPTSDIR)/hdf5_select.opt

# this is not really an option, but it follows the same mechanism
GPUSPH_VERSION_OPTFILE=$(OPTSDIR)/gpusph_version.opt

OPTFILES=$(PROBLEM_SELECT_OPTFILE) \
		 $(DBG_SELECT_OPTFILE) \
		 $(COMPUTE_SELECT_OPTFILE) \
		 $(FASTMATH_SELECT_OPTFILE) \
		 $(MPI_SELECT_OPTFILE) \
		 $(HDF5_SELECT_OPTFILE) \
		 $(GPUSPH_VERSION_OPTFILE)

# Let make know that .opt and .i dependencies are to be looked for in $(OPTSDIR)
vpath %.opt $(OPTSDIR)

# update GPUSPH_VERSION_OPTFILE if git version changed
LAST_GPUSPH_VERSION=$(shell test -e $(GPUSPH_VERSION_OPTFILE) && \
	grep "\#define GPUSPH_VERSION" $(GPUSPH_VERSION_OPTFILE) | cut -f2 -d\")

ifneq ($(LAST_GPUSPH_VERSION),$(GPUSPH_VERSION))
	TMP:=$(shell test -e $(GPUSPH_VERSION_OPTFILE) && \
		$(SED_COMMAND) 's/$(LAST_GPUSPH_VERSION)/$(GPUSPH_VERSION)/' $(GPUSPH_VERSION_OPTFILE) )
endif


# option: problem - Name of the problem. Default: $(PROBLEM) in makefile
ifdef problem
	# if choice differs from last...
	ifneq ($(PROBLEM),$(problem))
		# check that the problem is in the problem list
		ifneq ($(filter $(problem),$(PROBLEM_LIST)),$(problem))
			TMP:=$(error No such problem ‘$(problem)’. Known problems: $(PROBLEM_LIST))
		endif
		# empty string in sed for Mac compatibility
		TMP:=$(shell test -e $(PROBLEM_SELECT_OPTFILE) && \
			$(SED_COMMAND) 's:$(PROBLEM):$(problem):' $(PROBLEM_SELECT_OPTFILE) )
		# user choice
		PROBLEM=$(problem)
	endif
else
	PROBLEM ?= DamBreak3D
endif

# option: dbg - 0 no debugging, 1 enable debugging
# does dbg differ from last?
ifdef dbg
	ifneq ($(DBG), $(dbg))
		ifeq ($(dbg),1)
			_SRC=undef
			_REP=define
		else
			_SRC=define
			_REP=undef
		endif
		TMP:=$(shell test -e $(DBG_SELECT_OPTFILE) && \
			$(SED_COMMAND) 's/$(_SRC)/$(_REP)/' $(DBG_SELECT_OPTFILE) )
		DBG=$(dbg)
	endif
endif

# option: compute - 11, 12, 13, 20, 21, 30, 35, etc: compute capability to compile for (default: autodetect)
# does dbg differ from last?
ifdef compute
	# does it differ from last?
	ifneq ($(COMPUTE),$(compute))
		TMP:=$(shell test -e $(COMPUTE_SELECT_OPTFILE) && \
			$(SED_COMMAND) 's/$(COMPUTE)/$(compute)/' $(COMPUTE_SELECT_OPTFILE) )
		# user choice
		COMPUTE=$(compute)
	endif
endif

# option: fastmath - Enable or disable fastmath. Default: 0 (disabled)
ifdef fastmath
	# does it differ from last?
	ifneq ($(FASTMATH),$(fastmath))
		TMP:=$(shell test -e $(FASTMATH_SELECT_OPTFILE) && \
			$(SED_COMMAND) 's/FASTMATH $(FASTMATH)/FASTMATH $(fastmath)/' $(FASTMATH_SELECT_OPTFILE) )
		# user choice
		FASTMATH=$(fastmath)
	endif
else
	FASTMATH ?= 0
endif

# option: mpi - 0 do not use MPI (no multi-node support), 1 use MPI (enable multi-node support). Default: autodetect
ifdef mpi
	# does it differ from last?
	ifneq ($(USE_MPI),$(mpi))
		TMP := $(shell test -e $(MPI_SELECT_OPTFILE) && \
			$(SED_COMMAND) 's/$(USE_MPI)/$(mpi)/' $(MPI_SELECT_OPTFILE) )
		# user choice
		USE_MPI=$(mpi)
	endif
endif

# override: MPICXX - the MPI compiler
MPICXX ?= $(shell which mpicxx 2> /dev/null)

ifeq ($(MPICXX),)
	ifeq ($(USE_MPI),1)
		TMP := $(error MPI use for requested, but no MPI compiler was found, aborting)
	else
		ifeq ($(USE_MPI),)
			TMP := $(info MPI compiler not found, multi-node will NOT be supported)
			USE_MPI = 0
		endif
	endif
else
	ifeq ($(USE_MPI),)
		USE_MPI = 1
	endif
endif

ifeq ($(USE_MPI),0)

	# MPICXXOBJS will be compiled with the standard compiler
	MPICXX=$(CXX)

	# We have to link with NVCC because otherwise thrust has issues on Mac OSX.
	LINKER ?= $(NVCC)

else

	# We have to link with NVCC because otherwise thrust has issues on Mac OSX,
	# but we also need to link with MPICXX, so we would like to use:
	#LINKER ?= $(filter-out -ccbin=%,$(NVCC)) -ccbin=$(MPICXX)
	# which fails on recent Mac OSX because then NVCC thinks we're compiling with the GNU
	# compiler, and thus passes the -dumpspecs option to it, which fails because Mac OSX
	# is actually using clang in the end. ‘Gotta love them heuristics’, as Kevin puts it.
	# The solution is to _still_ use NVCC with -ccbin=$(CXX) as linker, but add the
	# options required by MPICXX at link time:

	## TODO FIXME this is a horrible hack, there should be a better way to handle this nvcc+mpicxx mess
	# We use -show because it's supported by all implementations (otherwise we'd have to detect
	# if our compiler uses --showme:link or -link_info):
	MPISHOWFLAGS := $(shell $(MPICXX) -show)
	# But then we have to remove the compiler name from the proposed command line:
	MPISHOWFLAGS := $(filter-out $(firstword $(MPISHOWFLAGS)),$(MPISHOWFLAGS))

	# mpicxx might pass to the compiler options which nvcc might not understand
	# (e.g. -pthread), so we need to pass mpicxx options through --compiler-options,
	# but that means that we cannot pass options which contains commas in them,
	# since commas are already used to separate the parameters in --compiler-options.
	# We can't do sophisticated patter-matching (e.g. filtering on strings _containing_
	# a comma), so for the time being we just filter out the flags that we _know_
	# will contain commas (i.e. -Wl,stuff,stuff,stuff).
	# To make things even more complicated, nvcc does not accept -Wl, so we need
	# to replace -Wl with --linker-options.
	# The other options will be gathered into the --compiler-options passed at nvcc
	# at link time.
	MPILDFLAGS = $(subst -Wl$(comma),--linker-options$(space),$(filter -Wl%,$(MPISHOWFLAGS))) $(filter -L%,$(MPISHOWFLAGS)) $(filter -l%,$(MPISHOWFLAGS))
	MPICXXFLAGS = $(filter-out -L%,$(filter-out -l%,$(filter-out -Wl%,$(MPISHOWFLAGS))))

	LINKER ?= $(NVCC) --compiler-options $(subst $(space),$(comma),$(strip $(MPICXXFLAGS))) $(MPILDFLAGS)

	# (the solution is not perfect as it still generates some warnings, but at least it rolls)

endif

# END of MPICXX mess

# override: HDF5_CPP - preprocessor flags to find/use HDF5
HDF5_CPP ?= $(shell pkg-config --cflags-only-I hdf5 2> /dev/null)
# override: HDF5_CXX - compiler flags to use HDF5
HDF5_CXX ?= $(shell pkg-config --cflags-only-other hdf5 2> /dev/null)
# override: HDF5_LD - LD flags to use HDF5
HDF5_LD ?= $(shell pkg-config --libs hdf5 2> /dev/null || echo -lhdf5)

# option: hdf5 - 0 do not use HDF5, 1 use HDF5, 2 use HDF5 and HDF5 requires MPI. Default: autodetect
ifdef hdf5
	# does it differ from last?
	ifneq ($(USE_HDF5),$(hdf5))
		TMP := $(shell test -e $(HDF5_SELECT_OPTFILE) && \
			$(SED_COMMAND) 's/$(USE_HDF5)/$(hdf5)/' $(HDF5_SELECT_OPTFILE) )
		# user choice
		USE_HDF5=$(hdf5)
	endif
else
	# Check if we can link to the HDF5 library, and disable HDF5 otherwise.
	# On some configurations, HDF5 requires MPI, so we check HDF5 twice,
	# once with CXX and once with MPICXX.
	# During the CXX test we return -1 in case of failure to differentiate from
	# a case such as 'make hdf5=0 ; make', in which case we want to skip also
	# the MPICXX test.
	# We use a for loop in the shell to echo each line because users might have
	# different interactive shells that do (or do not) interpret a \n escape,
	# so the only portable way seems to echo each line separately,
	# and grouping the echos in { } doesn't seem to work from a Makefile
	# shell invocation
	USE_HDF5 ?= $(shell for line in '\#include <hdf5.h>' 'main(){}' ; do echo $$line ; done | $(CXX) -xc++ $(INCPATH) $(LIBPATH) $(HDF5_CPP) $(HDF5_CXX) $(HDF5_LD) -o /dev/null - 2> /dev/null && echo 1 || echo -1)
	ifeq ($(USE_HDF5),-1)
		USE_HDF5 := $(shell for line in '\#include <hdf5.h>' 'main(){}' ; do echo $$line ; done | $(MPICXX) -xc++ $(INCPATH) $(LIBPATH) $(HDF5_CPP) $(HDF5_CXX) $(HDF5_LD) -o /dev/null - 2> /dev/null && echo 2 || echo 0)
		ifeq ($(USE_HDF5),0)
			TMP := $(info HDF5 library not found, HDF5 input will NOT be supported)
		endif
	endif
endif

# --- Includes and library section start ---

LIB_PATH_SFX =

# override: TARGET_ARCH - set the target architecture
# override:               defaults to -m64 for 64-bit machines
# override:                           -m32 for 32-bit machines
ifeq ($(arch), x86_64)
	TARGET_ARCH ?= -m64
	# on Linux, toolkit libraries are under /lib64 for 64-bit
	ifeq ($(platform), Linux)
		LIB_PATH_SFX = 64
	endif
else # i386 or i686
	TARGET_ARCH ?= -m32
endif

# override: INCPATH - paths for include files
# override:           add entries in the form: -I/some/path
INCPATH ?=
# override: LIBPATH - paths for library searches
# override:           add entries in the form: -L/some/path
LIBPATH ?=
# override: LIBS - additional libraries
# override:        add entries in the form: -lsomelib
LIBS ?=

# override: LDFLAGS - flags passed to the linker
LDFLAGS ?=

# override: LDLIBS - libraries to link against
LDLIBS ?=

# Most of these settings are platform independent

# INCPATH
# make GPUSph.cc find problem_select.opt, and problem_select.opt find the problem header
INCPATH += -I$(SRCDIR) $(foreach adir,$(SRCSUBS),-I$(adir)) -I$(USER_PROBLEM_DIR) -I$(OPTSDIR)

# access the CUDA include files from the C++ compiler too, but mark their path as a system include path
# so that they can be skipped when generating dependencies. This must only be done for the host compiler,
# because otherwise some nvcc version will complain about kernels not being allowed in system files
# while compiling some thrust functions
CC_INCPATH += -isystem $(CUDA_INSTALL_PATH)/include

# LIBPATH
LIBPATH += -L/usr/local/lib

# CUDA libaries
LIBPATH += -L$(CUDA_INSTALL_PATH)/lib$(LIB_PATH_SFX)

# On Darwin 10.9 with CUDA 5.5 using clang we want to link with the clang c++ stdlib.
# This is exactly the conditions under which we set WE_USE_CLANG
ifeq ($(WE_USE_CLANG),1)
	LIBS += -lstdc++
endif

# link to the CUDA runtime library
LIBS += -lcudart
# link to ODE for the objects
LIBS += -lode

ifneq ($(USE_HDF5),0)
	# link to HDF5 for input reading
	LIBS += $(HDF5_LD)
endif

# pthread needed for the UDP writer
LIBS += -lpthread

# Realtime Extensions library (for clock_gettime) (not on Mac)
ifneq ($(platform), Darwin)
	LIBS += -lrt
endif

# search paths (for CUDA 5 and higher) are platform-specific
ifeq ($(platform), Darwin)
	LIBPATH += -L$(CUDA_SDK_PATH)/common/lib/$(platform_lcase)/
else
	LIBPATH += -L$(CUDA_SDK_PATH)/common/lib/$(platform_lcase)/$(arch)/
endif

LDFLAGS += $(LIBPATH)

LDLIBS += $(LIBS)

# -- Includes and library section end ---

# -------------------------- CFLAGS section -------------------------- #
# We have three sets of flags:
# CPPFLAGS are preprocessor flags; they are common to both compilers
# CXXFLAGS are flags passed to the C++ when compiling C++ files (either directly,
#     for .cc files, or via nvcc, for .cu files), and when linking
# CUFLAGS are flags passed to the CUDA compiler when compiling CUDA files

# override: CPPFLAGS - preprocessor flags
CPPFLAGS ?=
# override: CXXFLAGS - C++ host compiler options
CXXFLAGS ?=
# override: CUFLAGS - nvcc compiler options
CUFLAGS  ?=

# First of all, put the include paths into the CPPFLAGS
CPPFLAGS += $(INCPATH)

# We use type limits and constants (e.g. UINT64_MAX), which are defined
# in C99 but not in C++ versions before C++11, so on (very) old compilers
# (e.g. gcc 4.1) they will not be available. The workaround for this
# is to define the __STDC_LIMIT_MACROS and __STDC_CONSTANT_MACROS.
# Put their definition in the command line to ensure it precedes any
# (direct or indirect) inclusion of stdint.h
CPPFLAGS += -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS

# Define USE_HDF5 according to the availability of the HDF5 library
CPPFLAGS += -DUSE_HDF5=$(USE_HDF5)
ifneq ($(USE_HDF5),0)
	CPPFLAGS += $(HDF5_CPP)
endif

# We set __COMPUTE__ on the host to match that automatically defined
# by the compiler on the device. Since this might be done before COMPUTE
# is actually defined, substitute 0 in that case
ifeq ($(COMPUTE),)
	CPPFLAGS += -D__COMPUTE__=0
else
	CPPFLAGS += -D__COMPUTE__=$(COMPUTE)
endif

# The ODE library link is in single precision mode
CPPFLAGS += -DdSINGLE

# CXXFLAGS start with the target architecture
CXXFLAGS += $(TARGET_ARCH)

# HDF5 might require specific flags
ifneq ($(USE_HDF5),0)
	CXXFLAGS += $(HDF5_CXX)
endif

# nvcc-specific flags

# We want to know the version of NVCC in the code because
# the GCC pragma support depends on NVCC version. Sadly,
# the pre-defined macros __NVCC__ and __CUDACC__ do not
# give us anything about the version, so we will define our own
# macros

CUFLAGS += -D__NVCC_VERSION__=$(CUDA_MAJOR)$(CUDA_MINOR)

# compute capability specification, if defined
ifneq ($(COMPUTE),)
	CUFLAGS += -arch=sm_$(COMPUTE)
	LDFLAGS += -arch=sm_$(COMPUTE)
endif

# generate line info on CUDA 5 or higher
ifeq ($(CUDA_PRE_5),0)
	CUFLAGS += --generate-line-info
endif

ifeq ($(FASTMATH),1)
	CUFLAGS += --use_fast_math
endif


# Note: -D_DEBUG_ is defined in $(DBG_SELECT_OPTFILE); however, to avoid adding an
# include to every source, the _DEBUG_ macro is actually passed on the compiler command line
ifeq ($(dbg), 1)
	CPPFLAGS += -D_DEBUG_
	CXXFLAGS += -g
	CUFLAGS  += -G
else
	CXXFLAGS += -O3
endif
# add -O3 anyway - workaround to fix a linking error when compiling with dbg=1, CUDA 5.0
CXXFLAGS += -O3

# option: verbose - 0 quiet compiler, 1 ptx assembler, 2 all warnings
ifeq ($(verbose), 1)
	CUFLAGS += --ptxas-options=-v
else ifeq ($(verbose), 2)
	CUFLAGS += --ptxas-options=-v
	CXXFLAGS += -Wall
endif

# Enable host profile with gprof. Pipeline to profile:
# enable -pg, make, run, gprof ./GPUSPH gmon.out > results.txt
# http://gcc.gnu.org/onlinedocs/gcc/Debugging-Options.html#index-pg-621
# CXXFLAGS += -pg
# LDFLAGS += -pg

# Finally, add CXXFLAGS to CUFLAGS

CUFLAGS += --compiler-options $(subst $(space),$(comma),$(strip $(CXXFLAGS)))

# CFLAGS notes
# * Architecture (sm_XX and compute_XX):
#    sm_10 - NOT supported: we need atomics
#    sm_11 - Compute Capability 1.1: "old" cards, no double
#    sm_12 - Compute Capability 1.2: G92... cards (e.g. GTX280), hw double
#    sm_20 - Compute Capability 2.0: Fermi generation cards (e.g. GTX480), hw double
#    Note: choosing only one architecture may speed up compilation up to 200%
#    e.g.: for forces.o on a Vaio laptop, it takes 4m for one arch, 8m for boh
# * To compile for two archs use a syntax like:
#    -gencode arch=compute_12,code=sm_12 -gencode arch=compute_20,code=sm_20
# * -O3 may not be supported by nvcc (check)
# * To add options only for the C compiler, add them after --compiler-options.
# * -fno-strict-aliasing: see http://goo.gl/cIkzG
# * For Mac: it is mac-specific (but not required) to add:
#    CFLAGS += -D__APPLE__ -D__MACH__

# ------------------------ CFLAGS section end ---------------------- #

# Doxygen configuration
DOXYCONF = ./Doxygen_settings

# Snapshot date: date of last commit (if possible), or current date
snap_date := $(shell git log -1 --format='%cd' --date=iso 2> /dev/null | cut -f1,4 -d' ' | tr ' ' '-' || date +%Y-%m-%d)
# snapshot tarball filename
SNAPSHOT_FILE = ./GPUSPH-$(GPUSPH_VERSION)-$(snap_date).tgz

# option: plain - 0 fancy line-recycling stage announce, 1 plain multi-line stage announce
ifeq ($(plain), 1)
	show_stage=@printf "[$(1)] $(2)\n"
	show_stage_nl=$(show_stage)
else
	show_stage=@printf "\r                                 \r[$(1)] $(2)"
	show_stage_nl=@printf "\r                                 \r[$(1)] $(2)\n"
endif

# when listing problems, we don't want debug info to show anywhere
ifeq ($(MAKECMDGOALS), list-problems)
	show_stage=@
	show_stage_nl=@
endif

# option: echo - 0 silent, 1 show commands
ifeq ($(echo), 1)
	CMDECHO :=
else
	CMDECHO := @
endif

.PHONY: all run showobjs show snapshot expand deps docs test help
.PHONY: clean cpuclean gpuclean cookiesclean computeclean docsclean confclean

# target: all - Make subdirs, compile objects, link and produce $(TARGET)
# link objects in target
all: $(OBJS) | $(DISTDIR)
	@echo
	@echo "Compiled with problem $(PROBLEM)"
	@[ $(FASTMATH) -eq 1 ] && echo "Compiled with fastmath" || echo "Compiled without fastmath"
	$(call show_stage_nl,LINK,$(TARGET))
	$(CMDECHO)$(LINKER) -o $(TARGET) $(OBJS) $(LDFLAGS) $(LDLIBS) && \
	ln -sf $(TARGET) $(CURDIR)/$(TARGETNAME) && echo "Success."

# target: run - Make all && run
run: all
	$(TARGET)

# internal targets to (re)create the "selected option headers" if they're missing
$(PROBLEM_SELECT_OPTFILE): | $(OPTSDIR)
	@echo "/* Define the problem compiled into the main executable. */" \
		> $(PROBLEM_SELECT_OPTFILE)
	@echo "#define PROBLEM $(PROBLEM)" >> $(PROBLEM_SELECT_OPTFILE)
	@echo "#define QUOTED_PROBLEM \"$(PROBLEM)\"" >> $(PROBLEM_SELECT_OPTFILE)
	@echo "#include \"$(PROBLEM).h\"" >> $(PROBLEM_SELECT_OPTFILE)
$(DBG_SELECT_OPTFILE): | $(OPTSDIR)
	@echo "/* Define if debug option is on. */" \
		> $(DBG_SELECT_OPTFILE)
	@if test "$(dbg)" = "1" ; then echo "#define _DEBUG_" >> $(DBG_SELECT_OPTFILE); \
	else echo "#undef _DEBUG_" >> $(DBG_SELECT_OPTFILE); fi
$(COMPUTE_SELECT_OPTFILE): $(LIST_CUDA_CC) | $(OPTSDIR)
	@echo "/* Define the compute capability GPU code was compiled for. */" \
		> $(COMPUTE_SELECT_OPTFILE)
	$(call show_stage_nl,SCRIPTS,compute detection)
	@$(SCRIPTSDIR)/define-cuda-cc.sh $(COMPUTE) >> $(COMPUTE_SELECT_OPTFILE)
$(FASTMATH_SELECT_OPTFILE): | $(OPTSDIR)
	@echo "/* Determines if fastmath is enabled for GPU code. */" \
		> $@
	@echo "#define FASTMATH $(FASTMATH)" >> $@
$(MPI_SELECT_OPTFILE): | $(OPTSDIR)
	@echo "/* Determines if we are using MPI (for multi-node) or not. */" \
		> $@
	@echo "#define USE_MPI $(USE_MPI)" >> $@
$(HDF5_SELECT_OPTFILE): | $(OPTSDIR)
	@echo "/* Determines if we are using HDF5 or not. */" \
		> $@
	@echo "#define USE_HDF5 $(USE_HDF5)" >> $@

$(GPUSPH_VERSION_OPTFILE): | $(OPTSDIR)
	@echo "/* git version of GPUSPH. */" \
		> $@
	@echo "#define GPUSPH_VERSION \"$(GPUSPH_VERSION)\"" >> $@


$(OBJS): $(DBG_SELECT_OPTFILE)

# compile CPU objects
$(CCOBJS): $(OBJDIR)/%.o: $(SRCDIR)/%.cc | $(OBJSUBS)
	$(call show_stage,CC,$(@F))
	$(CMDECHO)$(CXX) $(CC_INCPATH) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

$(MPICXXOBJS): $(OBJDIR)/%.o: $(SRCDIR)/%.cc | $(OBJSUBS)
	$(call show_stage,MPI,$(@F))
	$(CMDECHO)$(MPICXX) $(CC_INCPATH) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

# compile GPU objects
$(CUOBJS): $(OBJDIR)/%.o: $(SRCDIR)/%.cu $(COMPUTE_SELECT_OPTFILE) $(FASTMATH_SELECT_OPTFILE) | $(OBJSUBS)
	$(call show_stage,CU,$(@F))
	$(CMDECHO)$(NVCC) $(CPPFLAGS) $(CUFLAGS) -c -o $@ $<

# compile program to list compute capabilities of installed devices.
# Filter out -arch=sm_$(COMPUTE) from LDFLAGS becaus we already have it in CUFLAGS
# and it being present twice causes complains from nvcc
$(LIST_CUDA_CC): $(LIST_CUDA_CC).cc
	$(call show_stage,SCRIPTS,$(@F))
	$(CMDECHO)$(NVCC) $(CPPFLAGS) $(filter-out --ptxas-options=%,$(filter-out --generate-line-info,$(CUFLAGS))) -o $@ $< $(filter-out -arch=sm_%,$(LDFLAGS))

# create distdir
$(DISTDIR):
	$(CMDECHO)mkdir -p $(DISTDIR)

# create objdir and subs
$(OBJDIR) $(OBJSUBS):
	$(CMDECHO)mkdir -p $(OBJDIR) $(OBJSUBS)

# create optsdir
$(OPTSDIR):
	$(CMDECHO)mkdir -p $(OPTSDIR)

# target: clean - Clean everything but last compile choices
# clean: cpuobjs, gpuobjs, deps makefiles, target, target symlink, dbg target
clean: cpuclean gpuclean
	$(RM) $(TARGET) $(CURDIR)/$(TARGETNAME)
	if [ -f $(TARGET)$(DBG_SFX) ] ; then \
		$(RM) $(TARGET)$(DBG_SFX) $(CURDIR)/$(TARGETNAME)$(DBG_SFX) ; fi

# target: cpuclean - Clean CPU stuff
cpuclean:
	$(RM) $(CCOBJS) $(MPICXXOBJS) $(CPUDEPS)

# target: gpuclean - Clean GPU stuff
gpuclean: computeclean
	$(RM) $(CUOBJS) $(GPUDEPS)

# target: computeclean - Clean compute capability selection stuff
computeclean:
	$(RM) $(LIST_CUDA_CC) $(COMPUTE_SELECT_OPTFILE)
	$(SED_COMMAND) '/COMPUTE=/d' Makefile.conf

# target: cookiesclean - Clean last dbg, problem, compute and fastmath choices,
# target:                forcing .*_select.opt files to be regenerated (use if they're messed up)
cookiesclean:
	$(RM) -r $(OPTFILES) $(OPTSDIR)

# target: confclean - Clean all configuration options: like cookiesclean, but also purges Makefile.conf
confclean: cookiesclean
	$(RM) -f Makefile.conf

# target: showobjs - List detected sources and target objects
showobjs:
	@echo "> CCFILES: $(CCFILES)"
	@echo " --- "
	@echo "> MPICXXFILES: $(MPICXXFILES)"
	@echo " --- "
	@echo "> CUFILES: $(CUFILES)"
	@echo " --- "
	@echo "> CCOBJS: $(CCOBJS)"
	@echo " --- "
	@echo "> MPICXXOBJS: $(MPICXXOBJS)"
	@echo " --- "
	@echo "> CUOBJS: $(CUOBJS)"
	@echo " --- "
	@echo "> OBJS: $(OBJS)"

# target: show - Show platform info and compiling options
show:
	@echo "GPUSPH version:  $(GPUSPH_VERSION)"
	@echo "Platform:        $(platform)"
	@echo "Architecture:    $(arch)"
	@echo "Current dir:     $(CURDIR)"
	@echo "This Makefile:   $(MAKEFILE)"
	@echo "Problem:         $(PROBLEM)"
#	@echo "   last:         $(LAST_PROBLEM)"
	@echo "Snapshot file:   $(SNAPSHOT_FILE)"
	@echo "Target binary:   $(TARGET)"
	@echo "Sources dir:     $(SRCDIR) $(SRCSUBS)"
	@echo "Options dir:     $(OPTSDIR)"
	@echo "Objects dir:     $(OBJDIR) $(OBJSUBS)"
	@echo "Scripts dir:     $(SCRIPTSDIR)"
	@echo "Docs dir:        $(DOCSDIR)"
	@echo "Doxygen conf:    $(DOXYCONF)"
	@echo "Verbose:         $(verbose)"
	@echo "Debug:           $(DBG)"
	@echo "CXX:             $(CXX)"
	@echo "MPICXX:          $(MPICXX)"
	@echo "nvcc:            $(NVCC)"
	@echo "nvcc version:    $(NVCC_VER)"
	@echo "LINKER:          $(LINKER)"
	@echo "Compute cap.:    $(COMPUTE)"
	@echo "Fastmath:        $(FASTMATH)"
	@echo "USE_MPI:         $(USE_MPI)"
	@echo "USE_HDF5:        $(USE_HDF5)"
	@echo "INCPATH:         $(INCPATH)"
	@echo "LIBPATH:         $(LIBPATH)"
	@echo "LIBS:            $(LIBS)"
	@echo "LDFLAGS:         $(LDFLAGS)"
	@echo "CPPFLAGS:        $(CPPFLAGS)"
	@echo "CXXFLAGS:        $(CXXFLAGS)"
	@echo "CUFLAGS:         $(CUFLAGS)"
#	@echo "Suffixes:        $(SUFFIXES)"

# target: snapshot - Make a snapshot of current sourcecode in $(SNAPSHOT_FILE)
# it seems tar option --totals doesn't work
# use $(shell date +%F_%T) to include date and time in filename
snapshot: $(SNAPSHOT_FILE)
	$(CMDECHO)echo "Created $(SNAPSHOT_FILE)"

# One possibility to add the source files: $(SRCDIR)/*.{cc,h} $(SRCDIR)/*.{cc,h,cu,cuh,def}
# However, Makefile does not support this bash-like expansion, so we take a shortcut.
$(SNAPSHOT_FILE):  ./$(MFILE_NAME) $(EXTRA_PROBLEM_FILES) $(DOXYCONF) $(SRCDIR)/ $(SCRIPTSDIR)/
	$(CMDECHO)tar czf $@ $^

# target: expand - Expand euler* and forces* GPU code in $(EXPDIR)
# it is safe to say we don't actualy need this
expand:
	$(CMDECHO)mkdir -p $(EXPDIR)
	$(CMDECHO)$(NVCC) $(CPPFLAGS) $(CUFLAGS) -E \
		$(SRCDIR)/euler.cu -o $(EXPDIR)/euler.expand.cc && \
	$(NVCC) $(CPPFLAGS) $(CUFLAGS) -E \
		$(SRCDIR)/euler_kernel.cu -o $(EXPDIR)/euler_kernel.expand.cc && \
	$(NVCC) $(CPPFLAGS) $(CUFLAGS) -E \
		$(SRCDIR)/forces.cu -o $(EXPDIR)/forces.expand.cc && \
	$(NVCC) $(CPPFLAGS) $(CUFLAGS) -E \
		$(SRCDIR)/forces_kernel.cu -o $(EXPDIR)/forces_kernel.expand.cc && \
	echo "euler* and forces* expanded in $(EXPDIR)."

# target: deps - Update dependencies in $(MAKEFILE)
deps: $(GPUDEPS) $(CPUDEPS)
	@true

# We want all of the OPTFILES to be built before anything else, which we achieve by
# making Makefile.conf depend on them.
Makefile.conf: Makefile $(OPTFILES)
	$(call show_stage,CONF,$@)
	$(CMDECHO)# Create Makefile.conf with standard disclaimer
	$(CMDECHO)echo '# This is an autogenerated configuration file. Please DO NOT edit it manually' > $@
	$(CMDECHO)echo '# Run make with the appropriate option to change a configured value' >> $@
	$(CMDECHO)echo '# Use `make help-options` to see a list of available options' >> $@
	$(CMDECHO)echo '# Use `make confclean` to reset your configuration' >> $@
	$(CMDECHO)# recover value of PROBLEM from OPTFILES
	$(CMDECHO)grep "\#define PROBLEM" $(PROBLEM_SELECT_OPTFILE) | cut -f2-3 -d ' ' | tr ' ' '=' >> $@
	$(CMDECHO)# recover value of _DEBUG_ from OPTFILES
	$(CMDECHO)echo "DBG=$$(grep '\#define _DEBUG_' $(DBG_SELECT_OPTFILE) | wc -l)" >> $@
	$(CMDECHO)# recover value of COMPUTE from OPTFILES
	$(CMDECHO)grep "\#define COMPUTE" $(COMPUTE_SELECT_OPTFILE) | cut -f2-3 -d ' ' | tr ' ' '=' >> $@
	$(CMDECHO)# recover value of FASTMATH from OPTFILES
	$(CMDECHO)grep "\#define FASTMATH" $(FASTMATH_SELECT_OPTFILE) | cut -f2-3 -d ' ' | tr ' ' '=' >> $@
	$(CMDECHO)# recover value of USE_MPI from OPTFILES
	$(CMDECHO)grep "\#define USE_MPI" $(MPI_SELECT_OPTFILE) | cut -f2-3 -d ' ' | tr ' ' '=' >> $@
	$(CMDECHO)# recover value of USE_HDF5 from OPTFILES
	$(CMDECHO)grep "\#define USE_HDF5" $(HDF5_SELECT_OPTFILE) | cut -f2-3 -d ' ' | tr ' ' '=' >> $@

# Dependecies are generated by the C++ compiler, since nvcc does not understand the
# more sophisticated -MM and -MT dependency generation options.
# The -MM flag is used to not include system includes.
# The -MG flag is used to add missing includes (useful to depend on the .opt files).
# The -MT flag is used to define the object file.
#
# We need to process each source file independently because of the way -MT works.
#
# When generating the dependencies for the .cu files, we must specify that they are
# to be interpeted as C++ files and not some other funky format. We also need
# to define __CUDA_INTERNAL_COMPILATION__ to mute an error during traversal of
# some CUDA system includes
#
# Both GPUDEPS and CPUS also depend from Makefile.conf, to ensure they are rebuilt when
# e.g. the problem changes. This avoids a situation like the following:
# * developer builds with problem A
# * developer builds with problem B
# * developer changes e.g. a kernel file
# * developer builds with problem B => it gets compiled new
# * developer builds with problem A => A doesn't get recompiled because the deps
#   file only have the deps for B, not A, and the .o file for A is there already
# This is particularly important to ensure that `make compile-problems` works correctly.
# Of course, Makefile.conf has to be stripped from the list of dependencies before passing them
# to the loop that builds the deps.
$(GPUDEPS): $(CUFILES) Makefile.conf
	$(call show_stage,DEPS,GPU)
	$(CMDECHO)echo '# GPU sources dependencies generated with "make deps"' > $@
	$(CMDECHO)for srcfile in $(filter-out Makefile.conf,$^) ; do \
		objfile="$(OBJDIR)/$${srcfile#$(SRCDIR)/}" ; \
		objfile="$${objfile%.*}.o" ; \
		$(CXX) -x c++ \
			-D__CUDA_INTERNAL_COMPILATION__ $(CC_INCPATH) $(CPPFLAGS) \
			$(filter -D%,$(CUFLAGS)) \
		-MG -MM $$srcfile -MT $$objfile >> $@ ; \
		done

$(CPUDEPS): $(CCFILES) $(MPICXXFILES) Makefile.conf
	$(call show_stage,DEPS,CPU)
	$(CMDECHO)echo '# CPU sources dependencies generated with "make deps"' > $@
	$(CMDECHO)for srcfile in $(filter-out Makefile.conf,$^) ; do \
		objfile="$(OBJDIR)/$${srcfile#$(SRCDIR)/}" ; \
		objfile="$${objfile%.*}.o" ; \
		$(CXX) $(CC_INCPATH) $(CPPFLAGS) \
		-MG -MM $$srcfile -MT $$objfile >> $@ ; \
		done

# target: docs - Generate Doxygen documentation in $(DOCSDIR);
# target:        to produce refman.pdf, run "make pdf" in $(DOCSDIR)/latex/.
docs: $(DOXYCONF) img/logo.png
	$(CMDECHO)mkdir -p $(DOCSDIR)
	@echo Running Doxygen...
	$(CMDECHO)doxygen $(DOXYCONF) && \
	echo Generated Doxygen documentation in $(DOCSDIR)

img/logo.png:
	$(CMDECHO)mkdir -p img && \
		wget http://www.gpusph.org/img/logo.png -O img/logo.png

# target: docsclean - Remove $(DOCSDIR)
docsclean:
	$(CMDECHO)rm -rf $(DOCSDIR)

# target: tags - Create TAGS file
tags: TAGS cscope.out
TAGS: $(ALLSRCFILES)
	$(CMDECHO)etags -R -h=.h.cuh.inc --langmap=c++:.cc.cuh.cu.def.h.inc src/ options/ scripts/
cscope.out: $(ALLSRCFILES)
	$(CMDECHO)which cscope > /dev/null && cscope -b $(INCPATH) -R -ssrc/ -soptions/ || touch cscope.out


# target: test - Run GPUSPH with WaveTank. Compile it if needed
test: all
	$(CMDECHO)$(TARGET)
	@echo Do "$(SCRIPTSDIR)/rmtests" to remove all tests

# target: compile-problems - Test that all problems compile
compile-problems:
	$(CMDECHO)pn=1 ; for prob in $(PROBLEM_LIST) ; do \
		echo [TEST-BUILD $${pn}/$(words $(PROBLEM_LIST))] $${prob} ; \
		$(MAKE) problem=$${prob} || exit 1 ; pn=$$(($$pn+1)) ; \
		done

# target: <problem_name> - Compile the given problem
$(PROBLEM_LIST):
	$(CMDECHO)$(MAKE) problem=$@

# target: list-problems - List available problems
list-problems:
	$(CMDECHO)echo $(PROBLEM_LIST) | sed 's/ /\n/g' | sort

# target: help - Display help
help:
	@echo "Type"
	@echo
	@echo "  $$ make help-targets"
	@echo
	@echo "for a description of all callable targets,"
	@echo
	@echo "  $$ make help-options"
	@echo
	@echo "for a description of available options."
	@echo
	@echo "  $$ make help-overrides"
	@echo
	@echo "for a description of possible options."

# target: help-targets - Display callable targets
help-targets:
	@echo "Available targets:"
	@grep -e "^# target:" $(MAKEFILE) | sed 's/^# target: /    /' # | sed 's/ - /\t/'

# target: help-options - Display available options
help-options:
	@echo "Options:"
	@grep -e "^# option:" $(MAKEFILE) | sed 's/^# option: /    /' # | sed 's/ - /\t/'
	@echo "(usage: make option=value)"

# target: help-overrides - Document available overrides
help-overrides:
	@echo "Default settings for these can be overridden/extended"
	@echo "by creating a Makefile.local that sets them:"
	@grep -e "^# override:" $(MAKEFILE) | sed 's/^# override: /    /' # | sed 's/ - /\t/'


# "sinclude" instead of "include" tells make not to print errors if files are missing.
# This is necessary because during the first processing of the makefile, make complains
# before creating them.
sinclude $(GPUDEPS)
sinclude $(CPUDEPS)

