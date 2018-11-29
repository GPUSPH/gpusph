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


# need for some substitutions
comma:=,
empty:=
space:=$(empty) $(empty)

# When running a `make clean`, we do not want to generate the files that we are going to remove
# anyway. Let us do an early check on this case, by checking if MAKECMDGOALS is one of the clean targets.
# We only do this if a single target was specified
cleaning:=0
ifeq ($(words $(MAKECMDGOALS)),1)
ifeq ($(findstring clean,$(MAKECMDGOALS)),clean)
	cleaning:=1
endif
endif

# Cached configuration. All settings that should be persistent across compilation
# (until changed) should be stored here.
ifneq ($(cleaning),1)
sinclude Makefile.conf
endif

# Include, if present, a local Makefile.
# This can be used by the user to set additional include paths (see Makefile.local.template file)
# (INCPATH = ....)
# library search paths
# (LIBPATH = ...)
# libaries
# (LIBS = ...)
# and general flags
# CPPFLAGS, CXXFLAGS, CUFLAGS, LDFLAGS,
sinclude Makefile.local

# GPUSPH version
GPUSPH_VERSION=$(shell git describe --tags --dirty=+custom 2> /dev/null | sed -e 's/-\([0-9]\+\)/+\1/' -e 's/-g/-/' 2> /dev/null)

ifeq ($(GPUSPH_VERSION), $(empty))
$(warning Unable to determine GPUSPH version)
GPUSPH_VERSION=unknown-version
endif
export GPUSPH_VERSION

# Git info output
GIT_INFO_OUTPUT=$(shell git branch -vv)

# system information
platform=$(shell uname -s 2>/dev/null)
platform_lcase=$(shell uname -s 2>/dev/null | tr '[:upper:]' '[:lower:]')
arch=$(shell uname -m)
# check if running on the Windows Subsystem for Linux
wsl=$(shell uname -r 2>/dev/null | grep Microsoft > /dev/null ; echo $$((1 - $$?)))

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

# on Windows, executables have .exe extension
ifeq ($(wsl), 1)
	EXE_SFX=.exe
else
	EXE_SFX=
endif

# directories: binary, objects, sources, expanded sources
DISTDIR = dist/$(platform_lcase)/$(arch)
DEPDIR = dep
OBJDIR = build
SRCDIR = src
EXPDIR = $(SRCDIR)/expanded
SCRIPTSDIR = scripts
DOCSDIR = docs
OPTSDIR = options
INFODIR = info

# target binary
exename = $(1)$(TARGET_SFX)$(EXE_SFX)
exe = $(DISTDIR)/$(call exename,$1)

dbgexename = $(1)$(DBG_SFX)$(EXE_SFX)
dbgexe = $(DISTDIR)/$(call dbgexename,$1)
nodbgexename = $(1)$(DBG_SFX)$(EXE_SFX)
nodbgexe = $(DISTDIR)/$(call nodbgexename,$1)

# binary to list compute capabilities of installed devices
LIST_CUDA_CC=$(SCRIPTSDIR)/list-cuda-cc$(EXE_SFX)
# binary to check for availability of CUDA installation
NVCC_EXE=nvcc$(EXE_SFX)

# --------------- File lists

# all files under $(SRCDIR), needed by tags files
ALLSRCFILES = $(shell find $(SRCDIR) -type f)

# .cc source files (CPU)
MPICXXFILES = $(SRCDIR)/NetworkManager.cc
ifeq ($(USE_HDF5),2)
	MPICXXFILES += $(SRCDIR)/HDF5SphReader.cc
	MPICXXFILES += $(SRCDIR)/HDF5SphWriter.cc
endif

OUR_PROBLEM_DIRS=$(SRCDIR)/problems
USER_PROBLEM_DIRS=$(SRCDIR)/problems/user
PROBLEM_DIRS=$(OUR_PROBLEM_DIRS) $(USER_PROBLEM_DIRS)

SRCSUBS=$(sort $(filter-out $(PROBLEM_DIRS),\
	$(filter %/,$(wildcard $(SRCDIR)/*/))))
SRCSUBS:=$(SRCSUBS:/=)
OBJSUBS=$(sort $(patsubst $(SRCDIR)/%,$(OBJDIR)/%,$(SRCSUBS) $(PROBLEM_DIRS)))
DEPSUBS=$(sort $(patsubst $(SRCDIR)/%,$(DEPDIR)/%,$(SRCSUBS) $(PROBLEM_DIRS)))

# list of problems
PROBLEM_LIST = $(filter-out GenericProblem, \
	$(foreach adir, $(PROBLEM_DIRS), \
	$(notdir $(basename $(wildcard $(adir)/*.h)))))

# list of problem executables, both debug and non-debug versions, for the clean target
PROBLEM_EXES = $(foreach p, $(PROBLEM_LIST),$(call dbgexe,$p) $(CURDIR)/$(call dbgexename,$p)) \
	       $(foreach p, $(PROBLEM_LIST),$(call nodbgexe,$p) $(CURDIR)/$(call nodbgexename,$p))

# use $(call problem_src,someproblem) to get the sources
# needed for that problem
problem_src = $(foreach adir, $(PROBLEM_DIRS), $(filter \
		$(adir)/$(1).cu, \
		$(wildcard $(adir)/*)))
problem_obj = $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(call problem_src,$1))
problem_gen = $(OPTSDIR)/$(1).gen.cc
problem_gen_obj = $(OBJDIR)/$(1).gen.o

problem_objs = $(call problem_obj,$1) $(call problem_gen_obj,$1)

# list of .cc files, exclusing MPI and problem sources
CCFILES = $(filter-out $(MPICXXFILES),\
	  $(foreach adir, $(SRCDIR) $(SRCSUBS),\
	  $(wildcard $(adir)/*.cc)))

# list of .cu files: we only compile problems directly, all other
# CUDA files are included via the cudasimframework
CUFILES = $(foreach p,$(PROBLEM_LIST),$(call problem_src,$p))

# dependency files via filename replacement
CCDEPS = $(patsubst $(SRCDIR)/%.cc,$(DEPDIR)/%.d,$(CCFILES) $(MPICXXFILES))
CUDEPS = $(patsubst $(SRCDIR)/%.cu,$(DEPDIR)/%.d,$(CUFILES))
GENDEPS = $(foreach p,$(PROBLEM_LIST),$(DEPDIR)/$(p).gen.d)

# headers
HEADERS = $(foreach adir, $(SRCDIR) $(SRCSUBS),$(wildcard $(adir)/*.h))

# object files via filename replacement
MPICXXOBJS = $(patsubst %.cc,$(OBJDIR)/%.o,$(notdir $(MPICXXFILES)))
CCOBJS = $(patsubst $(SRCDIR)/%.cc,$(OBJDIR)/%.o,$(CCFILES))
CUOBJS = $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(CUFILES))
GENOBJS = $(foreach p,$(PROBLEM_LIST),$(call problem_gen_obj,$p))

OBJS = $(CCOBJS) $(MPICXXOBJS)

# data files needed by some problems
EXTRA_PROBLEM_FILES ?=
# TestTopo uses this DEM:
EXTRA_PROBLEM_FILES += half_wave0.1m.txt

# --------------- Locate and set up compilers and flags

# flag to determine the compiler version
CXX_VERSION_FLAG= --version | head -1

ifeq ($(wsl),1)
	CXX=cl.exe
	CXX_VERSION_FLAG= 2>&1 > /dev/null | grep C/C++
endif

# override: CUDA_INSTALL_PATH - where CUDA is installed
# override:                     defaults to nvcc path, if found in PATH
#                               falls back to /usr/local/cuda otherwise
# override:                     validity is checked by looking for bin/nvcc under it,
# override:                     /usr is always tried as a last resort
CUDA_INSTALL_PATH ?= $(shell dirname "$$(dirname "$$(which $(NVCC_EXE))")")

ifeq ($(CUDA_INSTALL_PATH),$(empty))
	CUDA_INSTALL_PATH = /usr/local/cuda
endif

CUDA_INSTALL_PATH:=$(subst $(space),\ ,$(CUDA_INSTALL_PATH))

# We check the validity of the path by looking for bin/nvcc under it.
# if not found, we look into /usr, and finally abort
ifeq ($(wildcard $(CUDA_INSTALL_PATH)/bin/$(NVCC_EXE)),)
	CUDA_INSTALL_PATH = /usr
	# check again
	ifeq ($(wildcard $(CUDA_INSTALL_PATH)/bin/$(NVCC_EXE)),)
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
NVCC=$(CUDA_INSTALL_PATH)/bin/$(NVCC_EXE)
NVCC_VER=$(shell $(NVCC) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
versions_tmp  := $(subst ., ,$(NVCC_VER))
CUDA_MAJOR := $(firstword  $(versions_tmp))
CUDA_MINOR := $(lastword  $(versions_tmp))

# We only support CUDA 7 onwards, error out if this is an earlier version
# NOTE: the test is reversed because test returns 0 for true (shell-like)
OLD_CUDA=$(shell test $(CUDA_MAJOR) -ge 7; echo $$?)

ifeq ($(OLD_CUDA),1)
$(error CUDA version too old)
endif

# Make sure nvcc uses the same host compile that we use for the host
# code.
# Note that this requires the compiler to be supported by nvcc.
NVCC += -ccbin=$(CXX)

# Get the include path(s) used by default by our compiler
CXX_SYSTEM_INCLUDE_PATH=$(abspath $(shell echo | $(CXX) -x c++ -E -Wp,-v - 2>&1 | grep '^ ' | grep -v ' (framework directory)'))

# files to store last compile options: dbg, compute, fastmath, MPI usage, Chrono, linearization preference, Catalyst
DBG_SELECT_OPTFILE=$(OPTSDIR)/dbg_select.opt
COMPUTE_SELECT_OPTFILE=$(OPTSDIR)/compute_select.opt
FASTMATH_SELECT_OPTFILE=$(OPTSDIR)/fastmath_select.opt
MPI_SELECT_OPTFILE=$(OPTSDIR)/mpi_select.opt
HDF5_SELECT_OPTFILE=$(OPTSDIR)/hdf5_select.opt
CHRONO_SELECT_OPTFILE=$(OPTSDIR)/chrono_select.opt
LINEARIZATION_SELECT_OPTFILE=$(OPTSDIR)/linearization_select.opt
CATALYST_SELECT_OPTFILE=$(OPTSDIR)/catalyst_select.opt

# Autogenerated files
AUTOGEN_SRC=$(SRCDIR)/parse-debugflags.h $(SRCDIR)/describe-debugflags.h

# these are not really options, but they follow the same mechanism
GPUSPH_VERSION_OPTFILE=$(OPTSDIR)/gpusph_version.opt
MAKE_SHOW_TXT=$(INFODIR)/show.txt
MAKE_SHOW_TMP=$(INFODIR)/show.tmp
MAKE_SHOW_OPTFILE=$(OPTSDIR)/make_show.opt
GIT_INFO_OPTFILE=$(OPTSDIR)/git_info.opt

# Optfile that influence the device code
DEVCODE_OPTFILES = \
	  $(DBG_SELECT_OPTFILE) \
	  $(COMPUTE_SELECT_OPTFILE) \
	  $(FASTMATH_SELECT_OPTFILE) \
	  $(LINEARIZATION_SELECT_OPTFILE) \

# Actual optfiles, that define specific options
ACTUAL_OPTFILES= \
	  $(DEVCODE_OPTFILES) \
	  $(MPI_SELECT_OPTFILE) \
	  $(HDF5_SELECT_OPTFILE) \
	  $(CHRONO_SELECT_OPTFILE) \
	  $(CATALYST_SELECT_OPTFILE)

# Pseudo-optfiles, documentig GPUSPH version and build environment
PSEUDO_OPTFILES= \
	  $(GPUSPH_VERSION_OPTFILE) \
	  $(MAKE_SHOW_OPTFILE) \
	  $(GIT_INFO_OPTFILE)

# Both of them
OPTFILES=$(ACTUAL_OPTFILES) $(PSEUDO_OPTFILES)

# Let make know that .opt and .i dependencies are to be looked for in $(OPTSDIR)
vpath %.opt $(OPTSDIR)

# update GPUSPH_VERSION_OPTFILE if git version changed
LAST_GPUSPH_VERSION=$(shell test -e $(GPUSPH_VERSION_OPTFILE) && \
	grep "\#define GPUSPH_VERSION" $(GPUSPH_VERSION_OPTFILE) | cut -f2 -d\")

ifneq ($(LAST_GPUSPH_VERSION),$(GPUSPH_VERSION))
	TMP:=$(shell test -e $(GPUSPH_VERSION_OPTFILE) && \
		$(SED_COMMAND) 's/$(LAST_GPUSPH_VERSION)/$(GPUSPH_VERSION)/' $(GPUSPH_VERSION_OPTFILE) )
endif

# Selection of last built problem (rebuilt by `make` without
# further specification
LAST_BUILT_PROBLEM ?= DamBreak3D
REQUESTED_PROBLEMS = $(filter $(PROBLEM_LIST),$(MAKECMDGOALS))
CURRENT_LAST = $(lastword DamBreak3D $(LAST_BUILT_PROBLEM) $(REQUESTED_PROBLEMS))
ifneq ($(CURRENT_LAST),$(LAST_BUILT_PROBLEM))
	LAST_BUILT_PROBLEM:=$(CURRENT_LAST)
	TMP:=$(shell test -e Makefile.conf && \
		$(SED_COMMAND) '/LAST_BUILT_PROBLEM/c\LAST_BUILT_PROBLEM=$(LAST_BUILT_PROBLEM)' Makefile.conf)
endif

# A tricky dependency chain: if the user specified a problem to be built,
# we want to also rebuild GPUSPH to link to it, so $(LAST_BUILT_PROBLEM)
# must depend on GPUSPH. However, if the user did _not_ specify a problem,
# then we want GPUSPH to depend on $(LAST_BUILT_PROBLEM)
ifneq ($(REQUESTED_PROBLEMS),$(empty))
	# problem was specified, it should depend on GPUSPH
	PROBLEM_GPUSPH_DEP=GPUSPH
	GPUSPH_PROBLEM_DEP=
else
	PROBLEM_GPUSPH_DEP=
	GPUSPH_PROBLEM_DEP=$(LAST_BUILT_PROBLEM)
endif
#$(info => $(PROBLEM_GPUSPH_DEP) <= $(GPUSPH_PROBLEM_DEP))

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
		TMP := $(error MPI use requested, but no MPI compiler was found, aborting)
	else
		ifeq ($(USE_MPI),)
			TMP := $(info MPI compiler not found, multi-node will NOT be supported)
			USE_MPI = 0
		endif
	endif
else
	# autodetect the MPI version
	MPI_VERSION = $(shell printf '\#include <mpi.h>\nstandard/MPI_VERSION/MPI_SUBVERSION' | $(MPICXX) -E -x c - 2> /dev/null | grep '^standard/' | cut -d/ -f2- | tr '/' '.')

	# if we have USE_MPI, but MPI_VERSION is empty, abort because it means we couldn't successfully get the
	# MPI version from mpi.h (e.g. because the development headers were not installed)
	# Note that if MPI version is defined, it will have a dot inside, so $(USE_MPI)$(MPI_VERSION) cannot be
	# equal to the string 1 in that case, so the following condition is only for USE_MPI=1 and MPI_VERSION empty
	ifeq ($(USE_MPI)$(MPI_VERSION),1)
		TMP := $(error MPI use requested, but the MPI version could not be determined, aborting)
	endif

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
	# Also try to detect implementation-specific version.
	# OpenMPI exposes the version via individual numeric macros
	OMPI_VERSION = $(shell printf '\#include <mpi.h>\nversion/OMPI_MAJOR_VERSION/OMPI_MINOR_VERSION/OMPI_RELEASE_VERSION' | $(MPICXX) -E -x c - 2> /dev/null | grep '^version/' | grep -v 'OMPI_' | cut -d/ -f2- | tr '/' '.')
	# MPICH exposes a version string macro
	MPICH_VERSION = $(shell printf '\#include <mpi.h>\nversion/MPICH_VERSION' | $(MPICXX) -E -x c - 2> /dev/null | grep '^version/' | grep -v 'MPICH_' | cut -d\" -f2 )
	# MVAPICH2 exposes its own version string (MVAPICH 1.x apparently does not)
	MVAPICH_VERSION = $(shell printf '\#include <mpi.h>\nversion/MVAPICH2_VERSION' | $(MPICXX) -E -x c - 2> /dev/null | grep '^version/' | grep -v 'MVAPICH2_' | cut -d\" -f2 )

	MPI_VERSION :=$(MPI_VERSION)$(if $(OMPI_VERSION),$(space)(OpenMPI $(OMPI_VERSION)))
	MPI_VERSION :=$(MPI_VERSION)$(if $(MPICH_VERSION),$(space)(MPICH $(MPICH_VERSION)))
	MPI_VERSION :=$(MPI_VERSION)$(if $(MVAPICH_VERSION),$(space)(MVAPICH $(MVAPICH_VERSION)))

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

# option: chrono - 0 do not use Chrono (no floating objects support), 1 use Chrono (enable floating object support). Default: 0
ifdef chrono
	# does it differ from last?
	ifneq ($(USE_CHRONO),$(chrono))
		TMP := $(shell test -e $(CHRONO_SELECT_OPTFILE) && \
			$(SED_COMMAND) 's/$(USE_CHRONO)/$(chrono)/' $(CHRONO_SELECT_OPTFILE) )
		# user choice
		USE_CHRONO=$(chrono)
	endif
else
	USE_CHRONO ?= 0
endif

# option: linearization - something like xyz or yzx to indicate the order
# option:                 of coordinates when linearizing cell indices,
# option:                 from fastest to slowest growing coordinate
ifdef linearization
	ifneq ($(LINEARIZATION),$(linearization))
		LINEARIZATION=$(linearization)
		FORCE_MAKE_LINEARIZATION=FORCE
	endif
else
	ifndef LINEARIZATION
		FORCE_MAKE_LINEARIZATION=FORCE
		LINEARIZATION=yzx
	endif
endif
# split the linearization string into individual characters, space-separated
LINEARIZATION_WORDS=$(shell echo $(LINEARIZATION) | sed 's/./\0 /g')

# option: catalyst - 0 do not use Catalyst (disable co-processing visualization support), 1 use Catalyst (enable co-processing visualization support). Default: 0
ifdef catalyst
	# does it differ from last?
	ifneq ($(USE_CATALYST),$(catalyst))
		TMP := $(shell test -e $(CATALYST_SELECT_OPTFILE) && \
			$(SED_COMMAND) 's/$(USE_CATALYST)/$(catalyst)/' $(CATALYST_SELECT_OPTFILE) )
		# user choice
		USE_CATALYST=$(catalyst)
	endif
else
	USE_CATALYST ?= 0
endif

# If Catalyst is disabled - exclude adaptors and display writer from the list of .cc files
ifeq ($(USE_CATALYST),0)
	CCFILES := $(filter-out $(wildcard $(SRCDIR)/adaptors/*.cc),\
		$(filter-out $(SRCDIR)/writers/DisplayWriter.cc,\
		$(CCFILES)))
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
INCPATH += -I$(SRCDIR) \
	   $(foreach adir,$(SRCSUBS),-I$(adir)) \
	   $(foreach adir,$(PROBLEM_DIRS),-I$(adir)) \
	   -I$(OPTSDIR)

# access the CUDA include files from the C++ compiler too, but mark their path as a system include path
# so that they can be skipped when generating dependencies. This must only be done for the host compiler,
# because otherwise some nvcc version will complain about kernels not being allowed in system files
# while compiling some thrust functions.
# Note that we do this only if the include path is not already in the system
# include path. This is particularly important in the case where CUDA_INCLUDE_PATH
# is /usr/include, since otherwise GCC 6 (and later) will fail to find standard
# includes such as stdint.h
CUDA_INCLUDE_PATH = $(abspath $(CUDA_INSTALL_PATH)/include)
ifneq ($(CUDA_INCLUDE_PATH),$(filter $(CUDA_INCLUDE_PATH),$(CXX_SYSTEM_INCLUDE_PATH)))
	CC_INCPATH += -I $(CUDA_INCLUDE_PATH)
endif

# LIBPATH
LIBPATH += -L/usr/local/lib

# On Darwin, make sure we link with the GNU C++ standard library
# TODO make sure this is still needed
ifeq ($(platform), Darwin)
       LIBS += -lstdc++
endif


# CUDA libaries
LIBPATH += -L$(CUDA_INSTALL_PATH)/lib$(LIB_PATH_SFX)

# link to the CUDA runtime library
LIBS += -lcudart

ifneq ($(USE_HDF5),0)
	# link to HDF5 for input reading
	LIBS += $(HDF5_LD)
endif

ifeq ($(USE_CATALYST),1)
	# link to Catalyst
	LIBS += $(CATALYST_LD)
endif

# pthread needed for the UDP writer
LIBS += -lpthread

# Realtime Extensions library (for clock_gettime) (not on Mac)
ifneq ($(platform), Darwin)
	LIBS += -lrt
endif

# override: CHRONO_PATH         - where Chrono is installed
# override:                       defaults to /usr/local/, may be set to
# override:                       the build directory of Chrono
CHRONO_PATH ?= /usr/local
# override: CHRONO_INCLUDE_PATH - where Chrono include are installed
# override:                       if unset, auto-detection will be attempted
# override:                       from $(CHRONO_PATH) adding either include/ or src/
# override: CHRONO_LIB_PATH     - where Chrono lib is installed
# override:                       if unset, auto-detection will be attempted
# override:                       from $(CHRONO_PATH) adding lib64/

CHRONO_INCLUDE_PATH ?=
CHRONO_LIB_PATH ?=

ifneq ($(USE_CHRONO),0)
	ifeq ($(CHRONO_INCLUDE_PATH),$(empty))
		# If CHRONO_INCLUDE_PATH is not set, look for chrono/core/ChChrono.h
		# under $(CHRONO_PATH)/include and $(CHRONO_PATH)/src in sequence,
		# and then build the include path by getting the up-up-up-dir
		CHRONO_INCLUDE_PATH := $(realpath $(dir $(or \
			$(wildcard $(CHRONO_PATH)/include/chrono/core/ChChrono.h), \
			$(wildcard $(CHRONO_PATH)/src/chrono/core/ChChrono.h), \
			$(error Could not find Chrono include files, please set CHRONO_PATH or CHRONO_INCLUDE_PATH) \
		))/../../)
	else
		# otherwise, check that the user-specified path is correct
		ifeq ($(wildcard $(CHRONO_INCLUDE_PATH)/chrono/core/ChChrono.h),$(empty))
			TMP := $(error CHRONO_INCLUDE_PATH is set incorrectly, chrono/core/ChChrono.h not found)
		endif
	endif

	ifeq ($(CHRONO_LIB_PATH),$(empty))
		# If CHRONO_LIB_PATH is not set, look for libChronoEngine.*
		# under $(CHRONO_PATH)/lib64 and then build the include path by getting the up-dir
		CHRONO_LIB_PATH := $(dir $(or \
			$(wildcard $(CHRONO_PATH)/lib64/libChronoEngine.so) \
			$(wildcard $(CHRONO_PATH)/build/lib64/libChronoEngine.so), \
			$(error Could not find Chrono include files, please set CHRONO_PATH or CHRONO_LIB_PATH) \
			))
	else
		# otherwise, check that the user-specified path is correct
		ifeq ($(wildcard $(CHRONO_LIB_PATH)/libChronoEngine.*),$(empty))
			TMP := $(error CHRONO_LIB_PATH is set incorrectly, libChronoEngine not found)
		endif
	endif

	# When using Chrono from the build directory, chrono/ChConfig.h is under the build dir,
	# otherwise it's under the include directory
	CHRONO_CONFIG_PATH=$(realpath $(dir $(or \
	   $(wildcard $(CHRONO_INCLUDE_PATH)/chrono/ChConfig.h), \
	   $(wildcard $(CHRONO_PATH)/build/chrono/ChConfig.h), \
	   $(error Could not find Chrono configuration header. Include path: $(CHRONO_INCLUDE_PATH), Chrono path: $(CHRONO_PATH)) \
	))/../)

	ifneq ($(CHRONO_CONFIG_PATH),/usr/include)
		INCPATH += -isystem $(CHRONO_CONFIG_PATH)
	endif

	ifneq ($(CHRONO_INCLUDE_PATH),/usr/include)
		INCPATH += -isystem $(CHRONO_INCLUDE_PATH)
	endif

	# This is needed because on some versions of Chrono, headers include each other without the chrono/ prefix 8-/
	INCPATH += -isystem $(CHRONO_INCLUDE_PATH)/chrono

	INCPATH += -isystem $(CHRONO_INCLUDE_PATH)/chrono/collision/bullet

	LIBPATH += -L$(CHRONO_LIB_PATH)
	LIBS += -lChronoEngine
	LDFLAGS += --linker-options -rpath,$(CHRONO_LIB_PATH)
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
# Likewise, for some reasons some versions g++ (such as g++-5 on Ubuntu)
# don't include functions such as isnan under std when including <cmath>
CPPFLAGS += -D_GLIBCXX_USE_C99_MATH

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


# CXXFLAGS start with the target architecture
CXXFLAGS += $(TARGET_ARCH)

# We also force C++11 mode, since we are no relying on C++11 features
# TODO Check if any -std is present in CXXFLAGS (added by the user) and if
# the specified value is not 11, warn before removing it
CXXFLAGS += -std=c++11

# HDF5 might require specific flags
ifneq ($(USE_HDF5),0)
	CXXFLAGS += $(HDF5_CXX)
endif

# Catalyst might require specific flags
ifeq ($(USE_CATALYST),1)
	CXXFLAGS += $(CATALYST_CXX)
endif

# nvcc-specific flags

# compute capability specification, if defined
ifneq ($(COMPUTE),)
	CUFLAGS += -arch=sm_$(COMPUTE)
	LDFLAGS += -arch=sm_$(COMPUTE)
endif

# generate line info
# TODO this should only be done in debug mode
CUFLAGS += --generate-line-info

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

# Finally, add CXXFLAGS to CUFLAGS, except for -std, which gets moved outside

CUFLAGS += $(filter -std=%,$(CXXFLAGS)) --compiler-options \
	   $(subst $(space),$(comma),$(strip $(filter-out -std=%,$(CXXFLAGS))))

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

# Snapshot date: date of last commit (if possible), or current date
snap_date := $(shell git log -1 --format='%cd' --date=iso 2> /dev/null | cut -f1,4 -d' ' | tr ' ' '-' || date +%Y-%m-%d)
# snapshot tarball filename
SNAPSHOT_FILE = ./GPUSPH-$(GPUSPH_VERSION)-$(snap_date).tgz

# option: plain - 0 fancy line-recycling stage announce, 1 plain multi-line stage announce
ifeq ($(plain), 1)
	show_stage=@printf "[$(1)] $(2)\n"
	show_stage_nl=$(show_stage)
else
	show_stage   =@printf "\r                                                                  \r[$(1)] $(2)"
	show_stage_nl=@printf "\r                                                                  \r[$(1)] $(2)\n"
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
export CMDECHO

.PHONY: all run showobjs show snapshot expand deps docs test help
.PHONY: clean cpuclean gpuclean cookiesclean computeclean docsclean confclean genclean depsclean
.PHONY: dev-guide user-guide
.PHONY: FORCE

# target: GPUSPH - A symlink to the last built problem, or the default problem (DamBreak3D)
GPUSPH: $(GPUSPH_PROBLEM_DEP) Makefile.conf
	$(call show_stage_nl,SYM,$@)
	$(CMDECHO)ln -sf $(LAST_BUILT_PROBLEM) $(CURDIR)/$@

# Support for legacy/classic 'all' target
all: GPUSPH

# For each problem, we define the following target chain:
# * the PROBLEM.gen.cc generator, from the template and the optsdir
# * the binary in dist, from all the object files
# * the symlink in the root directory, from the binary in dist
#   this is a FORCEd target, so symlinking always happens, even when
#   the target is fresh
define problem_deps
$(call problem_gen,$1): $(SRCDIR)/problem_gen.tpl | $(OPTSDIR)
	$(call show_stage,GEN,$$(@F))
	$(CMDECHO)sed -e 's/PROBLEM/$1/g' $$< > $$@
$(call exe,$1): $(call problem_objs,$1) $(OBJS) | $(DISTDIR)
	$(call show_stage_nl,LINK,$$@)
	$(CMDECHO)$(LINKER) -o $$@ $$^ $(LDFLAGS) $(LDLIBS)
$1: $(call exe,$1)
	@echo
	@echo "Compiled with problem $$@"
	@[ $(FASTMATH) -eq 1 ] && echo "Compiled with fastmath" || echo "Compiled without fastmath"
	$(call show_stage_nl,SYM,$$@)
	$(CMDECHO)ln -sf $$< $(CURDIR)/$(call exename,$$@) && echo "Success"
endef

$(LAST_BUILT_PROBLEM): $(PROBLEM_GPUSPH_DEP)

# target: <problem_name> - Compile the given problem
$(foreach p,$(PROBLEM_LIST),$(eval $(call problem_deps,$p)))

# internal targets to (re)create the "selected option headers" if they're missing
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
$(CHRONO_SELECT_OPTFILE): | $(OPTSDIR)
	@echo "/* Determines if Chrono is enabled. */" \
		> $@
	@echo "#define USE_CHRONO $(USE_CHRONO)" >> $@
$(LINEARIZATION_SELECT_OPTFILE): $(FORCE_MAKE_LINEARIZATION) | $(OPTSDIR)
	@echo "/* Linearization order */" > $@
	@echo "#define LINEARIZATION \"$(LINEARIZATION)\"" >> $@
	@echo "#define COORD1 $(word 1, $(LINEARIZATION_WORDS))" >> $@
	@echo "#define COORD2 $(word 2, $(LINEARIZATION_WORDS))" >> $@
	@echo "#define COORD3 $(word 3, $(LINEARIZATION_WORDS))" >> $@

$(GPUSPH_VERSION_OPTFILE): | $(OPTSDIR)
	@echo "/* git version of GPUSPH. */" \
		> $@
	@echo "#define GPUSPH_VERSION \"$(GPUSPH_VERSION)\"" >> $@

$(CATALYST_SELECT_OPTFILE): | $(OPTSDIR)
	@echo "/* Determines if we are using Catalyst or not. */" \
		> $@
	@echo "#define USE_CATALYST $(USE_CATALYST)" >> $@

# TODO proper escaping for special characters in the GIT_INFO_OUTPUT
$(GIT_INFO_OPTFILE): | $(OPTSDIR)
	@echo "/* git branch --v. */" > $@
	@echo -n "#define GIT_INFO_OUTPUT \"" >> $@
	@echo -n "$(GIT_INFO_OUTPUT)" >> $@
	@echo "\"" >> $@

# TODO proper escaping for special characters in the MAKE_SHOW_TXT
# Presently we only handle EOL and double-quotes
$(MAKE_SHOW_OPTFILE): $(MAKE_SHOW_TXT) | $(OPTSDIR)
	@echo "/* make show of GPUSPH. */" > $@
	@echo -n "#define MAKE_SHOW_OUTPUT \"" >> $@
	@sed -e 's/"/\\"/' -e 's/$$/\\n\\/' $(MAKE_SHOW_TXT) >> $@
	@echo "\"" >> $@

$(OBJS): $(DBG_SELECT_OPTFILE)

# Autogenerated files
$(SRCDIR)/parse-debugflags.h: $(SCRIPTSDIR)/parse-debugflags.awk $(SRCDIR)/debugflags.def
	$(CMDECHO)awk -f $^ > $@

$(SRCDIR)/describe-debugflags.h: $(SCRIPTSDIR)/describe-debugflags.awk $(SRCDIR)/debugflags.def
	$(CMDECHO)awk -f $^ > $@

# compile CPU objects and generate dependencies
# we use here a trick discussed at
# http://make.mad-scientist.net/papers/advanced-auto-dependency-generation/#depdelete
# The idea is that to avoid restarting make when the .d files change,
# and to avoid issues with deleted dependency files, the .d files themselves
# have an empty rule, and they are generated when the object files are generated.
# However, we do not use the "generate dependencies while compiling” feature of GCC
# The -MM flag is used to not include system includes.
# The -MG flag is used to add missing includes (useful to depend on the .opt files).
# The -MT flag is used to define the object file.
$(CCOBJS): $(OBJDIR)/%.o: $(SRCDIR)/%.cc $(DEPDIR)/%.d | $(OBJSUBS)
	$(call show_stage,CC,$(@F))
	$(CMDECHO)$(CXX) $(CC_INCPATH) $(CPPFLAGS) $(CXXFLAGS) -MG -MM -MT $@ $< > $(word 2,$^)
	$(CMDECHO)$(CXX) $(CC_INCPATH) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<
$(GENOBJS): $(OBJDIR)/%.gen.o: $(OPTSDIR)/%.gen.cc $(DEPDIR)/%.gen.d | $(OBJSUBS)
	$(call show_stage,CC,$(@F))
	$(CMDECHO)$(CXX) $(CC_INCPATH) $(CPPFLAGS) $(CXXFLAGS) -MG -MM -MT $@ $< > $(word 2,$^)
	$(CMDECHO)$(CXX) $(CC_INCPATH) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<
$(MPICXXOBJS): $(OBJDIR)/%.o: $(SRCDIR)/%.cc $(DEPDIR)/%.d | $(OBJSUBS)
	$(call show_stage,MPI,$(@F))
	$(CMDECHO)OMPI_CXX=$(CXX) MPICH_CXX=$(CXX) \
		$(MPICXX) $(CC_INCPATH) $(CPPFLAGS) $(CXXFLAGS) -MG -MM -MT $@ $< > $(word 2,$^)
	$(CMDECHO)OMPI_CXX=$(CXX) MPICH_CXX=$(CXX) \
		$(MPICXX) $(CC_INCPATH) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

# compile GPU objects
$(CUOBJS): $(OBJDIR)/%.o: $(SRCDIR)/%.cu $(DEPDIR)/%.d $(DEVCODE_OPTFILES) | $(OBJSUBS)
	$(call show_stage,CU,$(@F))
	$(CMDECHO)$(NVCC) $(CPPFLAGS) $(CUFLAGS) -E $< \
		 --compiler-options -MG,-MM,-MT,$@ > $(word 2,$^)
	$(CMDECHO)$(NVCC) $(CPPFLAGS) $(CUFLAGS) -c -o $@ $<

# deps: empty rule, but require the directories and optfiles to be present
$(CCDEPS): | $(DEPSUBS) $(OPTFILES) $(AUTOGEN_SRC) ;
$(DEPDIR)/%.gen.d: | $(DEPSUBS) $(OPTFILES) ;
$(CUDEPS): | $(DEPSUBS) $(OPTFILES) ;

# compile program to list compute capabilities of installed devices.
# Filter out all architecture specification flags (-arch=sm_*), since they
# can cause the compiler to error out when the architecture is not supported
# (for example too recent architectures on older compilers, or obsolete architectures
# not supported in the most recent version of the SDK)
$(LIST_CUDA_CC): $(LIST_CUDA_CC).cc
	$(call show_stage,SCRIPTS,$(@F))
	$(CMDECHO)$(NVCC) $(CPPFLAGS) -Wno-deprecated-gpu-targets $(filter-out -arch=sm_%,$(filter-out --ptxas-options=%,$(filter-out --generate-line-info,$(CUFLAGS)))) -o $@ $< $(filter-out -arch=sm_%,$(LDFLAGS))

# create distdir
$(DISTDIR):
	$(CMDECHO)mkdir -p $(DISTDIR)

# create objdir and subs
$(OBJDIR) $(OBJSUBS):
	$(CMDECHO)mkdir -p $(OBJDIR) $(OBJSUBS)

# create optsdir
$(OPTSDIR):
	$(CMDECHO)mkdir -p $(OPTSDIR)

# create infodir
$(INFODIR):
	$(CMDECHO)mkdir -p $(INFODIR)

# target: clean - Clean everything but last compile choices
# clean: cpuobjs, gpuobjs, deps makefiles, targets, target symlinks
clean: genclean depsclean
	$(CMDECHO)$(RM) -f $(PROBLEM_EXES) GPUSPH
	$(CMDECHO)find $(CURDIR) -maxdepth 1 -lname $(DISTDIR)/\* -delete

# target: cpuclean - Clean CPU stuff
cpuclean:
	$(RM) $(CCOBJS) $(MPICXXOBJS) $(CCDEPS)

# target: gpuclean - Clean GPU stuff
gpuclean: computeclean
	$(RM) $(CUOBJS) $(CUDEPS)

# target: computeclean - Clean compute capability selection stuff
computeclean:
	$(RM) $(LIST_CUDA_CC) $(COMPUTE_SELECT_OPTFILE)
	if [ -e Makefile.conf ] ; then $(SED_COMMAND) '/COMPUTE=/d' Makefile.conf ; fi

# target: cookiesclean - Clean last dbg, problem, compute and fastmath choices,
# target:                forcing .*_select.opt files to be regenerated (use if they're messed up)
cookiesclean:
	$(RM) -r $(OPTFILES) $(OPTSDIR) $(INFODIR)

# target: genclean - Clean all problem generators
genclean:
	$(RM) $(OBJDIR)/*.gen.o $(OPTSDIR)/*.gen.cc

# target: confclean - Clean all configuration options: like cookiesclean, but also purges Makefile.conf
confclean: cookiesclean
	$(RM) -f Makefile.conf

# target: depsclean - Clean all dependencies
depsclean:
	$(RM) -rf $(DEPDIR)

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
show: $(MAKE_SHOW_TXT)
	@cat $(MAKE_SHOW_TXT)

# To avoid infinite recursions in dependencies, we create the show output
# as a separate temporary file, and overwrite the current one if different
$(MAKE_SHOW_TXT): $(MAKE_SHOW_TMP)
	@cmp -s $< $@ 2> /dev/null || cp $< $@ ; $(RM) $<

$(MAKE_SHOW_TMP): Makefile Makefile.conf $(filter Makefile.local,$(MAKEFILE_LIST)) FORCE | $(INFODIR)
	$(call show_stage,CONF,make show)
	@echo "GPUSPH version:  $(GPUSPH_VERSION)"							 > $@
	@echo "Platform:        $(platform)"								>> $@
	@echo "WSL:             $(wsl)"										>> $@
	@echo "Architecture:    $(arch)"									>> $@
	@echo "Current dir:     $(CURDIR)"									>> $@
	@echo "This Makefile:   $(MAKEFILE)"								>> $@
	@echo "Used Makefiles:  $(MAKEFILE_LIST)"							>> $@
	@echo "Problem:         $(PROBLEM)"									>> $@
	@echo "Linearization:   $(LINEARIZATION)"							>> $@
#	@echo "   last:         $(LAST_PROBLEM)"							>> $@
	@echo "Snapshot file:   $(SNAPSHOT_FILE)"							>> $@
	@echo "Last problem:    $(LAST_BUILT_PROBLEM)"						>> $@
	@echo "Sources dir:     $(SRCDIR) $(SRCSUBS)"						>> $@
	@echo "Options dir:     $(OPTSDIR)"									>> $@
	@echo "Objects dir:     $(OBJDIR) $(OBJSUBS)"						>> $@
	@echo "Scripts dir:     $(SCRIPTSDIR)"								>> $@
	@echo "Docs dir:        $(DOCSDIR)"									>> $@
	@echo "Doxygen conf:    $(DOXYCONF)"								>> $@
	@echo "Verbose:         $(verbose)"									>> $@
	@echo "Debug:           $(DBG)"										>> $@
	@echo "CXX:             $(CXX)"										>> $@
	@echo "CXX version:     $(shell $(CXX) $(CXX_VERSION_FLAG))"		>> $@
	@echo "MPICXX:          $(MPICXX)"									>> $@
	@echo "nvcc:            $(NVCC)"									>> $@
	@echo "nvcc version:    $(NVCC_VER)"								>> $@
	@echo "LINKER:          $(LINKER)"									>> $@
	@echo "Compute cap.:    $(COMPUTE)"									>> $@
	@echo "Fastmath:        $(FASTMATH)"								>> $@
	@echo "USE_MPI:         $(USE_MPI)"									>> $@
	@[ 1 = $(USE_MPI) ] && echo "    MPI version: $(MPI_VERSION)"					>> $@ || true
	@echo "USE_HDF5:        $(USE_HDF5)"								>> $@
	@echo "USE_CHRONO:      $(USE_CHRONO)"								>> $@
	@echo "default paths:   $(CXX_SYSTEM_INCLUDE_PATH)"					>> $@
	@echo "INCPATH:         $(INCPATH)"									>> $@
	@echo "LIBPATH:         $(LIBPATH)"									>> $@
	@echo "LIBS:            $(LIBS)"									>> $@
	@echo "LDFLAGS:         $(LDFLAGS)"									>> $@
	@echo "CPPFLAGS:        $(CPPFLAGS)"								>> $@
	@echo "CXXFLAGS:        $(CXXFLAGS)"								>> $@
	@echo "CUFLAGS:         $(CUFLAGS)"									>> $@
#	@echo "Suffixes:        $(SUFFIXES)"								>> $@

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
deps: $(CCDEPS) $(CUDEPS) ;

$(DEPDIR) $(DEPSUBS):
	$(CMDECHO)mkdir -p $@

# We want all of the OPTFILES to be built before anything else, which we achieve by
# making Makefile.conf depend on them.
Makefile.conf: Makefile $(ACTUAL_OPTFILES)
	$(call show_stage,CONF,$@)
	$(CMDECHO)# Create Makefile.conf with standard disclaimer
	$(CMDECHO)echo '# This is an autogenerated configuration file. Please DO NOT edit it manually' > $@
	$(CMDECHO)echo '# Run make with the appropriate option to change a configured value' >> $@
	$(CMDECHO)echo '# Use `make help-options` to see a list of available options' >> $@
	$(CMDECHO)echo '# Use `make confclean` to reset your configuration' >> $@
	$(CMDECHO)# recover value of LAST_BUILT_PROBLEM
	$(CMDECHO)echo 'LAST_BUILT_PROBLEM=$(LAST_BUILT_PROBLEM)' >> $@
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
	$(CMDECHO)# recover value of USE_CHRONO from OPTFILES
	$(CMDECHO)grep "\#define USE_CHRONO" $(CHRONO_SELECT_OPTFILE) | cut -f2-3 -d ' ' | tr ' ' '=' >> $@
	$(CMDECHO)# recover value of LINEARIZATION from OPTFILES
	$(CMDECHO)grep "\#define LINEARIZATION" $(LINEARIZATION_SELECT_OPTFILE) | cut -f2-3 -d ' ' | tr ' ' '=' | tr -d '"'>> $@
	$(CMDECHO)# recover value of USE_CATALYST from OPTFILES
	$(CMDECHO)grep "\#define USE_CATALYST" $(CATALYST_SELECT_OPTFILE) | cut -f2-3 -d ' ' | tr ' ' '=' >> $@

# TODO docs should also build the user-guide, but since we don't ship images
# this can't be normally done, so let's not include this for the time being.
#
# target: docs - Generate the developers' guide
docs: dev-guide

# target: dev-guide - Generate the developers' guide
dev-guide:
	@echo "Generating developer documentation..."
	$(CMDECHO) $(MAKE) -C $(DOCSDIR)/$@

# target: user-guide - Generate the user's manuals
user-guide:
	@echo "Generating user documentation..."
	$(CMDECHO) $(MAKE) -C $(DOCSDIR)/$@

# target: docsclean - Remove $(DOCSDIR)
docsclean:
	$(CMDECHO)rm -rf $(DOCSDIR)/dev-guide/html/ $(DOCSDIR)/user-guide/gpusph-manual.pdf

# target: tags - Create TAGS file
tags: TAGS cscope.out
TAGS: $(ALLSRCFILES)
	$(CMDECHO)ctags-exuberant -R -h=.h.cuh.inc --langmap=c++:.cc.cuh.cu.def.h.inc src/ options/ scripts/
cscope.out: $(ALLSRCFILES)
	$(CMDECHO)which cscope > /dev/null && cscope -b -k $(patsubst %,-I%,$(CXX_SYSTEM_INCLUDE_PATH)) $(subst -isystem$(space),-I,$(INCPATH)) -R -ssrc/ -soptions/ || touch cscope.out


# target: test - Run GPUSPH with WaveTank. Compile it if needed
test: $(LAST_BUILT_PROBLEM)
	$(CMDECHO)$(CURDIR)/$(LAST_BUILT_PROBLEM)
	@echo Do "$(SCRIPTSDIR)/rmtests" to remove all tests

# target: compile-problems - Test that all problems compile
compile-problems: $(PROBLEM_LIST)

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

# FORCE target: add it to the dependecy f another target to force-rebuild it
# Used e.g. by the LINEARIZATION_SELECT_OPTFILE to remake it when the linearization
# changes (note that we use this mechanism instead of the sed commands used in
# other circumstances because it's more complex to rebuild
FORCE:

# "sinclude" instead of "include" tells make not to print errors if files are missing.
# This is necessary because during the first processing of the makefile, make complains
# before creating them.
ifneq ($(cleaning),1)
sinclude $(CCDEPS)
sinclude $(CUDEPS)
sinclude $(GENDEPS)
endif

