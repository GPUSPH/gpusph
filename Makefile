# GPUSPH Makefile
#
# TODO:
# - Improve list-problems problem detection (move problems to separate dir?)
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
DISTDIR = ./dist/$(platform_lcase)/$(arch)
OBJDIR = ./build
SRCDIR = ./src
EXPDIR = $(SRCDIR)/expanded
SCRIPTSDIR = ./scripts
DOCSDIR = ./docs
OPTSDIR = ./options

# target binary
TARGETNAME := GPUSPH$(TARGET_SFX)
TARGET := $(DISTDIR)/$(TARGETNAME)

# binary to list compute capabilities of installed devices
LIST_CUDA_CC=$(SCRIPTSDIR)/list-cuda-cc


# --------------- File lists

# makedepend will generate dependencies in these file
GPUDEPS = $(MAKEFILE).gpu
CPUDEPS = $(MAKEFILE).cpu

# .cc source files (CPU)
MPICXXFILES = $(SRCDIR)/NetworkManager.cc
CCFILES = $(filter-out $(MPICXXFILES),$(wildcard $(SRCDIR)/*.cc))

# .cu source files (GPU), excluding *_kernel.cu
CUFILES = $(filter-out %_kernel.cu,$(wildcard $(SRCDIR)/*.cu))

# headers
HEADERS = $(wildcard $(SRCDIR)/*.h)

# object files via filename replacement
MPICXXOBJS = $(patsubst %.cc,$(OBJDIR)/%.o,$(notdir $(MPICXXFILES)))
CCOBJS = $(patsubst %.cc,$(OBJDIR)/%.o,$(notdir $(CCFILES)))
CUOBJS = $(patsubst %.cu,$(OBJDIR)/%.o,$(notdir $(CUFILES)))

OBJS = $(CCOBJS) $(MPICXXOBJS) $(CUOBJS)

PROBLEM_LIST = $(basename $(notdir $(shell egrep -l 'class.*:.*Problem' $(HEADERS))))

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
HASH_KEY_SIZE_SELECT_OPTFILE=$(OPTSDIR)/hash_key_size_select.opt
MPI_SELECT_OPTFILE=$(OPTSDIR)/mpi_select.opt
HDF5_SELECT_OPTFILE=$(OPTSDIR)/hdf5_select.opt

# this is not really an option, but it follows the same mechanism
GPUSPH_VERSION_OPTFILE=$(OPTSDIR)/gpusph_version.opt

OPTFILES=$(PROBLEM_SELECT_OPTFILE) \
		 $(DBG_SELECT_OPTFILE) \
		 $(COMPUTE_SELECT_OPTFILE) \
		 $(FASTMATH_SELECT_OPTFILE) \
		 $(HASH_KEY_SIZE_SELECT_OPTFILE) \
		 $(MPI_SELECT_OPTFILE) \
		 $(HDF5_SELECT_OPTFILE) \
		 $(GPUSPH_VERSION_OPTFILE)

# Let make know that .opt dependencies are to be looked for in $(OPTSDIR)
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
			$(SED_COMMAND) 's/$(PROBLEM)/$(problem)/' $(PROBLEM_SELECT_OPTFILE) )
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

# option: hash_key_size - Size in bits of the hash used to sort particles, currently 32 or 64. Must
# option:                 be 64 to enable multi-device simulations. For single-device simulations,
# option:                 can be set to 32 to reduce memory usage. Default: 64
ifdef hash_key_size
	# does it differ from last?
	ifneq ($(HASH_KEY_SIZE),$(hash_key_size))
		TMP:=$(shell test -e $(HASH_KEY_SIZE_SELECT_OPTFILE) && \
			$(SED_COMMAND) 's/HASH_KEY_SIZE $(HASH_KEY_SIZE)/HASH_KEY_SIZE $(hash_key_size)/' $(HASH_KEY_SIZE_SELECT_OPTFILE) )
	endif
	# user choice
	HASH_KEY_SIZE=$(hash_key_size)
else
	HASH_KEY_SIZE ?= 64
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
MPICXX ?= $(shell which mpicxx)

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

# option: hdf5 - 0 do not use HDF5, 1 use HDF5. Default: autodetect
ifdef hdf5
	# does it differ from last?
	ifneq ($(USE_HDF5),$(hdf5))
		TMP := $(shell test -e $(HDF5_SELECT_OPTFILE) && \
			$(SED_COMMAND) 's/$(USE_HDF5)/$(hdf5)/' $(HDF5_SELECT_OPTFILE) )
		# user choice
		USE_HDF5=$(hdf5)
	endif
else
	# check if we can link to the HDF5 library, and disable HDF5 otherwise
	USE_HDF5 ?= $(shell $(CXX) $(LIBPATH) -shared -lhdf5 -o hdf5test 2> /dev/null && rm hdf5test && echo 1 || echo 0)
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
INCPATH += -I$(SRCDIR) -I$(OPTSDIR)

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
	LIBS += -lc++
endif

# link to the CUDA runtime library
LIBS += -lcudart
# link to ODE for the objects
LIBS += -lode

ifeq ($(USE_HDF5),1)
	# link to HDF5 for input reading
	LIBS += -lhdf5
else
	TMP := $(info HDF5 library not found, HDF5 input will NOT be supported)
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

# Define USE_HDF5 according to the availability of the HDF5 library
CPPFLAGS += -DUSE_HDF5=$(USE_HDF5)

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

# nvcc-specific flags

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
$(HASH_KEY_SIZE_SELECT_OPTFILE): | $(OPTSDIR)
	@echo "/* Determines the size in bits of the hashKey used to sort the particles on the device. */" \
		> $@
	@echo "#define HASH_KEY_SIZE $(HASH_KEY_SIZE)" >> $@
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
$(CCOBJS): $(OBJDIR)/%.o: $(SRCDIR)/%.cc $(HASH_KEY_SIZE_SELECT_OPTFILE) | $(OBJDIR)
	$(call show_stage,CC,$(@F))
	$(CMDECHO)$(CXX) $(CC_INCPATH) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

$(MPICXXOBJS): $(OBJDIR)/%.o: $(SRCDIR)/%.cc | $(OBJDIR)
	$(call show_stage,MPI,$(@F))
	$(CMDECHO)$(MPICXX) $(CC_INCPATH) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

# compile GPU objects
$(CUOBJS): $(OBJDIR)/%.o: $(SRCDIR)/%.cu $(COMPUTE_SELECT_OPTFILE) $(FASTMATH_SELECT_OPTFILE) $(HASH_KEY_SIZE_SELECT_OPTFILE) | $(OBJDIR)
	$(call show_stage,CU,$(@F))
	$(CMDECHO)$(NVCC) $(CPPFLAGS) $(CUFLAGS) -c -o $@ $<

# compile program to list compute capabilities of installed devices.
# Filter out -arch=sm_$(COMPUTE) from LDFLAGS becaus we already have it in CUFLAGS
# and it being present twice causes complains from nvcc
$(LIST_CUDA_CC): $(LIST_CUDA_CC).cu
	$(call show_stage,SCRIPTS,$(@F))
	$(CMDECHO)$(NVCC) $(CPPFLAGS) $(filter-out --ptxas-options=%,$(filter-out --generate-line-info,$(CUFLAGS))) -o $@ $< $(filter-out -arch=sm_%,$(LDFLAGS))

# create distdir
$(DISTDIR):
	$(CMDECHO)mkdir -p $(DISTDIR)

# create objdir
$(OBJDIR):
	$(CMDECHO)mkdir -p $(OBJDIR)

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

# target: cookiesclean - Clean last dbg, problem, compute, hash_key_size and fastmath choices,
# target:                forcing .*_select.opt files to be regenerated (use if they're messed up)
cookiesclean:
	$(RM) -r $(OPTFILES) $(OPTSDIR)

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
	@echo "Sources dir:     $(SRCDIR)"
	@echo "Options dir:     $(OPTSDIR)"
	@echo "Objects dir:     $(OBJDIR)"
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
	@echo "Hashkey size:    $(HASH_KEY_SIZE)"
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

$(SNAPSHOT_FILE):  ./$(MFILE_NAME) $(EXTRA_PROBLEM_FILES) $(DOXYCONF) $(SRCDIR)/*.cc $(SRCDIR)/*.h $(SRCDIR)/*.cu $(SRCDIR)/*.cuh $(SRCDIR)/*.def $(SCRIPTSDIR)/
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
	$(show_stage CONF,$@)
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
	$(CMDECHO)# recover value of HASH_KEY_SIZE from OPTFILES
	$(CMDECHO)grep "\#define HASH_KEY_SIZE" $(HASH_KEY_SIZE_SELECT_OPTFILE) | cut -f2-3 -d ' ' | tr ' ' '=' >> $@
	$(CMDECHO)# recover value of USE_MPI from OPTFILES
	$(CMDECHO)grep "\#define USE_MPI" $(MPI_SELECT_OPTFILE) | cut -f2-3 -d ' ' | tr ' ' '=' >> $@
	$(CMDECHO)# recover value of USE_HDF5 from OPTFILES
	$(CMDECHO)grep "\#define USE_HDF5" $(HDF5_SELECT_OPTFILE) | cut -f2-3 -d ' ' | tr ' ' '=' >> $@

confclean: cookiesclean
	$(RM) -f Makefile.conf

# Dependecies are generated by the C++ compiler, since nvcc does not understand the
# more sophisticated -MM and -MT dependency generation options.
# The -MM flag is used to not include system includes.
# The -MG flag is used to add missing includes (useful to depend on the .opt files).
# The sed expression adds $(OBJDIR) to all lines matching a literal '.o:',
# thereby correcting the path of the would-be output of compilation.
#
# When generating the dependencies for the .cu files, we must specify that they are
# to be interpeted as C++ files and not some other funky format. We also need
# to define __CUDA_INTERNAL_COMPILATION__ to mute an error during traversal of
# some CUDA system includes
#
# Finally, both GPUDEPS and CPUDEPS must depend on the _presence_ of the option file
# for hash keys, to prevent errors about undefined hash key size every time
# `src/hashkey.h` is preprocessed
$(GPUDEPS): $(CUFILES) | $(HASH_KEY_SIZE_SELECT_OPTFILE)
	$(show_stage DEPS,GPU)
	$(CMDECHO)echo '# GPU sources dependencies generated with "make deps"' > $@
	$(CMDECHO)$(CXX) -x c++ -D__CUDA_INTERNAL_COMPILATION__ \
		$(CC_INCPATH) $(CPPFLAGS) -MG -MM $^ | sed '/\.o:/ s!^!$(OBJDIR)/!' >> $@

$(CPUDEPS): $(CCFILES) $(MPICXXFILES) | $(HASH_KEY_SIZE_SELECT_OPTFILE)
	$(show_stage DEPS,CPU)
	$(CMDECHO)echo '# CPU sources dependencies generated with "make deps"' > $@
	$(CMDECHO)$(CXX) $(CC_INCPATH) $(CPPFLAGS) -MG -MM $^ | sed '/\.o:/ s!^!$(OBJDIR)/!' >> $@

# target: docs - Generate Doxygen documentation in $(DOCSDIR);
# target:        to produce refman.pdf, run "make pdf" in $(DOCSDIR)/latex/.
docs: $(DOXYCONF)
	$(CMDECHO)mkdir -p $(DOCSDIR)
	@echo Running Doxygen...
	$(CMDECHO)doxygen $(DOXYCONF) && \
	echo Generated Doxygen documentation in $(DOCSDIR)

# target: docsclean - Remove $(DOCSDIR)
docsclean:
	$(CMDECHO)rm -rf $(DOCSDIR)

# target: tags - Create TAGS file
tags: TAGS
TAGS: $(wildcard $(SRCDIR)/*)
	$(CMDECHO)etags -R -h=.h.cuh.inc --exclude=docs --langmap=c++:.cc.cuh.cu.def


# target: test - Run GPUSPH with WaveTank. Compile it if needed
test: all
	$(CMDECHO)$(TARGET)
	@echo Do "$(SCRIPTSDIR)/rmtests" to remove all tests

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

