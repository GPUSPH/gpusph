# GPUSPH Makefile
#
# TODO:
# - Add support for versioning with git
# - Improve list-problems problem detection (move problems to separate dir?)
# - Recompile also if target_arch changes (like dbg and compute)
#
# Notes:
# - When adding a target, comment it with "# target: name - desc" and help
#   will always be up-to-date
# - When adding an option, comment it with "# option: name - desc" and help
#   will always be up-to-date
# - Makefile is assumed to be GNU (see http://www.gnu.org/software/make/manual/)
# - Source C++ files have extension .cc (NOT .cpp)
# - C++ Headers have extension .h
# - CUDA C++ files have extension .cu
# - CUDA C++ headers have extension .cuh

# Include, if present a local Makefile.
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
GPUSPH_VERSION=$(shell git describe --tags --dirty 2> /dev/null)

ifeq ($(GPUSPH_VERSION), $(empty))
$(warning Unable to determine GPUSPH version)
GPUSPH_VERSION=unknown-version
endif

# system information
platform=$(shell uname -s 2>/dev/null)
platform_lcase=$(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
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

# option: dbg - 0 no debugging, 1 enable debugging
# (there are several switches below)
DBG_SFX=_dbg
ifeq ($(dbg), 1)
	TARGET_SXF=$(DBG_SFX)
else
	dbg=0
	TARGET_SXF=
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
TARGETNAME := GPUSPH$(TARGET_SXF)
TARGET := $(DISTDIR)/$(TARGETNAME)

# --------------- File lists

# makedepend will generate dependencies in these file
GPUDEPS = $(MAKEFILE).gpu
CPUDEPS = $(MAKEFILE).cpu

# .cc source files (CPU)
CCFILES = $(wildcard $(SRCDIR)/*.cc)
#CCFILES = $(shell ls $(SRCDIR)/*.cc) # via shell

# .cu source files (GPU), excluding *_kernel.cu
CUFILES = $(filter-out %_kernel.cu,$(wildcard $(SRCDIR)/*.cu))
#CUFILES = $(shell ls $(SRCDIR)/*.cu | grep -v _kernel.cu)  # via shell

# headers
HEADERS = $(wildcard $(SRCDIR)/*.h)

# object files via filename replacement
CCOBJS = $(patsubst %.cc,$(OBJDIR)/%.o,$(notdir $(CCFILES)))
CUOBJS = $(patsubst %.cu,$(OBJDIR)/%.o,$(notdir $(CUFILES)))
OBJS = $(CCOBJS) $(CUOBJS)

PROBLEM_LIST = $(basename $(notdir $(shell egrep -l 'class.*:.*Problem' $(HEADERS))))

# data files needed by some problems
EXTRA_PROBLEM_FILES ?=
# TestTopo uses this DEM:
EXTRA_PROBLEM_FILES += half_wave0.1m.txt

# --------------- Locate and set up compilers and flags

# CUDA installation/lib/include paths
# override by setting them in the enviornment
# Default to /usr/local/cuda if the install path
# has not been specified
CUDA_INSTALL_PATH ?= /usr/local/cuda

# We check the validity of the path by looking for /bin/nvcc under it.
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

ifeq ($(CUDA_MAJOR), 5)
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

# files to store last compile options: problem, dbg, compute, fastmath
PROBLEM_SELECT_OPTFILE=$(OPTSDIR)/problem_select.opt
DBG_SELECT_OPTFILE=$(OPTSDIR)/dbg_select.opt
COMPUTE_SELECT_OPTFILE=$(OPTSDIR)/compute_select.opt
FASTMATH_SELECT_OPTFILE=$(OPTSDIR)/fastmath_select.opt
# this is not really an option, but it follows the same mechanism
GPUSPH_VERSION_OPTFILE=$(OPTSDIR)/gpusph_version.opt

OPTFILES=$(PROBLEM_SELECT_OPTFILE) $(DBG_SELECT_OPTFILE) $(COMPUTE_SELECT_OPTFILE) $(FASTMATH_SELECT_OPTFILE) $(GPUSPH_VERSION_OPTFILE)

# Let make know that .opt dependencies are to be looked for in $(OPTSDIR)
vpath %.opt $(OPTSDIR)

# check compile options used last time:
# - which problem? (name of problem, empty if file doesn't exist)
LAST_PROBLEM=$(shell test -e $(PROBLEM_SELECT_OPTFILE) && \
	grep "\#define PROBLEM" $(PROBLEM_SELECT_OPTFILE) | cut -f3 -d " ")
# - was dbg enabled? (1 or 0, empty if file doesn't exist)
# "strip" added for Mac compatibility: on MacOS wc outputs a tab...
LAST_DBG=$(strip $(shell test -e $(DBG_SELECT_OPTFILE) && \
	grep "\#define _DEBUG_" $(DBG_SELECT_OPTFILE) | wc -l))
# - for which compute capability? (11, 12 or 20, empty if file doesn't exist)
LAST_COMPUTE=$(shell test -e $(COMPUTE_SELECT_OPTFILE) && \
	grep "\#define COMPUTE" $(COMPUTE_SELECT_OPTFILE) | cut -f3 -d " ")
# - was fastmath enabled? (1 or 0, empty if file doesn't exist)
LAST_FASTMATH=$(shell test -e $(FASTMATH_SELECT_OPTFILE) && \
	grep "\#define FASTMATH" $(FASTMATH_SELECT_OPTFILE) | cut -f3 -d " ")

# update GPUSPH_VERSION_OPTFILE if git version changed
LAST_GPUSPH_VERSION=$(shell test -e $(GPUSPH_VERSION_OPTFILE) && \
	grep "\#define GPUSPH_VERSION" $(GPUSPH_VERSION_OPTFILE) | cut -f2 -d\")

ifneq ($(LAST_GPUSPH_VERSION),$(GPUSPH_VERSION))
	TMP:=$(shell test -e $(GPUSPH_VERSION_OPTFILE) && \
		$(SED_COMMAND) 's/$(LAST_GPUSPH_VERSION)/$(GPUSPH_VERSION)/' $(GPUSPH_VERSION_OPTFILE) )
endif


# option: problem - Name of the problem. Default: $(PROBLEM) in makefile
ifdef problem
	# user chooses
	PROBLEM=$(problem)
	# if choice differs from last...
	ifneq ($(LAST_PROBLEM),$(PROBLEM))
		# check that the problem is in the problem list
		ifneq ($(filter $(PROBLEM),$(PROBLEM_LIST)),$(PROBLEM))
			TMP:=$(error No such problem ‘$(PROBLEM)’. Known problems: $(PROBLEM_LIST))
		endif
		# empty string in sed for Mac compatibility
		TMP:=$(shell test -e $(PROBLEM_SELECT_OPTFILE) && \
			$(SED_COMMAND) 's/$(LAST_PROBLEM)/$(PROBLEM)/' $(PROBLEM_SELECT_OPTFILE) )
	endif
else
	# no user choice, use last
	ifeq ($(strip $(LAST_PROBLEM)),)
		PROBLEM=DamBreak3D
	else
		PROBLEM=$(LAST_PROBLEM)
	endif
endif

# see above for dbg option description
# does dbg differ from last?
ifneq ($(dbg), $(LAST_DBG))
	# if last choice is empty, file probably doesn't exist and will be regenerated
	ifneq ($(strip $(LAST_DBG)),)
		ifeq ($(dbg),1)
			_SRC=undef
			_REP=define
		else
			_SRC=define
			_REP=undef
		endif
		# empty string in sed for Mac compatibility
		TMP:=$(shell test -e $(DBG_SELECT_OPTFILE) && \
			$(SED_COMMAND) 's/$(_SRC)/$(_REP)/' $(DBG_SELECT_OPTFILE) )
	endif
endif

# option: compute - 11, 12 or 20: compute capability to compile for (default: 12)
# option:           Override CFLAGS to compile for multiple architectures.
# does dbg differ from last?
ifdef compute
	# user choice
	COMPUTE=$(compute)
	# does it differ from last?
	ifneq ($(LAST_COMPUTE),$(COMPUTE))
		# empty string in sed for Mac compatibility
		TMP:=$(shell test -e $(COMPUTE_SELECT_OPTFILE) && \
			$(SED_COMMAND) 's/$(LAST_COMPUTE)/$(COMPUTE)/' $(COMPUTE_SELECT_OPTFILE) )
	endif
else
	# no user choice, use last (if any) or default
	ifeq ($(strip $(LAST_COMPUTE)),)
		COMPUTE=12
	else
		COMPUTE=$(LAST_COMPUTE)
	endif
endif

# option: fastmath - Enable or disable fastmath. Default: 1 (enabled)
ifdef fastmath
	# user chooses
	FASTMATH=$(fastmath)
	# does it differ from last?
	ifneq ($(LAST_FASTMATH),$(FASTMATH))
		TMP:=$(shell test -e $(FASTMATH_SELECT_OPTFILE) && \
			$(SED_COMMAND) 's/FASTMATH $(LAST_FASTMATH)/FASTMATH $(FASTMATH)/' $(FASTMATH_SELECT_OPTFILE) )
	endif
else
	ifeq ($(LAST_FASTMATH),)
		FASTMATH=1
	else
		FASTMATH=$(LAST_FASTMATH)
	endif
endif

# --- Includes and library section start ---

# architecture switch. The *_SFX vars will be used later.
GLEW_ARCH_SFX=
LIB_PATH_SFX=

ifeq ($(arch), x86_64)
	TARGET_ARCH ?= -m64
	# cuda 5.x with 64bit libs does not require the suffix anymore
	ifneq ($(CUDA_MAJOR), 5)
		GLEW_ARCH_SFX=_x86_64
	endif
	# on Linux, toolkit libraries are under /lib64 for 64-bit
	ifeq ($(platform), Linux)
		LIB_PATH_SFX=64
	endif
else # i386 or i686
	TARGET_ARCH ?= -m32
endif

# paths for include files
INCPATH ?=
# paths for library searches
LIBPATH ?=
# libraries
LIBS ?=

# flags passed to the linker
LDFLAGS ?=

# Most of these settings are platform independent

# INCPATH
# make GPUSph.cc find problem_select.opt, and problem_select.opt find the problem header
INCPATH += -I$(SRCDIR) -I$(OPTSDIR)

# access the CUDA include files from the C++ compiler too, but mark it as a system include path
# Note: -isystem is supported by nvcc, g++ and clang++, so this should be fine
INCPATH += -isystem $(CUDA_INSTALL_PATH)/include

# since we prefer to use GLEW from the SDK samples, we also need these
# up to 4.x
INCPATH += -isystem $(CUDA_SDK_PATH)/C/common/inc
# 5.x
INCPATH += -isystem $(CUDA_SDK_PATH)/common/inc


# LIBPATH
LIBPATH += -L/usr/local/lib

# CUDA libaries
LIBPATH += -L$(CUDA_INSTALL_PATH)/lib$(LIB_PATH_SFX)
# search path for GLEW from the SDK samples
# up to 4.x
LIBPATH += -L$(CUDA_SDK_PATH)/C/common/lib/$(platform_lcase)/

# On Darwin 10.9 with CUDA 5.5 using clang we want to link with the clang c++ stdlib.
# This is exactly the conditions under which we set WE_USE_CLANG
ifeq ($(WE_USE_CLANG),1)
	LIBS += -lc++
endif

# link to the CUDA runtime library
LIBS += -lcudart
# link to ODE for the objects
LIBS += -lode

LIBS += -lGLEW$(GLEW_ARCH_SFX)

# search paths (for CUDA 5 and higher) and linking to OpenGL are
# platform-specific
ifeq ($(platform), Darwin)
	LIBPATH += -L$(CUDA_SDK_PATH)/common/lib/$(platform_lcase)/
	LDFLAGS += -Xlinker -framework,OpenGL,-framework,GLUT
else
	LIBPATH += -L$(CUDA_SDK_PATH)/common/lib/$(platform_lcase)/$(arch)/
	LIBS += -lGL -lGLU -lglut
endif

LDFLAGS += $(LIBPATH) $(LIBS)

# -- Includes and library section end ---

# -------------------------- CFLAGS section -------------------------- #
# We have three sets of flags:
# CPPFLAGS are preprocessor flags; they are common to both compilers
# CXXFLAGS are flags passed to the C++ when compiling C++ files (either directly,
#     for .cc files, or via nvcc, for .cu files), and when linking
# CUFLAGS are flags passed to the CUDA compiler when compiling CUDA files

# initialize them by reading them from the environment, if present
# this allows the user to add specific options
CPPFLAGS ?=
CXXFLAGS ?=
CUFLAGS  ?=

# First of all, put the include paths into the CPPFLAGS
CPPFLAGS += $(INCPATH)

# We set __COMPUTE__ on the host to match that automatically defined
# by the compiler on the device
CPPFLAGS += -D__COMPUTE__=$(COMPUTE)

# The ODE library link is in single precision mode
CPPFLAGS += -DdSINGLE

# CXXFLAGS start with the target architecture
CXXFLAGS += $(TARGET_ARCH)

# nvcc-specific flags
CUFLAGS += -arch=sm_$(COMPUTE) -lineinfo

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

# snapshot tarball filename
SNAPSHOT_FILE = ./GPUSPHsnapshot.tgz

# option: plain - 0 fancy line-recycling stage announce, 1 plain multi-line stage announce
ifeq ($(plain), 1)
	show_stage = @printf "[$(1)] $(2)\n"
else
	show_stage = @printf "\r                                 \r[$(1)] $(2)"
endif

# option: echo - 0 silent, 1 show commands
ifeq ($(echo), 1)
	CMDECHO :=
else
	CMDECHO := @
endif

.PHONY: all showobjs show snapshot expand deps docs test help
.PHONY: clean cpuclean gpuclean docsclean

# target: all - Make subdirs, compile objects, link and produce $(TARGET)
# link objects in target
all: $(OBJS) | $(DISTDIR)
	@echo
	@echo "Compiled with problem $(PROBLEM)"
	@[ $(FASTMATH) -eq 1 ] && echo "Compiled with fastmath" || echo "Compiled without fastmath"
	$(call show_stage,LINK,$(TARGET)\\n)
	$(CMDECHO)$(NVCC) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET) $(OBJS) $(LIBS) && \
	ln -sf $(TARGET) $(CURDIR)/$(TARGETNAME) && echo "Success."

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
$(COMPUTE_SELECT_OPTFILE): | $(OPTSDIR)
	@echo "/* Define the compute capability GPU code was compiled for. */" \
		> $(COMPUTE_SELECT_OPTFILE)
	@echo "#define COMPUTE $(COMPUTE)" >> $(COMPUTE_SELECT_OPTFILE)
$(FASTMATH_SELECT_OPTFILE): | $(OPTSDIR)
	@echo "/* Determines if fastmath is enabled for GPU code. */" \
		> $@
	@echo "#define FASTMATH $(FASTMATH)" >> $@

$(GPUSPH_VERSION_OPTFILE): | $(OPTSDIR)
	@echo "/* git version of GPUSPH. */" \
		> $@
	@echo "#define GPUSPH_VERSION \"$(GPUSPH_VERSION)\"" >> $@


$(OBJS): $(DBG_SELECT_OPTFILE)

# compile CPU objects
$(CCOBJS): $(OBJDIR)/%.o: $(SRCDIR)/%.cc | $(OBJDIR)
	$(call show_stage,CC,$(@F))
	$(CMDECHO)$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

# compile GPU objects
$(CUOBJS): $(OBJDIR)/%.o: $(SRCDIR)/%.cu $(COMPUTE_SELECT_OPTFILE) $(FASTMATH_SELECT_OPTFILE) | $(OBJDIR)
	$(call show_stage,CU,$(@F))
	$(CMDECHO)$(NVCC) $(CPPFLAGS) $(CUFLAGS) -c -o $@ $<

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
	$(RM) $(CPUDEPS) $(GPUDEPS) $(TARGET) $(CURDIR)/$(TARGETNAME)
	if [ -f $(TARGET)$(DBG_SFX) ] ; then \
		$(RM) $(TARGET)$(DBG_SFX) $(CURDIR)/$(TARGETNAME)$(DBG_SFX) ; fi

# target: cpuclean - Clean CPU stuff
cpuclean:
	$(RM) $(CCOBJS)

# target: gpuclean - Clean GPU stuff
gpuclean:
	$(RM) $(CUOBJS)

# target: cookiesclean - Clean last dbg, problem, compute and fastmath choices, forcing
# target:                .*_select.opt files to be regenerated (use if they're
# target:                messed up)
cookiesclean:
	$(RM) -r $(OPTFILES) $(OPTSDIR)

# target: showobjs - List detected sources and target objects
showobjs:
	@echo "> CCFILES: $(CCFILES)"
	@echo " --- "
	@echo "> CUFILES: $(CUFILES)"
	@echo " --- "
	@echo "> CCOBJS: $(CCOBJS)"
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
	@echo "Sources dir:     $(SRCDIR)"
	@echo "Options dir:     $(OPTSDIR)"
	@echo "Objects dir:     $(OBJDIR)"
	@echo "Scripts dir:     $(SCRIPTSDIR)"
	@echo "Docs dir:        $(DOCSDIR)"
	@echo "Doxygen conf:    $(DOXYCONF)"
	@echo "Verbose:         $(verbose)"
	@echo "Debug:           $(dbg)"
	@echo "CXX:             $(CXX)"
	@echo "nvcc:            $(NVCC)"
	@echo "nvcc version:    $(NVCC_VER)"
	@echo "Compute cap.:    $(COMPUTE)"
	@echo "Fastmath:        $(FASTMATH)"
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
snapshot:
	$(CMDECHO)tar czf $(SNAPSHOT_FILE) ./$(MFILE_NAME) $(EXTRA_PROBLEM_FILES) $(DOXYCONF) $(SRCDIR)/*.cc $(HEADERS) $(SRCDIR)/*.cu $(SRCDIR)/*.cuh $(SRCDIR)/*.inc $(SRCDIR)/*.def $(SCRIPTSDIR)/ && echo "Created $(SNAPSHOT_FILE)"

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

# Let Makefile depend on the presence of the option files. This ensures that they
# are built before anything else
Makefile: | $(OPTFILES)

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
$(GPUDEPS): $(CUFILES)
	@echo [DEPS] GPU
	$(CMDECHO)echo '# GPU sources dependencies generated with "make deps"' > $@
	$(CMDECHO)$(CXX) -x c++ -D__CUDA_INTERNAL_COMPILATION__ \
		$(CPPFLAGS) -MG -MM $^ | sed '/\.o:/ s!^!$(OBJDIR)/!' >> $@

$(CPUDEPS): $(CCFILES)
	@echo [DEPS] CPU
	$(CMDECHO)echo '# CPU sources dependencies generated with "make deps"' > $@
	$(CMDECHO)$(CXX) $(CPPFLAGS) -MG -MM $^ | sed '/\.o:/ s!^!$(OBJDIR)/!' >> $@

# target: docs - Generate Doxygen documentation in $(DOCSDIR);
# target:        to produce refman.pdf, run "make pdf" in $(DOCSDIR)/latex/.
docs: $(DOXYCONF)
	$(CMDECHO)mkdir -p $(DOCSDIR)
	@echo Running Doxygen...
	$(CMDECHO)doxygen $(DOXYCONF) > /dev/null && \
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
	@echo "Note: if a CFLAGS environment variable is set, it is used as is."
	@echo "Unset it ('unset CFLAGS') to let Makefile set the flags."

# target: help-targets - Display callable targets
help-targets:
	@echo "Available targets:"
	@grep -e "^# target:" $(MAKEFILE) | sed 's/^# target: /    /' # | sed 's/ - /\t/'

# target: help-options - Display available options
help-options:
	@echo "Options:"
	@grep -e "^# option:" $(MAKEFILE) | sed 's/^# option: /    /' # | sed 's/ - /\t/'
	@echo "(usage: make option=value)"

# "sinclude" instead of "include" tells make not to print errors if files are missing.
# This is necessary because during the first processing of the makefile, make complains
# before creating them.
sinclude $(GPUDEPS)
sinclude $(CPUDEPS)

