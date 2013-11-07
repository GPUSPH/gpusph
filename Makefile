# GPU-SPH3D Makefile
# Rewritten from scratch by rustico@dmi.unict.it
#
# TODO:
# - Add support for versioning with git
# - Improve list-problems problem detection
# - If user overrides CFLAGS, avoid dbg and compute recompile?
# - Recompile also if target_arch changes (like dbg and compute)
# - Add automatic deps update?
#   "g++ -M" or http://make.paulandlesley.org/autodep.html
# - remember \callgraph in main classes (move from here!)
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

# system information
platform=$(shell uname -s 2>/dev/null)
platform_lcase=$(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
arch=$(shell uname -m)

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

# CUDA installation/lib/include paths
ifeq ($(platform), Linux)
	CUDA_INSTALL_PATH=/usr/local/cuda
	CUDA_SDK_PATH=/usr/local/cudasdk
else ifeq ($(platform), Darwin)
	CUDA_INSTALL_PATH=/usr/local/cuda
	CUDA_SDK_PATH=/usr/local/cuda/samples
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
#CUDA_MINOR := $(lastword  $(versions_tmp))

# files to store last compile options: problem, dbg, compute
PROBLEM_SELECT_OPTFILE=$(OPTSDIR)/problem_select.opt
DBG_SELECT_OPTFILE=$(OPTSDIR)/dbg_select.opt
COMPUTE_SELECT_OPTFILE=$(OPTSDIR)/compute_select.opt

# check compile options used last time:
# - which problem? (name of problem, empty if file doesn't exist)
LAST_PROBLEM=$(shell test -e $(PROBLEM_SELECT_OPTFILE) && \
	grep "\#define PROBLEM" $(PROBLEM_SELECT_OPTFILE) | cut -f3 -d " ")
# - was dbg enabled? (1 or 0, empty if file doesn't exist))
# "strip" added for Mac compatibility: on MacOS wc outputs a tab...
LAST_DBG=$(strip $(shell test -e $(DBG_SELECT_OPTFILE) && \
	grep "\#define _DEBUG_" $(DBG_SELECT_OPTFILE) | wc -l))
# - for which compute capability? (11, 12 or 20, empty if file doesn't exist)
LAST_COMPUTE=$(shell test -e $(COMPUTE_SELECT_OPTFILE) && \
	grep "\#define COMPUTE" $(COMPUTE_SELECT_OPTFILE) | cut -f3 -d " ")

# sed syntax differs a bit
ifeq ($(platform), Darwin)
	SED_COMAND=sed -i "" -e
else # Linux
	SED_COMAND=sed -i -e
endif

# option: problem - Name of the problem. Default: $(PROBLEM) in makefile
ifdef problem
	# user chooses
	PROBLEM=$(problem)
	# if choice differs from last...
	ifneq ($(LAST_PROBLEM),$(PROBLEM))
		# empty string in sed for Mac compatibility
		TMP:=$(shell test -e $(PROBLEM_SELECT_OPTFILE) && \
			$(SED_COMAND) 's/$(LAST_PROBLEM)/$(PROBLEM)/' $(PROBLEM_SELECT_OPTFILE) )
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
			$(SED_COMAND) 's/$(_SRC)/$(_REP)/' $(DBG_SELECT_OPTFILE) )
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
			$(SED_COMAND) 's/$(LAST_COMPUTE)/$(COMPUTE)/' $(COMPUTE_SELECT_OPTFILE) )
	endif
else
	# no user choice, use last (if any) or default
	ifeq ($(strip $(LAST_COMPUTE)),)
		COMPUTE=12
	else
		COMPUTE=$(LAST_COMPUTE)
	endif
endif

# -------------------------- CFLAGS section -------------------------- #

# nvcc-specific CFLAGS
CFLAGS_GPU = -arch=sm_$(COMPUTE) --use_fast_math -D__COMPUTE__=$(COMPUTE)

# add debug flag -G
ifeq ($(dbg), 1)
	CFLAGS_GPU += -G
endif

# Default CFLAGS (see notes below)
ifeq ($(platform), Darwin)
	CFLAGS_STANDARD =
else # Linux
	CFLAGS_STANDARD = -O3
endif
# For some strange reason, when compiled with optmisation, the CPU side code
# crashes in libstdc++ when acessing a file.
# Default debug CFLAGS: no -O optimizations, debug (-g) option
# Note: -D_DEBUG_ is defined in $(DBG_SELECT_OPTFILE); however, to avoid adding an
# include to every source, the _DEBUG_ macro is still passed through g++
CFLAGS_DEBUG = -g -D_DEBUG_

# see above for dbg option description
ifeq ($(dbg), 1)
	_CFLAGS = $(CFLAGS_DEBUG)
else
	_CFLAGS = $(CFLAGS_STANDARD)
endif

# architecture switch. The *_SFX vars will be used later.
ifeq ($(arch), x86_64)
	_CFLAGS_ARCH += -m64
	# cuda 5.0 with 64bit libs does not require the suffix anymore
	ifneq ($(CUDA_MAJOR), 5)
		GLEW_ARCH_SFX=_x86_64
	endif
else # i386 or i686
	_CFLAGS_ARCH += -m32
	GLEW_ARCH_SFX=
endif

# Only assign a default CFLAG if it wasn't already set by the user
# ("export CFLAGS=... ; make")
CFLAGS ?= $(_CFLAGS_ARCH) $(_CFLAGS) $(CFLAGS_GPU)

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

# compilers
CC=$(NVCC)
CXX=$(NVCC)
LINKER=$(NVCC)
#LINKER=g++

# Doxygen configuration
DOXYCONF = ./Doxygen_settings

# snapshot tarball filename
SNAPSHOT_FILE = ./GPUSPHsnapshot.tgz

# tool to clear current line
CARRIAGE_RETURN=printf "\r                                 \r"
# use the following instead to output on multiple lines
#CARRIAGE_RETURN=printf "\n"

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

# option: echo - 0 silent, 1 show commands
ifeq ($(echo), 1)
	CMDECHO :=
else
	CMDECHO := @
endif

# some platform-dependent configurations
ifeq ($(platform), Linux)
	# we need linking to SDKs for lGLEW
	INCPATH=-I $(CUDA_INSTALL_PATH)/include -I $(CUDA_SDK_PATH)/shared/inc
	LIBPATH=-L /usr/local/lib -L $(CUDA_SDK_PATH)shared/lib/linux -L $(CUDA_SDK_PATH)lib -L $(CUDA_SDK_PATH)/C/common/lib/linux/
	LIBS=-lstdc++ -lcudart -lGL -lGLU -lglut -lGLEW$(GLEW_ARCH_SFX)
	LFLAGS=
# 	CC=nvcc
# 	CXX=$(CC)
# 	LINKER=$(CXX)
else ifeq ($(platform), Darwin)
	INCPATH=-I $(CUDA_SDK_PATH)/common/inc/
	LIBPATH=-L/System/Library/Frameworks/OpenGL.framework/Libraries -L$(CUDA_SDK_PATH)/common/lib/darwin/ -L$(CUDA_INSTALL_PATH)/lib/
	LIBS=-lGL -lGLU $(CUDA_SDK_PATH)/common/lib/darwin/libGLEW.a -lcudart
	# Netbeans g++ flags: "-fPic -m32 -arch i386 -framework GLUT"
	LFLAGS=$(_CFLAGS_ARCH) -Xlinker -framework -Xlinker GLUT -Xlinker -rpath -Xlinker $(CUDA_INSTALL_PATH)/lib
	CC=$(NVCC)
	CXX=$(NVCC)
	LINKER=$(NVCC)
else
	$(warning architecture $(arch) not supported by this makefile)
endif

# make GPUSph.cc find problem_select.opt, and problem_select.opt find the problem header
INCPATH+= -I$(SRCDIR) -I$(OPTSDIR)

# option: verbose - 0 quiet compiler, 1 ptx assembler, 2 all warnings
ifeq ($(verbose), 1)
	CFLAGS += --ptxas-options=-v
else ifeq ($(verbose), 2)
	CFLAGS += --ptxas-options=-v
	CFLAGS += --compiler-options=-Wall
endif

.PHONY: all showobjs show snapshot expand deps docs test help
.PHONY: clean cpuclean gpuclean docsclean

# target: all - Make subdirs, compile objects, link and produce $(TARGET)
# link objects in target
all: $(OBJS) | $(DISTDIR)
	@echo
	@echo "Compiled with problem $(PROBLEM)"
	$(CMDECHO)$(CARRIAGE_RETURN) && \
	printf "[LINK] $(TARGET)\n" && \
	$(LINKER) $(LFLAGS) $(LIBPATH) -o $(TARGET) $(OBJS) $(LIBS) && \
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

$(OBJDIR)/GPUSph.o: $(PROBLEM_SELECT_OPTFILE)

$(OBJS): $(DBG_SELECT_OPTFILE)

# compile CPU objects
$(CCOBJS): $(OBJDIR)/%.o: $(SRCDIR)/%.cc | $(OBJDIR)
	$(CMDECHO)$(CARRIAGE_RETURN) && \
	printf "[CC] $(@F)..." && \
	$(CXX) $(CFLAGS) $(INCPATH) -c -o $@ $<

# compile GPU objects
$(CUOBJS): $(OBJDIR)/%.o: $(SRCDIR)/%.cu $(COMPUTE_SELECT_OPTFILE) | $(OBJDIR)
	$(CMDECHO)$(CARRIAGE_RETURN) && \
	printf "[CU] $(@F)" && \
	$(CXX) $(CFLAGS) $(INCPATH) -c -o $@ $<

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

# target: cookiesclean - Clean last dbg, problem and compute choices, forcing
# target:                .*_select.opt files to be regenerated (use if they're
# target:                messed up)
cookiesclean:
	$(RM) $(PROBLEM_SELECT_OPTFILE) $(DBG_SELECT_OPTFILE) $(COMPUTE_SELECT_OPTFILE) $(OPTSDIR)

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
	@echo "Platform:        $(platform)"
	@echo "NVCC version:    $(NVCC_VER)"
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
	@echo "CC:              $(CC)"
	@echo "CXX:             $(CXX)"
	@echo "Compute cap.:    $(COMPUTE)"
	@echo "INCPATH:         $(INCPATH)"
	@echo "LIBPATH:         $(LIBPATH)"
	@echo "LIBS:            $(LIBS)"
	@echo "LFLAGS:          $(LFLAGS)"
	@echo "CFLAGS:          $(CFLAGS)"
#	@echo "Suffixes:        $(SUFFIXES)"

# target: snapshot - Make a snapshot of current sourcecode in $(SNAPSHOT_FILE)
# it seems tar option --totals doesn't work
# use $(shell date +%F_%T) to include date and time in filename
snapshot:
	$(CMDECHO)tar czf $(SNAPSHOT_FILE) ./$(MFILE_NAME) $(DOXYCONF) $(SRCDIR)/*.cc $(HEADERS) $(SRCDIR)/*.cu $(SRCDIR)/*.cuh $(SRCDIR)/*.inc $(SRCDIR)/*.def $(SCRIPTSDIR)/ && echo "Created $(SNAPSHOT_FILE)"

# target: expand - Expand euler* and forces* GPU code in $(EXPDIR)
# it is safe to say we don't actualy need this
expand:
	$(CMDECHO)mkdir -p $(EXPDIR)
	$(CMDECHO)$(CXX) $(LFLAGS) $(CFLAGS) $(INCPATH) $(LIBPATH) -E \
		$(SRCDIR)/euler.cu -o $(EXPDIR)/euler.expand.cc && \
	$(CXX) $(LFLAGS) $(CFLAGS) $(INCPATH) $(LIBPATH) -E \
		$(SRCDIR)/euler_kernel.cu -o $(EXPDIR)/euler_kernel.expand.cc && \
	$(CXX) $(LFLAGS) $(CFLAGS) $(INCPATH) $(LIBPATH) -E \
		$(SRCDIR)/forces.cu -o $(EXPDIR)/forces.expand.cc && \
	$(CXX) $(LFLAGS) $(CFLAGS) $(INCPATH) $(LIBPATH) -E \
		$(SRCDIR)/forces_kernel.cu -o $(EXPDIR)/forces_kernel.expand.cc && \
	echo "euler* and forces* expanded in $(EXPDIR)."

# target: deps - Update dependencies in $(MAKEFILE) with "makedepend"
deps: $(GPUDEPS) $(CPUDEPS)
	@true

$(GPUDEPS): $(CUFILES) $(SRCDIR)
	@echo [DEPS] GPU
	$(CMDECHO)makedepend -Y -s'# GPU sources dependencies generated with "make deps"' \
		-w4096 -f- -- $(CFLAGS) -- $^ 2> /dev/null | \
		sed -e 's/$(subst ./,,$(SRCDIR))/$(subst ./,,$(OBJDIR))/' > $(GPUDEPS)

$(CPUDEPS): $(CCFILES) $(SRCDIR)
	@echo [DEPS] CPU
	$(CMDECHO)makedepend -Y -a -s'# CPU sources dependencies generated with "make deps"' \
		-w4096 -f- -- $(CFLAGS) -- $^ 2> /dev/null | \
		sed -e 's/$(subst ./,,$(SRCDIR))/$(subst ./,,$(OBJDIR))/' > $(CPUDEPS)

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
	$(CMDECHO)egrep "::.*:.*Problem\(" $(CCFILES) | \
		sed 's/.\/$(subst ./,,$(SRCDIR))\///g' | sed 's/\..*//g'

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

