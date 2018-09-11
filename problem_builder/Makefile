# GPUSPH Problem Builder Makefile
#
# Include, if present, a local Makefile.
# This can be used by the user to set the path to gtest in GTEST_DIR variable
sinclude Makefile.local

TARGET=../build_problem
TEST_TARGET=../Test_build_problem
PBCPPFILES = $(shell find . -type f \( -name \*.c -o -name \*.cpp \) -not \
\( -path "*/test/*" -o -path "*/user_functions_examples/*" \))
PBHEADERS = $(shell find . -type f -name \*.h -not \( -path "*/test/*" -o \
-path "*/user_functions_examples/*" \))
TSTSOURCES = $(shell find ./test -type f \( -name \*.c -o -name \*.cpp \) \
-not \( -path "*/test/data/*" \))
PBSOURCES = $(PBHEADERS) $(PBCPPFILES)
PBCPPTEST = $(filter-out ./generate_gpusph_sources.cpp, $(PBCPPFILES))

CXX_OPTS:= -std=gnu++11
#CXX_OPTS:= -std=c++11

empty:=
BUILD_TEST_CMD:=
TEST_LD_LIB:=
ifdef GTEST_DIR
	ifneq ($(GTEST_DIR), $(empty))
		BUILD_TEST_CMD =g++ $(CXX_OPTS) -lrt -lm -I$(GTEST_DIR)/include \
		           -I. \
	             -g $(TSTSOURCES) \
               -rdynamic \
               $(PBCPPTEST) \
	             $(GTEST_DIR)/lib/libgtest.so \
               $(GTEST_DIR)/lib/libgtest_main.so \
	             -o $(TEST_TARGET)
	  TEST_LD_LIB =$(GTEST_DIR)/lib
	endif
endif 

# target: all - Make all
all : $(TARGET) $(TEST_TARGET)
	@echo PB sources: $(PBSOURCES)
	@echo Test sources: $(TSTSOURCES)
	@echo PB headers: $(PBHEADERS)
	@echo PBCPPTEST: $(PBCPPTEST)
	@echo BUILD_TEST_CMD: $(BUILD_TEST_CMD)
	
# target: build_problem - Compile the problem builder executable
$(TARGET) : $(PBSOURCES) ../compile.sh
	cd .. && compile.sh

# target: clean - Clean Problem Builder project
clean :
	rm $(TARGET)
	rm $(TEST_TARGET)

.PHONY : clean

# target: test - Run unit tests
# do cd .. to keep relative paths to test data files
test : 
	export LD_LIBRARY_PATH=$(TEST_LD_LIB) && cd .. && problem_builder/$(TEST_TARGET)
	
.PHONY : test

# target: make_test - Build unit tests
$(TEST_TARGET) : $(TSTSOURCES) $(TARGET)
	$(BUILD_TEST_CMD)