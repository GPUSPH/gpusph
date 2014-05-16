#!/bin/bash

# Simulation parameters -------------------------------------- #

# - standard DamBreak3D: dp 0.003 -> ~4M particles
export DELTAP=0.003

# use very short TEND for profiling or logs will be huge
# (e.g. 0.05 for CPU prof, 0.002 for GPU prof); on the
# other hand, big TEND is better for performance tests
export TEND=1.5
#export TEND=0.002
#export TEND=0.05

# extra options for GPUSPH - it is a good idea to write them
# in the PREFIX
export EXTRAFLAGS="--nosave --striping"
#export EXTRAFLAGS="--nosave"

# Output parameters ------------------------------------------ #

# - use a descriptive prefix or logs will be a mess
export PREFIX="linXZY_noprof_strip"
#export PREFIX="linXYZ_nostriping"

# directories
export LOGDIR=./logs
export PROFILINGDIR=./profiling
export GPROFDIR=./gprof

# GPU profiling parameters ----------------------------------- #

# enable GPU profiling (small tend recommended, if active)
export COMPUTE_PROFILE=0

# - two profile configs: normal (with info such as shmem) or minimal (just for the timeline)
#export COMPUTE_PROFILE_CONFIG=${PROFILINGDIR}/cuda_profile_config
export COMPUTE_PROFILE_CONFIG=${PROFILINGDIR}/cuda_minimal_profile_config

# - CSV is compatible with the old, custom SVG timeline profiler
export COMPUTE_PROFILE_CSV=1

# Tests - here we iterate on one or more params -------------- #

for NUMGPUS in $(seq 6 -1 1)
do

	export GPUS=$(seq -s ',' 0 $(( $NUMGPUS - 1 )) )
	export NAME=${PREFIX}_${NUMGPUS}gpus_tend${TEND}_dp${DELTAP}
	export DIRNAME=./tests/${NAME}
	export LOGNAME=${LOGDIR}/${NAME}.log

	# if GPU profiling is active, create a subdirectory for the csv logs
	export PROFILING_SUBDIR=${PROFILINGDIR}/${NAME}
	if test "${COMPUTE_PROFILE}" -eq "1"
	then
		mkdir ${PROFILING_SUBDIR} 2> /dev/null
	else
		rm -rf ${PROFILING_SUBDIR} 2> /dev/null
	fi
	# NOTE: it important to include %p (pid) in the filename when profiling  multinode simulations
	export COMPUTE_PROFILE_LOG=${PROFILINGDIR}/${PROFILING_SUBDIR}/${NAME}_p%pd%d.csv

	# prepare command
	export COMMAND="./GPUSPH --deltap ${DELTAP} --tend ${TEND} --device ${GPUS} --dir ${DIRNAME} ${EXTRAFLAGS}"

	# remove leftovers of previous runs, if any
	rm -rf "${LOGNAME}" "${DIRNAME}" &> /dev/null

	# init logfile
	date > "${LOGNAME}"
	echo >> "${LOGNAME}"
	echo "${COMMAND}" >> "${LOGNAME}"

	echo "Running ${NAME} ..."

	# run, baby, run!
	(time ${COMMAND}) &>> "${LOGNAME}"

	# CPU profiling with gmon (if -pg was active)
	if test -e ./gmon.out
	then
		gprof ./GPUSPH ./gmon.out &> ${GPROFDIR}/analysis_${NAME}.txt
		mv gmon.out ${GPROFDIR}/gmon.out.${NAME}
	fi

	# print MIPPS for quick overview
	grep MIPPS "${LOGNAME}" | tail -n 1
done

echo End

