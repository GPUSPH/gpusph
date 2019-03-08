#!/bin/sh
# Output a line in the form '#define COMPUTE CC' where
# CC are two digits representing a CUDA compute capability (without the dot)
# If a parameter is passed, that's used as CC. Otherwise, we try to run
# list-cuda-cc in our own directory and use that to determine the CC.

CC="$1"

if [ -z "$CC" ] ; then
	lister=$(dirname "$0")/list-cuda-cc
	read CC card <<-AVOID_SUBSHELL_PROBLEM
		$(${lister} | cut -f2- | sort -n | head -1)
	AVOID_SUBSHELL_PROBLEM
	if [ -z "$CC" ] ; then
		CC=3.0
		echo "Unable to determine Compute Capability, assuming ${CC}"  >&2
		card="fallback, default"
	else
		echo "Auto-detected Compute Capability ${CC} (${card})" >&2
	fi
else
	card=assigned
fi

# transform CC from major.minor to their concatenation
CC=$(echo ${CC} | tr -d .)

echo "#define COMPUTE $CC /* $card */"
