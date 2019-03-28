#!/bin/sh

# Run all problems in GPUSPH, for the given number of iterations, creating simulations with the specified suffix
# Syntax: run-all-problems [suffix [maxiter [GPUSPH options]]]
# default suffix is 'reference', unless the environment variable GPUSPH_DEVICE includes a comma,
#	in which case the default suffix is mgpu_reference
# default maxiter is 1000
# if they are present as arguments but empty, they will be kept at the default values
# TODO generate repacking references where supported too

abort() {
	echo "$@" >&2
	exit 1
}

sfx=reference
maxiter=1000

case "$GPUSPH_DEVICE" in
*,*) sfx=mgpu-reference
esac

if [ 0 -lt "$#" ] ; then
	[ -z "$1" ] || sfx="$1"
	shift
	if [ 0 -lt "$#" ] ; then
		[ -z "$1" ] || maxiter="$1"
		shift
	fi
fi

for problem in $(make list-problems) ; do
	echo "Running test ${problem} ..."
	outdir="tests/${problem}_${sfx}"
	if [ -d "$outdir" ] ; then
		echo "$outdir exists, skipping"
		continue
	fi
	make $problem && ./$problem --dir "$outdir" --maxiter $maxiter "$@" || abort "Failed! ($problem)"
done
