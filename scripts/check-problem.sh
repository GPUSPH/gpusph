#!/bin/sh

# Run a specific problem for maxiter iterations and check against the reference
# Syntax: check-problem problem [reference_suffix] [maxiter]
# default reference_suffix is 'reference', unless the environment variable GPUSPH_DEVICE includes a comma,
#	in which case the default suffix is mgpu_reference
# default maxiter is 1000
# if they are present as arguments but empty, they will be kept at the default values

abort() {
	echo "$@" >&2
	exit 1
}

sfx=check
ref=reference
maxiter=1000

mgpu=0

case "$GPUSPH_DEVICE" in
*,*)
	sfx=mgpu-check
	ref=mgpu-reference
	mgpu=1
esac

problem="$1"
shift

if [ 0 -lt "$#" ] ; then
	[ -z "$1" ] || ref="$1"
	shift
	if [ 0 -lt "$#" ] ; then
		[ -z "$1" ] || maxiter="$1"
		shift
	fi
fi

echo "Testing ${problem} ..."
outdir="tests/${problem}_${sfx}"
refdir="tests/${problem}_${ref}"

[ -d "$refdir" ] || abort "Reference directory $refdir for problem $problem not found"

rm -rf "$outdir"

make $problem && ./GPUSPH --dir "$outdir" --maxiter $maxiter || abort "$problem failed!"
diff -q "${outdir}/data" "${refdir}/data" || abort "$problem differs!"

if [ $mgpu -eq 1 ] ; then
	make $problem && ./GPUSPH --dir "$outdir" --maxiter $maxiter --striping || abort "$problem failed! (striping)"
	diff -q "${outdir}/data" "${refdir}/data" || abort "$problem differs! (striping)"
fi

echo "OK"
