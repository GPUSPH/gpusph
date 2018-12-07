#!/bin/sh

# Run each problem for maxiter iterations and check against the reference
# Syntax: check-all-problems.sh [reference_suffix [maxiter [GPUSPH options]]]
# default reference_suffix is 'reference'
# default maxiter is 1000
# if they are present as arguments but empty, they will be kept at the default values

failed=

add_failed() {
	failed="${failed}$1 ($2)\n"
}


sfx=check
ref=reference
maxiter=1000

if [ 0 -lt "$#" ] ; then
	[ -z "$1" ] || ref="$1"
	shift
	if [ 0 -lt "$#" ] ; then
		[ -z "$1" ] || maxiter="$1"
		shift
	fi
fi

for problem in $(make list-problems) ; do
	echo "Testing ${problem} ..."
	outdir="tests/${problem}_${sfx}"
	refdir="tests/${problem}_${ref}"
	rm -rf "$outdir"
	if make $problem ; then
		if ./GPUSPH --dir "$outdir" --maxiter $maxiter "$@"; then
			diff -q "${outdir}/data" "${refdir}/data" || add_failed "$problem" diff
		else
			add_failed "$problem" run
		fi
	else
		add_failed "$problem" build
	fi
done

if [ -z "$failed" ] ; then
	echo "All OK"
	exit 0
else
	echo "Failures:\n${failed}"
	exit 1
fi
