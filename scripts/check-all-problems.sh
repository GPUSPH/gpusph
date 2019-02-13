#!/bin/sh

# Run each problem for maxiter iterations and check against the reference
# Syntax: check-all-problems.sh [reference_suffix [maxiter [GPUSPH options]]]
# default reference_suffix is 'reference', unless the environment variable GPUSPH_DEVICE includes a comma,
#	in which case the default suffix is mgpu_reference
# default maxiter is 1000
# if they are present as arguments but empty, they will be kept at the default values
# if the environment variable CHECK_ALL is defined and set to 0, checking will stop at the first problem
# that fails, otherwise all problems will be checked

failed=

add_failed() {
	failed="${failed}$1 ($2)\n"
}

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

if [ 0 -lt "$#" ] ; then
	[ -z "$1" ] || ref="$1"
	shift
	if [ 0 -lt "$#" ] ; then
		[ -z "$1" ] || maxiter="$1"
		shift
	fi
fi

problem_list="$(make list-problems)"

# Check that we have all the references first
for problem in $problem_list ; do
	refdir="tests/${problem}_${ref}"
	[ -d "$refdir" ] || abort "Reference directory $refdir for problem $problem not found —did you forget to run-all-problems?"
done

for problem in $problem_list ; do
	echo "Testing ${problem} ..."
	outdir="tests/${problem}_${sfx}"
	refdir="tests/${problem}_${ref}"
	rm -rf "$outdir"
	if make $problem ; then
		if ./$problem --dir "$outdir" --maxiter $maxiter "$@"; then
			diff -q "${outdir}/data" "${refdir}/data" || add_failed "$problem" diff
		else
			add_failed "$problem" run
		fi
		[ "x$CHECK_ALL" = x0 ] && [ -n "$failed" ] && break
		# In the multi-GPU case, also run with --striping, which should still give the same result
		if [ $mgpu -eq 1 ] ; then
			if ./$problem --dir "$outdir" --maxiter $maxiter --striping "$@"; then
				diff -q "${outdir}/data" "${refdir}/data" || add_failed "$problem" "striping diff"
			else
				add_failed "$problem" "striping run"
			fi
		fi
	else
		add_failed "$problem" build
	fi
	[ "x$CHECK_ALL" = x0 ] && [ -n "$failed" ] && break
done

if [ -z "$failed" ] ; then
	echo "All OK"
	exit 0
else
	echo "Failures:\n${failed}"
	exit 1
fi
