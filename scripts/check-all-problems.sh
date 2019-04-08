#!/bin/sh

# Run each problem for maxiter iterations and check against the reference
# Syntax: check-all-problems.sh [problem ...]
# default reference suffix is 'reference', unless the environment variable GPUSPH_DEVICE includes a comma,
#	in which case the default suffix is mgpu-reference
# default maxiter is 1000
# To change the default refence and check suffix or maxiter, define the sfx, ref and/or maxiter
# environment variables.
# If the environment variable CHECK_ALL is defined and set to 0, checking will stop at the first problem
# that fails, otherwise all problems will be checked.
# To run GPUSPH with additiona options, set the GPUSPH_OPTIONS environment variable
# TODO support repacking in our tests

. scripts/common.sh

_sfx=check
_ref=reference

case "$GPUSPH_DEVICE" in
*,*)
	_sfx=mgpu-check
	_ref=mgpu-reference
	mgpu=1
esac

sfx="${sfx:-$_sfx}"
ref="${ref:-$_ref}"
maxiter="${maxiter:-1000}"

# Check that we have all the references first
has_dir() {
	local problem=$1
	refdir="tests/${problem}_${ref}"
	[ -d "$refdir" ] || abort "Reference directory $refdir for problem $problem not found —did you forget to run-all-problems?"
}

check_problem() {
	local problem="$1"

	outdir="tests/${problem}_${sfx}"
	refdir="tests/${problem}_${ref}"
	rm -rf "$outdir"

	set_title_phase run
	if ! ./$problem --dir "$outdir" --maxiter $maxiter $GPUSPH_OPTIONS ; then
		add_failed "$problem" run
		return
	fi

	set_title_phase diff
	if ! diff -q "${outdir}/data" "${refdir}/data" ; then
		add_failed "$problem" diff
		return
	fi

	# We're done if not multi-GPU
	[ $mgpu -ne 1 ] && return

	# In the multi-GPU case, also run with --striping, which should still give the same result

	set_title_phase striping run
	if ! ./$problem --dir "$outdir" --maxiter $maxiter --striping $GPUSPH_OPTIONS ; then
		add_failed "$problem" "striping run"
		return
	fi

	if ! diff -q "${outdir}/data" "${refdir}/data" ; then
		add_failed "$problem" "striping diff"
	fi
}

for_each_problem_silent has_dir
for_each_problem check_problem
