#!/bin/sh

# Verify that it is possible to resume a specific problem producing the same result.
# Syntax: check-resume [problem ...]
# If no problem is specified, all problems are checked.
# if the environment variable CHECK_ALL is defined and set to 0, checking will
# stop at the first problem that fails (if more than one is to be tested).

sfx=resume-check
ref=resume-reference
maxiter=1000

mgpu=0

case "$GPUSPH_DEVICE" in
*,*)
	sfx=${sfx}-mgpu
	ref=${ref}-mgpu
	mgpu=1
esac

failed=

add_failed() {
	failed="${failed}$1 ($2)\n"
}

abort() {
	echo "$@" >&2
	exit 1
}

check_problem() {
	problem="$1"
	shift

	src="$(find src/problems -name ${problem}.cu)"

	echo "Testing ${problem} ($src)..."

	# finish after 3 writes
	tend="$(grep '[^/]add_writer' "$src" | cut -f2 -d, | cut -f1 -d\) | awk -e '{ printf "%.9g", $1*3}')"

	if [ 0 = "$tend" ] ; then
		add_failed "$problem" "Write frequency / tend 0 is not supported"
		return
	fi

	outdir="tests/${problem}_${sfx}"
	refdir="tests/${problem}_${ref}"

	rm -rf "$outdir" "$refdir"

	if make $problem ; then
		if ./$problem --dir "$refdir" --tend $tend ; then
			# Get the last 3 hotfiles from ref
			ref_hotfiles3="$(find "$refdir" -name hot\* | sort -n | tail -n 3)"

			# The oldest of these is the new starting point
			third_last="$(echo "$ref_hotfiles3" | head -n 1)"

			echo "Resuming from '$third_last'"

			if ./$problem --resume "$third_last" --dir "$outdir" --tend $tend ; then
				# Get the last 3 hofiles from out
				out_hotfiles3="$(find "$outdir" -name hot\* | sort -n | tail -n 3)"

				# Now we want to compare the corresponding hotfiles, which means that we need to iterate
				# concurrently on the two lists. To achieve this, we put the ref hotfiles in the arguments array,
				# iterate over the out hotfiles, and shift at each iteration
				set -- $ref_hotfiles3

				for hot_out in $out_hotfiles3 ; do
					hot_ref="$1"
					shift
					if ! diff -q "$hot_ref" "$hot_out" ; then
						add_failed "$problem" "$hot_ref != $hot_out"
						break
					fi
				done

			else
				add_failed "$problem" "second run"
			fi

		else
			add_failed "$problem" "first run"
		fi
	else
		add_failed "$problem" build
	fi
}

problem_list="$*"
[ -z "$problem_list" ] && problem_list="$(make list-problems)"

for problem in $problem_list ; do
	check_problem $problem
	[ "x$CHECK_ALL" = x0 ] && [ -n "$failed" ] && break
done

if [ -z "$failed" ] ; then
	echo "All OK"
	exit 0
else
	echo "Failures:\n${failed}"
	exit 1
fi
