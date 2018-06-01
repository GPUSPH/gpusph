#!/bin/sh

# Run each problem for maxiter iterations and check against the reference
# Syntax: check-all-problems.sh [reference_suffix] [maxiter]
# default reference_suffix is 'reference'
# default maxiter is 1000

abort() {
	echo "$@" >&2
	exit 1
}

sfx=check
ref="$2"
maxiter="$3"

[ -z "$ref" ] && ref=reference
[ -z "$maxiter" ] && maxiter=1000

rm -rf tests/*_${sfx}

for problem in $(make list-problems) ; do
	case $problem in
	CompleteSaExample)
		echo "Skipping ${problem}, known non-reproducible"
		continue
		;;
	esac
	echo "Testing ${problem} ..."
	outdir="tests/${problem}_${sfx}"
	refdir="tests/${problem}_${ref}"
	make $problem && ./GPUSPH --dir "$outdir" --maxiter $maxiter || abort "Failed!"
	diff -q "${outdir}/data" "${refdir}/data" || abort "$problem differs!"
done
