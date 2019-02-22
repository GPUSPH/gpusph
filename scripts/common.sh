#!/bin/sh

# Common functions and methods for the other scripts

problem_list="$*"
[ -z "$problem_list" ] && problem_list="$(make list-problems)"
# TODO support syntax such as ~ProblemName to _disable_ individual problems

failed=

mgpu=0

script="$(basename "$0" .sh)"

title_prefix=
title_phase=
problem=

# Set terminal title
set_title() {
	local msg="$*"
	printf "\e]2;${msg:-$title_prefix $title_phase}\a"
}

set_title_prefix() {
	title_prefix="$*"
	set_title
}
set_title_phase() {
	title_phase="$*"
	set_title
}

add_failed() {
	local p="$1"
	local r="$2"
	failed="${failed}${p:-problem} (${r:-title_phase})\n"
}

abort() {
	echo "$@" >&2
	exit 1
}

count_problems() {
	set -- $problem_list
	printf "%d" $#
}

i_of_n() {
	local i=$1
	local n=$2
	local str=""
	while [ $i -gt 0 ] ; do
		str="$str#"
		i=$((i-1))
	done
	printf "%-${2}s" "$str"
}

_bold="$(tput bold)"
_sgr0="$(tput sgr0)"

# Call the first argument (a function) for each problem
for_each_problem_silent() {
	for problem in $problem_list ; do
		"$1" "$problem"
	done
}

# Call the first argument (a function) for each problem, reporting progress
# The src variable will be set to the problem source, if available
for_each_problem() {
	local cmd
	cmd="$1"

	n=$(count_problems)
	i=0

	for problem in $problem_list ; do
		i=$((i+1))
		pattern="[$(i_of_n $i $n)]"
		boldproblem="${_bold}${problem}${_sgr0}"
		src="$(find -L src/problems -name ${problem}.cu)"

		set_title_prefix "$script: [${i}/${n}] $problem"
		set_title_phase build

		if [ -z "$src" ] ; then
			add_failed "$problem" "source not found"
		else
			echo "$pattern Testing ${boldproblem} ($src) ..."
			if ! make $problem ; then
				add_failed "$problem" "build"
			else
				"$cmd" "$problem"
			fi
		fi
		[ "x$CHECK_ALL" = x0 ] && [ -n "$failed" ] && break
	done

	set_title "$script"

	if [ -z "$failed" ] ; then
		echo "All OK"
		exit 0
	else
		echo "Failures:\n${failed}"
		exit 1
	fi
}
