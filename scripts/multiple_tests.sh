#!/bin/bash
export DELTAP=0.003
export EXTRAFLAGS="--nosave --tend 1.5"
export PREFIX="linXZY"

for NUMGPUS in $(seq 6 -1 1)
do

	export GPUS=$(seq -s ',' 0 $(( $NUMGPUS - 1 )) )
	export NAME=${PREFIX}_${NUMGPUS}gpus_dp${DELTAP}
	export DIRNAME=./tests/$NAME
	export LOGNAME=$NAME.log

	# prepare command
	export COMMAND="./GPUSPH --deltap ${DELTAP} --device $GPUS --dir $DIRNAME ${EXTRAFLAGS}"

	# prepare dir and logfile
	rm -rf "$LOGNAME" "$DIRNAME" &> /dev/null
	echo "$COMMAND" > "$LOGNAME"
	
	#echo "Running $COMMAND"
	echo "Running on $NUMGPUS GPUs..."
	
	# run, baby, run!
	(time $COMMAND) &>> "$LOGNAME"

	# profile with gmon if -pg was active
	if test -e ./gmon.out
	then
		gprof ./GPUSPH ./gmon.out &> analysis_$NAME.txt
		mv gmon.out gmon.out.$NAME
	fi

	# print MIPPS for quick overview
	grep MIPPS "$LOGNAME" | tail -n 1
done

echo End

