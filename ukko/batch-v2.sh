#!/bin/bash

# this file should sit in $HIBENCH_HOME

conf=$1

runscript=$(head -1 $1)

echo "Config: $conf"
echo "Running: $runscript"

IFS="
"
for trial in $(tail -n +2 $conf); do
	numexec=$(echo $trial | cut -d' ' -f1)
	numcores=$(echo $trial | cut -d' ' -f2)
	execmem=$(echo $trial | cut -d' ' -f3)

	echo "--------------------------------------------------"
	echo "hibench.yarn.executor.num=$numexec"
	echo "hibench.yarn.executor.cores=$numcores"
	echo "spark.executor.memory=$execmem"

	# update spark.conf on the fly
	sed -i \
	-e "/hibench.yarn.executor.num/ c\hibench.yarn.executor.num    $numexec" \
	-e "/hibench.yarn.executor.cores/ c\hibench.yarn.executor.cores    $numcores" \
	-e "/spark.executor.memory/ c\spark.executor.memory    $execmem" \
	conf/spark.conf

	runs=2
	for i in $(seq $runs)
	do
		echo "Start ($i/$runs) $trial" >> report/hibench.report
		bash $runscript
	done
done
 
cat report/hibench.report
