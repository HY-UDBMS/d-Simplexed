#!/bin/bash

$HADOOP_HOME/sbin/stop-yarn.sh
jps
$HADOOP_HOME/sbin/start-yarn.sh
