#!/bin/bash

# stop namenode and secondarynamenode
bash $HADOOP_HOME/sbin/stop-dfs.sh

# stop sole datanode
bash $HADOOP_HOME/sbin/hadoop-daemon.sh stop datanode

jps
