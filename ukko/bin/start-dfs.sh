#!/bin/bash

# starts namenode and secondarynamenode
bash $HADOOP_HOME/sbin/start-dfs.sh

# start single datanode
bash $HADOOP_HOME/sbin/hadoop-daemon.sh start datanode

jps
