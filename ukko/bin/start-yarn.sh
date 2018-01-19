#!/bin/bash

bash $HADOOP_HOME/sbin/start-yarn.sh

ssh $(cat ./yarn-slave) "bash $HADOOP_HOME/sbin/yarn-daemon.sh start nodemanager; jps"
