#!/bin/bash

bash $HADOOP_HOME/sbin/stop-yarn.sh

ssh $(cat ./yarn-slave) "bash $HADOOP_HOME/sbin/yarn-daemon.sh stop nodemanager; jps"
