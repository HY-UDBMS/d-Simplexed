#!/bin/bash

ssh ukko086.hpc.cs.helsinki.fi "dev/hadoop-2.8.1/sbin/hadoop-daemon.sh start datanode; jps;"
ssh ukko152.hpc.cs.helsinki.fi "dev/hadoop-2.8.1/sbin/hadoop-daemon.sh start datanode; jps;"
ssh ukko153.hpc.cs.helsinki.fi "dev/hadoop-2.8.1/sbin/hadoop-daemon.sh start datanode; jps;"
ssh ukko163.hpc.cs.helsinki.fi "dev/hadoop-2.8.1/sbin/hadoop-daemon.sh start datanode; jps;"
ssh ukko173.hpc.cs.helsinki.fi "dev/hadoop-2.8.1/sbin/hadoop-daemon.sh start datanode; jps;"
ssh ukko175.hpc.cs.helsinki.fi "dev/hadoop-2.8.1/sbin/hadoop-daemon.sh start datanode; jps;"
