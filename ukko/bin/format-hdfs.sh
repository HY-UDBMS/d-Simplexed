#!/bin/bash

bash ./stop-dfs.sh

rm -rf ~/tmp-hdfs

hdfs namenode -format
