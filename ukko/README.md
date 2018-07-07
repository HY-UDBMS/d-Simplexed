A set of utility scripts made for interacting with hadoop/yarn/spark on the (old) Ukko environment.  

In `bin/` we have utility scripts for interacting with the environment, and with `batch-v2.sh`, we have built a simple bulk-submitter for spark jobs on the hibench framework that should be run within something like `screen`, which will not kill the bash session on the user has exited.  It is useful for queing up a lot of jobs to run overnight without having to manually submit each one. 
