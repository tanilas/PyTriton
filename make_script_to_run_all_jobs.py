import os
cwd=os.getcwd()

filename="slurm_run_jobs_auto.sh"

if os.path.exists(filename):
    os.remove(filename)

f= open(filename,"w+")

f.write("#!/bin/bash")
f.write("\n")
f.write("\n")
f.write("chmod 755 "+cwd+"/jobs/*")
f.write("\n")
f.write("#This is the part where we submit the jobs that we cooked")
f.write("\n")
f.write('for j in $(ls -1 "'+cwd+'/jobs/");do')
f.write("\n")
f.write('sbatch "'+cwd+'/jobs/"$j')
f.write("\n")
f.write("sleep 0.01")
f.write("\n")
f.write("done")
f.write("\n")
f.write('echo "All jobs submitted"')
f.write("\n")
f.write("#rm slurm-*")
f.write("\n")
f.close()

