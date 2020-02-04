import os
def make_jobs(command_to_run="python test.py",jobind=1):
    if not os.path.exists("jobs"):
        os.makedirs("jobs")

    if not os.path.exists("logs"):
        os.makedirs("logs")
    filename="jobs/job"+str(jobind)+".sh"

    if os.path.exists(filename):
        os.remove(filename)

    f= open(filename,"w+")

    f.write("#!/bin/bash")
    f.write("\n")
    f.write("\n")
    f.write("#SBATCH -p batch")
    f.write("\n")
    f.write("#SBATCH -t 9:59:59")
    f.write("\n")
    f.write("#SBATCH --array=1-550")
    f.write("\n")
    f.write("#SBATCH -o ./logs/log_"+str(jobind)+"_%a")
    f.write("\n")
    f.write("#SBATCH --qos=normal")
    f.write("\n")
    f.write("#SBATCH --mem-per-cpu=5000")
    f.write("\n")
    f.write("\n")
    cwd=os.getcwd()
    f.write("cd "+cwd+"")
    f.write("\n")
    f.write(command_to_run)
    f.write("\n")
    f.close()


if __name__ == '__main__':
    make_jobs("python classify.py -p 0 -w 5 -c 0",1)