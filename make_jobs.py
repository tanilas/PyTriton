from makefile import make_jobs
import os

#os.system("rm jobs/*")

jobind=0;
for window in range(48):
    for permi in range(2):
        for clini in range(2):
            jobind=jobind+1
            command="python classify.py -w "+str(window)+" -p "+str(permi)+" -c "+str(clini)
            make_jobs(command, jobind)