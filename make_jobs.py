from makefile import make_jobs
import os

os.system("rm jobs/*")
os.system("rm logs/*")

jobind=0;

for permi in range(2):        
        jobind=jobind+1
        command="python classify.py -p "+str(permi)
        make_jobs(command, jobind)