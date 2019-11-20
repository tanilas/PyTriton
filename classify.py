import os
import argparse
from classify_functions import classify

if not os.path.exists("results"):
    os.makedirs("results")

parser = argparse.ArgumentParser(description='Classify fMRI data.')
parser.add_argument("-w", "--window", dest="window", default=1, type=int, help="Set a window from 1 to 48")
parser.add_argument("-p", "--permutation", dest="permutation", default=0, type=int, help="Should it be a permutation")
parser.add_argument("-c", "--clinical_data", dest="clinical_data", default=1,type=int, help="Should include clinical data?")

#parser.add_argument("-p", "--permutation", dest="permutation", default=False, type=bool, help="Should it be a permutation")
#parser.add_argument("-c", "--clinical_data", dest="clinical_data", default=True,type=bool, help="Should include clinical data?")

args = parser.parse_args()
print("Window chosen: "+str(args.window))
print("Permutation chosen: "+str(args.permutation))
print("Clinical data chosen: "+str(args.clinical_data))

classify(args.permutation, args.clinical_data,args.window)