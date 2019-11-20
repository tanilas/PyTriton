- classify_functions: defining the main function to classify data. Loads data prepares, splits to training and testing, prepares trains and evaluates the network

- classify.py: the function to use from the command line to run the classification with different arguments

- makefile.py: the function that creates a job

- make_jobs: the file that creates all the jobs to run, saving them to the jobs folder

- make_scripts_to_run_all_jobs.py: Function to make the bash script to run all the jobs in the jobs directory

- read_results.py: Reading the result accuracies from the results folder and storing them in a mat file called "acc.mat"

- visualize_results.py: Plot results stored in "acc.mat"