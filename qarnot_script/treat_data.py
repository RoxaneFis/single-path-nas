#!/usr/bin/env python3

import sys
import qarnot
import os

# Create a connection, from which all other objects will be derived
# Enter client token here
conn = qarnot.Connection('/Users/roxanefischer/Desktop/single_path_nas/single-path-nas/samples.conf')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--filename',
                    help=('The filename containing the input data.  '
                            'If a filename is not given then data is '
                            'read from stdin.'), default='machine learning task -- tensorflow')
args = parser.parse_args()

# Create a task
task = conn.create_task(args.filename, 'docker-nvidia-network', 1)

# Store if an error happened during the process
error_happened = False
try:
    # Create a resource bucket and add an input file
    #python files
    input_bucket = conn.retrieve_bucket('raw_imagenet')
    task.resources.append(input_bucket)


    # Create a result bucket and attach it to the task
    #output_bucket = conn.retrieve_bucket('output')
    output_bucket = conn.create_bucket('gcs_imagenet')
    task.results = output_bucket

    task.constants["DOCKER_REPO"] = "tensorflow/tensorflow"
    task.constants["DOCKER_TAG"] = "1.12.0-gpu"

    # Set the command to run when launching the container, by overriding a
    # constant.
    # Task constants are the main way of controlling a task's behaviour
    task.constants['DOCKER_CMD'] = "/bin/bash -c \"lambda_val=0.020; python imagenet_to_gcs.py \
                                                            --gcs_upload=false \
                                                            --raw_data_dir='????'\
                                                            --local_scratch_dir=/job \""
                                                            #--imagenet_username=Roxane \
                                                            #--imagenet_access_key=X@luro92 \""
    # Submit the task to the Api, that will launch it on the cluster
    task.submit()

    # Wait for the task to be finished, and monitor the progress of its
    # deployment
    last_state = ''
    done = False
    while not done:
        if task.state != last_state:
            last_state = task.state
            print("** {}".format(last_state))

        # Wait for the task to complete, with a timeout of 5 seconds.
        # This will return True as soon as the task is complete, or False
        # after the timeout.
        done = task.wait(1)

        # Display fresh stdout / stderr
        if task.stdout()!='':
            print()
            print(task.stdout())
        
        sys.stdout.write(task.fresh_stdout())
        sys.stderr.write(task.fresh_stderr())

    # Display errors on failure
    if task.state == 'Failure':
        print("** Errors: %s" % task.errors[0])
        error_happened = True

    else:
        task.download_results('gcs_imagenet')

finally:
    # Exit code in case of error
    if error_happened:
        sys.exit(1)
