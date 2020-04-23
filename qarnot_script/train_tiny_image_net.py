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
                            'read from stdin.'), default='train_model')
args = parser.parse_args()

# Create a task
task = conn.create_task(args.filename, 'docker-nvidia-network', 1)

# Store if an error happened during the process
error_happened = False
try:
    # Create a resource bucket and add an input file
    #python files
    input_bucket = conn.retrieve_bucket('input-resource-nas')
    input_bucket.sync_directory("/Users/roxanefischer/Documents/cours/3A/Stage_ML/single-path-nas/single-path-nas")

    image_net = conn.retrieve_bucket('tiny-imagenet')
    output_bucket = conn.retrieve_bucket('output')
    #output_bucket = conn.retrieve_bucket('output-tiny-imagenet')

    # Create a result bucket and attach it to the task

    task.resources.append(input_bucket)
    task.resources.append(image_net)
    task.resources.append(output_bucket)
   
    task.results = output_bucket

    task.constants["DOCKER_REPO"] = "tensorflow/tensorflow"
    task.constants["DOCKER_TAG"] = "1.12.0-gpu"


    task.constants['DOCKER_CMD'] = "/bin/bash -c \"lambda_val=0.020; python /job/train-final/main.py \
                                                            --use_tpu=False \
                                                            --data_dir=/job/image_net \
                                                            --parse_search_dir=/job/models \
                                                            --model_dir=/job/final \
                                                            --num_label_classes=200 \
                                                            --num_train_images=100000 --num_eval_images=10000 \
                                                            --eval_batch_size=1024 --train_batch_size=1024 \
                                                            --input_image_size=64 \
                                                            --train_steps=35000 --steps_per_eval=2000 --iterations_per_loop=35000\""


    
    
    # Submit the task to the Api, that will launch it on the cluster
    task.submit()


    #8 epoches = 781 train steps
    #350 epochs = 34 000 train steps

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
        done = task.wait(2)

        try:
            task.snapshot(120)
        except Exception as err:
            print()

        # Display fresh stdout / stderr
        sdout = task.fresh_stdout()
        if sdout!='':
            print()
            print(sdout)
        
        sys.stdout.write(task.fresh_stdout())
        sys.stderr.write(task.fresh_stderr())

    # Display errors on failure
    if task.state == 'Failure':
        print("** Errors: %s" % task.errors[0])
        error_happened = True

    else:
        #task.download_results('output-tiny-imagenet')
        task.download_results('output')

finally:
    # Exit code in case of error
    if error_happened:
        sys.exit(1)
