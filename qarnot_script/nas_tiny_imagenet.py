#!/usr/bin/env python3

import sys
import qarnot
import os

# Create a connection, from which all other objects will be derived
# Enter client token here
conn = qarnot.Connection('/Users/roxanefischer/Desktop/single_path_nas/single-path-nas/qarnot_script/samples.conf')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--filename',
                    help=('The filename containing the input data.  '
                            'If a filename is not given then data is '
                            'read from stdin.'), default='nas_search')
args = parser.parse_args()

# Create a task
#task = conn.create_task(args.filename, 'docker-nvidia-network', 1)
task = conn.create_task(args.filename, 'docker-batch', 5)

# Store if an error happened during the process
error_happened = False
try:
    import pdb 
    #pdb.set_trace()
    # Create a resource bucket and add an input file
    #python files
    input_bucket = conn.retrieve_bucket('input-resource-nas')
    input_bucket.sync_directory("/Users/roxanefischer/Desktop/single_path_nas/single-path-nas")

    image_net = conn.retrieve_bucket('tiny-imagenet')
    output_bucket = conn.retrieve_bucket('output-nas')
    #output_bucket = conn.retrieve_bucket('output-nas-tiny-imagenet')

    # Create a result bucket and attach it to the task

    task.resources.append(input_bucket)
    task.resources.append(image_net)
    task.resources.append(output_bucket)
   
    task.results = output_bucket

    task.constants["DOCKER_REPO"] = "tensorflow/tensorflow"
    task.constants["DOCKER_TAG"] = "1.12.0-gpu"


    task.constants['DOCKER_CMD'] = "/bin/bash -c \"lambda_val=0.0; python /job/nas-search/search_main.py \
                                                            --use_tpu=False \
                                                            --data_dir=/job/image_net \
                                                            --model_dir=/job/models/lambda-val-${lambda_val} \
                                                            --export_dir=None \
                                                            --mode=train_and_eval \
                                                            --train_steps=785 \
                                                            --warmup_steps=491 \
                                                            --input_image_size=64 \
                                                            --eval_batch_size=1024 --train_batch_size=1024 \
                                                            --num_train_images=100000 --num_eval_images=9832 \
                                                            --num_label_classes=200 \
                                                            --transpose_input=True \
                                                            --steps_per_eval=80 --iterations_per_loop=10000 \
                                                            --momentum=0.9 \
                                                            --moving_average_decay=0.9999 --weight_decay=1e-5 \
                                                            --label_smoothing=0.3 --dropout_rate=0.2 --runtime_lambda_val=${lambda_val}\""

    # Submit the task to the Api, that will launch it on the cluster
    task.submit()

    #8 epoches = 781 train steps
    # for 8 epochs with 781 step : end dropout at 491 steps
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
        #task.download_results('output-nas-tiny-imagenet')
        task.download_results('output-nas')

finally:
    # Exit code in case of error
    if error_happened:
        sys.exit(1)
