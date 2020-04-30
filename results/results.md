
# NAS.py


| Elapsed time  |     Values     |  Help |
|:-----------------------------:|:--------------------------------:|:------------------------------------:|
| examples/sec |  30             | 100000*8/30 = 27e3sec |
| global_step/sec |   0.230493             | 6250/0.23 = 27e3sec = 7,5h |


## Test v2 - local (check if at least learn on the training set)

| Parameters  |     Values     |  Help | 
|:-----------------------------:|:--------------------------------:|:------------------------------------:|
| num_classes |  200             | / |
| num_train_images |   100,000             | / |
| num_eval_images |  9832             | / |
| input_image_size |  64             | / |
| eval_batch_size |  128             | / |
| train_batch_size |  128             | /|
| steps_per_eval |  200             | 0.26 epochs|
| lambda |  0             | no runtime loss |
| train_steps |    6250   |   eq. 8 epochs |
| warmup_steps | 0 |    no dropout |
| dropout_rate |    (0.5)   |   not used here |
| label_smoothing | 0.2 |    / |

</br>


| Results  |     top_1_accuracy     |  top_5_accuracy  |
|:-----------------------------:|:--------------------------------:|:------------------------------------:|
| Train/chkpt0|                4,79e-3 | 2,4e-2 |
| Train/chkpt200 |                4,69e-3 | 2,36e-2 |
| Train/chkpt400|                5,5e-3 | 2,68e-2 |
| Train/chkpt600|                1,9e-2 | 8,3e-2 |
| Eval/chkpt600|                No improvement | No improvement |


EARLY STOP : Checkpoint 600 (0,768 epochs)

---------------------------



# Main.py


## Test v1 (GPU temporary borrowed - 12h)


### Architecure : Full 5x5-6 MBConvBlocks

| Elapsed time  |     Values     |  Help |
|:-----------------------------:|:--------------------------------:|:------------------------------------:|
| examples/sec |  1200             | /|
| global_step/sec |   1.15             | / |


| Parameters  |     Values     |  Help | 
|:-----------------------------:|:--------------------------------:|:------------------------------------:|
| num_classes |  200             | / |
| num_train_images |   100,000             | / |
| num_eval_images |  9832             | / |
| input_image_size |  64             | / |
| eval_batch_size |  1024             | / |
| train_batch_size |  1024             | /|
| steps_per_eval |  200             | 0.26 epochs|
| lambda |  0             | no runtime loss |
| train_steps |    35,000  |   eq. 350 epochs |
| dropout_rate |    0.2   |   for final output layer |
| label_smoothing | 0.1 |    / |



#### verification:

| Results  |     top_1_accuracy     |  top_5_accuracy  |
|:-----------------------------:|:--------------------------------:|:------------------------------------:|
| CORRECTED Eval/chkpt35000|                ≃ 0.5 |  ≃0.73 |
| Train/chkpt35000|                0.8 | 0.9 |


## Test v2 (Increased dropout_rate - 1h)


### Architecure : Full 5x5-6 MBConvBlocks



| Parameters  |     Values     |  Help | 
|:-----------------------------:|:--------------------------------:|:------------------------------------:|
| num_classes |  200             | / |
| num_train_images |   100,000             | / |
| num_eval_images |  9832             | / |
| input_image_size |  64             | / |
| eval_batch_size |  1024             | / |
| train_batch_size |  1024             | /|
| steps_per_eval |  200             | 0.26 epochs|
| lambda |  0             | no runtime loss |
| **train_steps** |    **3500**  |   eq. 35 epochs |
| **dropout_rate** |    **0.6**   |   / |
| label_smoothing | 0.1 |    / |


| Results  |     top_1_accuracy     |  top_5_accuracy  |
|:-----------------------------:|:--------------------------------:|:------------------------------------:|
| CORRECTED Eval/chkpt3500|                ≃ 0.38 |  ≃0.65 |



