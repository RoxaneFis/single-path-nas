
# Local - Tiny Imagenet


| Elapsed time  |     Values     |  Help |
|:-----------------------------:|:--------------------------------:|:------------------------------------:|
| examples/sec |  30             | 100000*8/30 = 27e3sec |
| global_step/sec |   0.230493             | 6250/0.23 = 27e3sec = 7,5h |


## Test v2

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
| **BAD IDEA : decay_epochs**|   **0.29**  |   **lr updated/230 stps** | 

</br>


| Results  |     top_1_accuracy     |  top_5_accuracy  |
|:-----------------------------:|:--------------------------------:|:------------------------------------:|
| Train/chkpt0|                4,79e-3 | 2,4e-2 |
| Train/chkpt200 |                4,69e-3 | 2,36e-2 |
| Train/chkpt400|                5,5e-3 | 2,68e-2 |
| Train/chkpt600|                1,9e-2 | 8,3e-2 |

STOP : Checkpoint 600 (0,768 epochs)

---------------------------

## Test v3 (change learning-rate decay :no diff v2)


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
| **decay_epochs** |    **Default 2.4**  |   lr updated/1562 stps| 

</br>



| Results  |     top_1_accuracy     |  top_5_accuracy  |
|:-----------------------------:|:--------------------------------:|:------------------------------------:|
| Train/chkpt0|                4,49e-3 | 2,37e-2 |
| Train/chkpt200 |                6,25e-3 | 2,46e-2 |
| Train/chkpt400|                4,20e-3 | 2,46e-2 |
| Train/chkpt600|                2,16e-2 | 8,68e-2 |

STOP : Checkpoint 600 (0,768 epochs)