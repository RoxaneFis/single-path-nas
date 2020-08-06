This markdown explains how to use several services on AWS.

# EC2 Services


## Create an instance or a volume : 

An aws instance gives access to cloud computing ressources.

Instance : 
https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html

! When choosing the AMI, take care of choosing one adapted to Machine Learning (contains tensorflow, pytorch...) !

A volume can be attached to an instance in order to store some data (the data will not be lost when the instance is stopped)

Volume : 
https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-creating-volume.html

## Launch the expirements on HAS

### A. Start the instace :

We used g4dn.xlarge instances (NVIDIA T4 GPUs). The aim of this section is to start an instance, connect to it and start a jupyter notebook in order to launch some experiences. Some volumes must be attached in order to save the data.


1) EC2 -> Instances -> Actions -> State of the instance -> Start
2) Retrieve the public DNS of the running instance (available in its *description*). It must look like : </br>
   PUBLIC DNS = "ec2-34-216-40-133.us-west-2.compute.amazonaws.com"
   
3) In a terminal, connect remotly to the instance by ssh (you need to have access to the path of the permission key created with the instance). The syntax is as follow : </br>
**ssh -i <PATH_PERMISSION_KEY> ubuntu@<PUBLIC_DNS>**
</br> 

ex: ssh -i /Users/.../keys/retrieve_vol.pem ubuntu@ec2-34-216-40-133.us-west-2.compute.amazonaws.com  
(use "ubuntu" or "ec2-user" depending of the instance).

4) Answer *yes* if the identity of the host is not established. You are now connected to your instance.



###  B. Attach a Volume : 

1) type "lsblk" in the instance terminal. You will have access to all your available volumes (*NAME* column). 
   </br> FIXME : To recognize which volum to be attached, I identified them by their sizes (the predictor data is in a 5G volume whereas processed Imagenet is stored in a 600G volume.

   
2)  To attach a volume to a folder of your instance, type : </br>
**sudo mount /dev/<NAME_OF_THE_VOLUME> <FOLDER_TO_BE_ATTACHED>**

ex: sudo mount /dev/nvme1n1 /data_predictor


Now you can have access to the data in your volume and write in it. When you will close the instance the volume will be automatically saved.

### C. Connect to a Jupyter Notebook

1) Create a detached session : type "screen" (detached this session : CTRL + A + D)
2) Activate your environment : "source activate tensorflow2_p36"
3) Before starting the notebook place you in the root folder : "cd /"
4) Open a notebook : **"jupyter notebook --ip=0.0.0.0"**. </br>
To do so you need to configure the security groups of your instance and the jupyter notebook configurations : 
https://medium.com/@margaretmz/setting-up-aws-ec2-for-running-jupyter-notebook-on-gpu-c281231fad3f


1) In your web browser connect locally to your jupyter :
   http://<PUBLIC_DNS>:8888 </br>
Ex : http://ec2-34-216-40-133.us-west-2.compute.amazonaws.com:8888
</br> You will have to use the password entered during the jupyter configuration.

If you connect to the already created g4dn.xlarge you should have access in the root folder to :
+ "data_predictor" (attached volume where the experimental data are stored)
+ "single-path-nas/HAS/" (see README.md)



# S3 Services

S3 Services store some data.
When you choose to upload some data, make sure it's in the same region as your instance. 

"imagenet-rf" contains Imagenet treated as tf-records in a "train" and a "validation" folders (used in single-path-nas). 


# Billing Services

Billing services allow to monitor your expenses. You can have access to your costs in "Costs explorer"