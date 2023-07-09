# PSOFL

We have posted the code for our paper in this repository. This repository consists of code to execute three frameworks, 
classic Federated Learning, FedAdapt and our original Framework PSO FL. 

In order to reproduce the results of our paper, setup 10
Virtual Machines with the configurations that have been discussed in the paper. 

The configurations of the server are relatively immaterial as long as it is significantly more powerful than the VMs and if it is kept uniform across evaluation of all the 
frameworks

Next, on each client replicate the Client folder in the folder associated with each framework. Next update the index and client ip
of the client in each file. 

We have used the sar command line utility to track the CPU utilisation of each framework. 

You may have to modify the code to fetch the percentage of time spent by the CPU in user mode based on the format in which sar works on your 
machine. This will appear in the user column of ```sar -u 1 ```

This includes making modifications to utils.py and to time.strftime("%I:%M:%S") wherever it appears in the code. 

Kindly contact us if you have any issues with this or if you are unable to deploy our code on your systems. 

We may be reached at : 
rachitverma.ea@gmail.com
