# AdaptPSOFL: Adaptive Particle Swarm Optimization-Based Layer Offloading Framework for Federated Learning

In this repository is contained the official code of our paper : AdaptPSOFL: Adaptive Particle Swarm Optimization-Based Layer Offloading Framework for Federated Learning


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

Particle Swarm Optimization was partially adopted from https://machinelearningmastery.com/a-gentle-introduction-to-particle-swarm-optimization/

Should our paper have been of use to you. Kindly cite the paper as follows : 

Verma, R., Benedict, S. (2023). AdaptPSOFL: Adaptive Particle Swarm Optimization-Based Layer Offloading Framework for Federated Learning. In: Shakya, S., Tavares, J.M.R.S., Fernández-Caballero, A., Papakostas, G. (eds) Fourth International Conference on Image Processing and Capsule Networks. ICIPCN 2023. Lecture Notes in Networks and Systems, vol 798. Springer, Singapore. https://doi.org/10.1007/978-981-99-7093-3_40
```
@InProceedings{10.1007/978-981-99-7093-3_40,
author="Verma, Rachit
and Benedict, Shajulin",
editor="Shakya, Subarna
and Tavares, Jo{\~a}o Manuel R. S.
and Fern{\'a}ndez-Caballero, Antonio
and Papakostas, George",
title="AdaptPSOFL: Adaptive Particle Swarm Optimization-Based Layer Offloading Framework for Federated Learning",
booktitle="Fourth International Conference on Image Processing and Capsule Networks",
year="2023",
publisher="Springer Nature Singapore",
address="Singapore",
pages="597--610",
isbn="978-981-99-7093-3"
}

```

We may be reached at : 
rachitverma.ea@gmail.com
