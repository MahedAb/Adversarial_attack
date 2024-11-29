### FSGM Implementation

This is an implementation for FSGM (fast sign gradient method) attack. 
See here for more information about this attack: 
https://christophm.github.io/interpretable-ml-book/adversarial.html.

### How to run it
The required dependencies can be found in requirements.txt. 
Run the main.py, make sure the image path in the main.py is correct and the desired
target class and epsilon are selected. 
Two methods, FGSM and iterative-FGSM are implemented which can be selected by passing the method argument.
The default is fgsm, use "--method ifgsm" for iterative-FGSM.

### Things that can be improved:

1- Accommodating for CUDA

2- Manage parameters and hyperparameters better, perhaps by using a config file.

3- The way the model takes an input

4- Maybe have utils file instead of putting everything in main

5- Unittest for different functions