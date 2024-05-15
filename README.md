# Cancer Image Classifier Using HQCNN(Hybrid Quantum Convolutional Neural Network
>Hey guys,
>Let me introdce HQ-CNN, 
>Hybrid Quantum Convolutional Neural Networks (HQ-CNNs) represent a cutting-edge approach in cancer image classification, leveraging the fusion of quantum computing and classical deep learning techniques.

# Contents:
* [Demo](#Demo)
* [Dataset](Dataset)
* [Used Dependences](#Dependences)
* [Working Tutorial](#Working-Tutorial)
* [What-is-HQCNN?](#What-is-HQCNN?)
* [Referance](#Referances)

# Demo:
![image](https://github.com/Tani2189/LipSyncInsight/assets/96855667/d599766a-1ae8-438e-bf77-d031bcb444ca)


# Dataset:
The dataset used for making this model is taken from online challange website. Heres the link: [data](https://challengedata.ens.fr/participants/challenges/11)
# Dependences:
* Python
* PyTorch
* Qiskit
* Machine Learning
* QCNN
* CNN
* Perceptron

# Working-Tutorial:
>First Step(Baby Step):
![image](https://github.com/Tani2189/Cancer-Image-Classifier-Using-HQCNN/assets/96855667/d6311994-4230-4181-a678-79e49cac07f4)
* Our Inital step is to classify our data based on the column named"class_number". This suggests the data likely has a label indicating its class (e.g., 1 for cancer, 0 for healthy tissue).
>Second Step:
![image](https://github.com/Tani2189/Cancer-Image-Classifier-Using-HQCNN/assets/96855667/c8714943-fb3e-45ed-bc29-bb127bca7764)
* After classifying we'll be training the model with perceptron which is an Unsupervised Machine Learning algorithm and got a pretty good test and train accuracy.
  
![Perceptron Traning confusion matrix](https://github.com/Tani2189/Cancer-Image-Classifier-Using-HQCNN/assets/96855667/82885e39-cd05-404f-abf5-8f70ef09a87c)
* After that we make an confusion matrix using the test data.

# What-is-HQCNN?
* A Hybrid Quantum-Classical Convolutional Neural Network (HQ-CNN) combines the strengths of regular Convolutional Neural Networks (CNNs) with quantum computing to potentially improve feature extraction in image recognition tasks. 

![image](https://github.com/Tani2189/Cancer-Image-Classifier-Using-HQCNN/assets/96855667/4f38eeac-b9fd-44d5-9724-138479dda58d)

# What are the benifit as compared to CNN?
* Improved image recognition accuracy, especially for complex or subtle patterns.
* Faster processing of large datasets (although current quantum computers are still limited).
# Okay so as it is a new technology it should definately have few challenges, what are they?
* Definately, as Quantum computing is in its early stages, and HQ-CNNs are an active research area.
* Implementing and training HQ-CNNs requires expertise in both CNNs and quantum computing.
* Current quantum computers have limited qubit capacity, restricting the complexity of HQ-CNNs.
# Cool Lets's get going!!
>Third Step:

![ZZfeature Map](https://github.com/Tani2189/Cancer-Image-Classifier-Using-HQCNN/assets/96855667/939d331e-d5ca-4c55-9ad1-699d10bd5082)

![ZZFeatureMap Graph or  Second-order Pauli-Z evolution circuit](https://github.com/Tani2189/Cancer-Image-Classifier-Using-HQCNN/assets/96855667/b41e2219-4851-4dac-bd6d-9279d49e8055)
* Here we are initalizing a ZZ feature map and a Second-order Pauli-Z evolution circuit.
* why are we using it, because first reason is Space Efficiency which encodes the data using fewer qubits and second reason is Potential Quantum Advantage which may outperform classical methord for large datasets.
  
# Quantum Entanglement Check:
* There are CNOT gates in the circuits like ZZFeatureMap and RealAmplitudes, thus the QNN makes use of entanglement.
* Probably in 99% of cases there is entanglement but not always (e.g. if the inputs to CNOTs are in only |0> or |1>), thus at the end is parameter dependant.
* We'll be checking the circuit parameters and after that we'll be generating it randomly.
* After that we'll bind the circuit parameter and get the circuit density matrix and purity state of it.
![Density Matrix](https://github.com/Tani2189/Cancer-Image-Classifier-Using-HQCNN/assets/96855667/616d88b9-337b-4206-a9e2-727a57b66489)
![matrix](https://github.com/Tani2189/Cancer-Image-Classifier-Using-HQCNN/assets/96855667/30c72f61-958f-491b-bdb0-eb0136e017d6)

>Forth Step:
* Now With the help of pytorch we'll be Making QCNN model.
  
* Overall, here the code defines a neural network architecture that interacts with a quantum and convolutional neural network for some task, here we're using image classification.
  
![Parameters](https://github.com/Tani2189/Cancer-Image-Classifier-Using-HQCNN/assets/96855667/a5024662-bbf7-412d-9c3e-693497980e46)

* After that we print the accuracy graph with an train data accuracy of 99.82%
  
![HQCNN Accuracy graph](https://github.com/Tani2189/Cancer-Image-Classifier-Using-HQCNN/assets/96855667/f4ddfa5d-1d95-464f-ab2c-55ca9b1b46c3)

* Down Below is the predicticed output:
  
![Prediction using HQCNN](https://github.com/Tani2189/Cancer-Image-Classifier-Using-HQCNN/assets/96855667/c8eae047-3377-4595-a40a-16d67a64e51f)

# Referances:
* [Hybrid quantum-classical Neural Networks with PyTorch and Qiskit: https://qiskit.org/textbook/ch-machine-learning/machine-learning-qiskit](https://qiskit.org/textbook/ch-machine-learning/machine-learning-qiskit-pytorch.html)
* [https://qiskit-community.github.io/qiskit-machine-learning/tutorials/05_torch_connector.html](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/05_torch_connector.html#)
* https://github.com/AnIsAsPe/ClassificadorCancerEsofago
