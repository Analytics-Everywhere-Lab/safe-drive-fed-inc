# A Federated Incremental Learning Framework for Driver Safety
Insert authors' names

## Abstract
The enforcement of driver safety recommendations requires an approach that ensures user privacy and adapts to changes in legislation. To apply these requirements, we propose a driver monitoring approach using federated incremental learning in which new classes are incrementally learned from data acquired on the server, and local models in the vehicles are trained with federated learning to preserve privacy. In our approach, class incremental learning provides a mechanism for leveraging information from previous classes and preventing catastrophic forgetting, while vanilla FedAvg provides a weight aggregation method for the local models. We build custom datasets and incrementally train YOLOv8 models to detect classes of drivers' actions in a setup consisting of a server and two client devices. Our experimental results showed an average mAP50 value of 0.9502 on the server for 5 incremental steps. On the client side, Client 1 had an average mAP50 of 0.9539 after federated learning and Client 2 had an average mAP50 value of 0.8854. The inference results also showed that the models could accurately detect and predict classes on test images. These results demonstrate the effectiveness of our approach in integrating new classes of actions and effectively ensuring privacy is maintained.

## Overview
We present a client-server safety monitoring approach that detects up to 6 classes of user actions: no seatbelt, mobile (phone usage), inattentive, seatbelt, drowsiness, and drinking. Our current design involves two client devices representing the vehicles' onboard edge devices and a single server coordinating the training.
- Insert dataset images - Figure 2
The setup initially accommodates two classes and the model is trained on these two classes. In later steps, more classes are added (one class for each step) and the model is trained using incremental learning to accommodate the new classes. Each training step includes a federated learning process between the client devices and the server that enables local models to be improved and fine-tuned to user data while maintaining data privacy. 
- Insert overview diagram - Figure 1

## Results
- Insert Figure 7
- Insert Figure 8
- Insert Figure 9
