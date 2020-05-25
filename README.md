# Medical-Report-Generation-Using-X-Ray-Images

### Overview
The aim of this project is to generate medical reports from X-ray images. This problem holds a great role when there is a lack of Quality doctors in remote parts of the world, and just lab technicians could take an X-Ray and send it to doctors anywhere across the globe.  It would bring in quality healthcare services at cheaper rates to parts of world where doctors would take weeks to reach.
This is my final year Major Project. 

Data Source: http://academictorrents.com/details/66450ba52ba3f83fbf82ef9c91f2bde0e845aba9
#### How Data Looks?
![alt text](https://github.com/sawarn69/Medical-Report-Generation-Using-X-Ray-Images/blob/master/Data_format.png)


#### Sample Outputs
![alt text](https://github.com/sawarn69/Medical-Report-Generation-Using-X-Ray-Images/blob/master/Outputs/CXR2536_IM-1049-1001_result.png)


![alt text](https://github.com/sawarn69/Medical-Report-Generation-Using-X-Ray-Images/blob/master/Outputs/CXR2689_IM-1160-1002001_result.png)

#### Notebooks Layout
1. References have the references used
2. Workflow has notebooks which were  used for experiments and tuning purposes
3. Loss plots has different plots for training and validation purposes
4. Training has various training notebooks
5. Outputs have output generated from the model
6. Checkpoints has model weights
7. Tokenizer has a .pikle file containing key value pairs from the tokenizer's vocabulary

#### Approach To Solve the problem

1. Use pretrained  weights from Inception V3 network. Extract features from the last CNN layer, as we would be using attention in this model.  Remove the last layer of the model, and transform all the input images using these weights. 

2. Define an encoder layer. It has a fully connected layer. Pass the pre trained vector  from Inception V3 network to featurize images at encoder stage. 

3. Now, Lets talk about the impressions. Tokenize the text data. Use top k words, and replace all other words with <UNK> tokens.  Pad sequences to the maximum length. 

#### Model and Training Details

1.  Start: Use the vectors that we got by transforming all images on Inception V3 weights. Pass it through encoder. We define a LSTM/GRU(not bidirectional) decoder. This decoder attends the image to predict the next word. 

2.  Train: To train the model, pass the saved vectors from images through encoder. Pass the encoder output, hidden states, and <start> token to the decoder network. The decoder's hidden state is passed back to model, and the output is used to calculate the loss. Now pass the target word as the next input to decoder. It is different and worth noting that we do not pass the predicted output but the original output as the next input. This is known as Teacher Forcing. Finally, calculate gradients and backpropagate. 

3.  While Generating the output, do the same as step above. But here we pass the previous output as input to the next time step of the decoder.  Stop at <end> token. Store the weights of attention at every timestep of generating the output. 

4.  Plot the Attention weight and the part it is focusing at every timestep of input. 

#### Training loss
![alt text](https://github.com/sawarn69/Medical-Report-Generation-Using-X-Ray-Images/blob/master/losses_plots/While%20experimenting/epoch_train_loss.jpg)
#### Validation loss
![alt text](https://github.com/sawarn69/Medical-Report-Generation-Using-X-Ray-Images/blob/master/losses_plots/While%20experimenting/epoch_val_loss.jpg)
#### Research-Papers/Solutions/Architectures/Kernels/References

1. https://github.com/wisdal/diagnose-and-explain

2. https://towardsdatascience.com/deep-learning-for-detecting-pneumonia-from-x-ray-images-fc9a3d9fdba8

3. https://towardsdatascience.com/deep-learning-for-detecting-pneumonia-from-x-ray-images-fc9a3d9fdba8

4. https://www.youtube.com/watch?v=MgrTRK5bbsg&list=PLQY2H8rRoyvxcmHHRftsuiO1GyinVAwUg&index=21&t=0s

5. https://conferences.oreilly.com/tensorflow/tf-ca-2019/public/schedule/proceedings

6. https://github.com/wangleihitcs/Papers/blob/master/medical%20report%20generation/2019(AAAI)%20Knowledge-Driven%20Encode%2C%20Retrieve%2C%20Paraphrase%20for%20Medical%20Image%20Report%20Generation.pdf

7. https://github.com/wangleihitcs/Papers/tree/master/medical%20report%20generation

8. https://github.com/shreyanshrs44/Medical-Report-Generation

9. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8867873

10. https://github.com/zhjohnchan/awesome-radiology-report-generation

11. https://github.com/omar-mohamed/X-Ray-Report-Generation/tree/master

12. https://www.tensorflow.org/tutorials/text/image_captioning

13. https://arxiv.org/pdf/1502.03044.pdf

#### Updates to be added soon:
1. Screenshots from tensorboard plots 
2. Model Weights 

