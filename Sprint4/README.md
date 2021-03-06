# Sprint4
## Dataset
In sprint4, our purpose is to recognize some famous tourist spots around the world. 
In sprint3, the dataset we make didn’t perform well when we using it to train our model. At that time, we actually use a semantic segmentation dataset to train our model and then use the dataset we made to test it. 
So, this time, we try to follow the format of the semantic segmentation dataset we used before to build the new dataset.
We choose to download the labelme to label our images. 
<p align="left">
  <img src="label.png" >
</p>

For example, we choose the Great Wall, pyramid and the Eiffel Tower to build our dataset.
<p align="left">
  <img src="the great wall1.jpg" >
</p>

<p align="left">
  <img src="pyramid1.jpg" >
</p>

<p align="left">
  <img src="the Eiffel Tower2.jpg">
</p>

## Model
### Expand the range of question
In sprint 4 the first thing we did was expand the range of question. In sprint 3 we only used one question,'What's the background of image?'. Though background-related question is limited, there are still some questions that apply. Unlike traditional VQA algorithms which use element-wise add or multiplication to combine the features from question and image, I use a if-else statement structure which divided the output of question into three bins. I transforms the output of question after LSTM into background-related scores.

<p align="center">
  <img src="test.jpg" width=400/>
  <img src="test.png" width=400/>
</p>

<p align="center">
  <img src="background.PNG" width=800/>
</p>

<p align="center">
  <img src="where.PNG" width=800/>
</p>

<p align="center">
  <img src="unrelated_person.PNG" width=800/>
</p>

<p align="center">
  <img src="unrelated.PNG" width=800/>
</p>

### Improve the performance
Based on previous 80 epoches checkpoint, I trained another 80 epoches needing 3.5 hrs with 3 V100 GPUs on SCC. The results from later training show that both mIOU and accuracy changed very little in last 20 epoches, in some epoches the results even got worse. I think the reason lied in the simple network with only 50 layers maybe having degradation in deep training.

<p align="center">
  <img src="sky (182).jpg" width=400/>
  <img src="sky (182).png" width=400/>
</p>

| Backbone     | CityScapes val mIoU | CityScapes accuracy | Pretrained Model |
| :----------: |:-----------------: |:-------------------:|:----------------:|
| ResNet 50 + CRF   | 60%                | 95%                 | [Dropbox](https://www.dropbox.com/s/qac5r3n0na69s9g/best_model.pth?dl) |


## Sprint 5
1. Finish the dataset and train on new dataset to expand the class.
2. Consider if we need increase the number of layers to get better performance.
