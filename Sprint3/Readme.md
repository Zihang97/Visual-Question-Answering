# Sprint3
## Image processing
### Building our own model
I used [CityScapes dataset](https://www.cityscapes-dataset.com/downloads/) to train the model, which use ResNet-50 as backbone combined with CRF. I choose CityScapes dataset as this dataset is made up of images from different cities, which matches our aim most. We can still use it in sprint4 to tell in which city the images are. Then I use our dataset to finetune the model. I don't use our dataset to train our model directly as our dataset don't have annotations. 

<p align="left">
  <img src="picture/label.PNG" width=800/>
</p>

I only use 19 classes out of 35 classes defined in CityScapes as our model don't have very complicated structure and there isn't too many parameters. So a lot of classes may lead to accuracy loss and low performance.

<p align="left">
  <img src="picture/gpus.PNG" width=250/>
</p>

<p align="left">
  <img src="picture/train_5.PNG" width=800/>
</p>

<p align="left">
  <img src="picture/train_80.PNG" width=800/>
</p>

### Trained Model:

| Backbone     | CityScapes val mIoU | CityScapes accuracy | Pretrained Model |
| :----------: |:-----------------: |:-------------------:|:----------------:|
| ResNet 50    | 56%                | 93%                 | [Dropbox](https://www.dropbox.com/s/qac5r3n0na69s9g/best_model.pth?dl=0) |
