# Background-related Visual Question Answering in PyTorch
This repo contains a PyTorch implementation of our background-related VQA model.

## Requirements
PyTorch and Torchvision needs to be installed before running the scripts, together with `PIL` and `opencv` for data-preprocessing and `tqdm` for showing the training progress. PyTorch v1.1 is supported (using the new supported tensoboard); can work with ealier versions, but instead of using tensoboard, use tensoboardX.

```bash
pip install requirements.txt
```

## Training
To train a model, first download the dataset to be used to train the model, then choose the desired architecture, add the correct path to the dataset and set the desired hyperparameters (the config file is detailed below), then simply run:

```bash
python train.py --config config.json
```


## Inference

For inference, we need a PyTorch trained model, the images we'd like to segment and the config used in training (to load the correct model and other parameters), 

```bash
python inference.py --config config.json --model best_model.pth --images images_folder
```

The predictions will be saved as `.png` images using the default palette in the passed fodler name, if not, `outputs\` is used, for Pacal VOC the default palette is:


Here are the parameters availble for inference:
```
--output       The folder where the results will be saved (default: outputs).
--extension    The extension of the images to segment (default: jpg).
--images       Folder containing the images to segment.
--model        Path to the trained model.
--mode         Mode to be used, choose either `multiscale` or `sliding` for inference (multiscale is the default behaviour).
--config       The config file used for training the model.
```




## Code structure
The code structure is based on [pytorch-template](https://github.com/victoresque/pytorch-template/blob/master/README.md)

  ```
  pytorch-template/
  │
  ├── train.py - main script to start training
  ├── inference.py - inference using a trained model
  ├── trainer.py - the main trained
  ├── config.json - holds configuration for training
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   ├── base_dataset.py - All the data augmentations are implemented here
  │   └── base_trainer.py
  │
  ├── dataloader/ - loading the data for different segmentation datasets
  │
  ├── models/ - contains semantic segmentation models
  │
  ├── saved/
  │   ├── runs/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │  
  └── utils/ - small utility functions
      ├── losses.py - losses used in training the model
      ├── metrics.py - evaluation metrics used
      └── lr_scheduler - learning rate schedulers 
  ```

## Config file format
Config files are in `.json` format:
```javascript
{
    "name": "PSPNet",
    "n_gpu": 3,
    "use_synch_bn": false,

    "arch": {
        "type": "PSPNet",
        "args": {
            "backbone": "resnet50",
            "freeze_bn": false,
            "freeze_backbone": false
        }
    },

    "train_loader": {
        "type": "CityScapes",
        "args":{
            "data_dir": "/projectnb2/ece601/VQA-team10/CityScapes",
            "batch_size": 32,
            "base_size": 400,
            "crop_size": 380,
            "augment": true,
            "shuffle": true,
            "scale": true,
            "flip": true,
            "rotate": true,
            "blur": false,
            "split": "train",
            "num_workers": 8
        }
    },

    "val_loader": {
        "type": "CityScapes",
        "args":{
            "data_dir": "/projectnb2/ece601/VQA-team10/CityScapes",
            "batch_size": 32,
            "crop_size": 480,
            "val": true,
            "split": "val",
            "num_workers": 4
        }
    },

    "optimizer": {
        "type": "SGD",
        "differential_lr": true,
        "args":{
            "lr": 0.01,
            "weight_decay": 1e-4,
            "momentum": 0.9
        }
    },

    "loss": "CrossEntropyLoss2d",
    "ignore_index": 255,
    "lr_scheduler": {
        "type": "Poly",
        "args": {}
    },

    "trainer": {
        "epochs": 80,
        "save_dir": "saved/",
        "save_period": 10,
  
        "monitor": "max Mean_IoU",
        "early_stop": 10,
        
        "tensorboard": true,
        "log_dir": "saved/runs",
        "log_per_iter": 20,

        "val": true,
        "val_per_epochs": 5
    }
}
```



## Acknowledgement
- [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
- [Pytorch-Template](https://github.com/victoresque/pytorch-template/blob/master/README.m)
- [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)
- [PyTorch-Segmentation](https://github.com/yassouali/pytorch_segmentation)
