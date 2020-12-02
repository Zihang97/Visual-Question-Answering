# Sprint5
## Dataset
In Sprint5, our purpose is to finish our dataset and use our model to train new dataset. Using Labelme to label our images so that to build our own dataset.

Here are the example of our dataset. We choose the Eiffel Tower, the Great Wall, colosseum and so on.

<p align="left">
  <img src="picture/image1.png" height=500/>
</p>

<p align="left">
  <img src="picture/image2.png" height=500/>
</p>

<p align="left">
  <img src="picture/image3.png" height=500/>
</p>

 We colllected over one thousand images from internet and labeled all of them. And atter saving them, a json file corresponding to the changed picture will be generated.
 
 <p align="left">
  <img src="picture/image4.png" height=500/>
</p>

```python
#coding:utf-8
import os
 
path = 'D:\\data' #path是你存放json的路径
json_file = os.listdir(path)
 
for file in json_file:
    os.system("python E:\Anocado\Anocado3\envs\labelme\Scripts\labelme_json_to_dataset.py %s"
              % (path + file))         #使用自己的labelme路径```
