# Sprint5
## Dataset
In Sprint5, our purpose is to finish our dataset and use our model to train new dataset. Using Labelme to label our images so that to build our own dataset.

Here are the example of our dataset. We choose the Eiffel Tower, the Great Wall, colosseum and pyramid
<p align="left">
  <img src="picture/image5.jpg" height=300/>
  <img src="picture/image6.jpg" height=300/>
  <img src="picture/image7.jpg" height=300/>
  <img src="picture/image8.jpg" height=300/>
</p>

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


In order to run the json file, I wrote a batch file to execute labelme_json_to_dataset.py

```python
#coding
import os
 
path = 'D:\\data' #path is the path you save json file
json_file = os.listdir(path)
 
for file in json_file:
    os.system("python E:\Anocado\Anocado3\envs\labelme\Scripts\labelme_json_to_dataset.py %s"
              % (path + file))         #use your own labelme path
```

The result of execution will generate a folder corresponding to the picture, which includes four files: img, info, label, label_viz
              
