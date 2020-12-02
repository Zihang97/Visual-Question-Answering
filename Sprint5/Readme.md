# Sprint5
## Dataset
In Sprint5, our purpose is to finish our dataset. Using Labelme to label our images so that to build our own dataset.

### Manaully label images
Here are the examples of our dataset. We have finished the Eiffel Tower, the Great Wall, colosseum and pyramid types.
<p align="left">
  <img src="picture/image5.jpg" height=300/>
  <img src="picture/image6.jpg" height=300/>
  <img src="picture/image7.jpg" height=250/>
  <img src="picture/image8.jpg" height=250/>
</p>

Here are the examples of our dataset which are labeled. 
<p align="center">
  <img src="picture/image1.png" height=400/>
</p>

<p align="center">
  <img src="picture/image2.png" height=400/>
</p>

<p align="center">
  <img src="picture/image3.png" height=400/>
</p>

<p align="center">
  <img src="picture/image9.png" height=400/>
</p>

 We colllected over one thousand images from internet and labeled all of them. And atter saving them, a json file corresponding to the changed picture will be generated.
 
 <p align="left">
  <img src="picture/image4.png" height=500/>
</p>


### Generate the dataset
In order to run the json file, I wrote a batch file to execute labelme_json_to_dataset.py

```python
now=`date +'%Y-%m-%d %H:%M:%S'`
start_time=$(date --date="$now" +%s);
c=0
echo "Now begin to search json file..."
cd ./greatwall
for file in ./*
do
    if [ "${file##*.}"x = "json"x ]
    then
    echo "transfering $file to dataset"
    labelme_json_to_dataset "$file"
    c=`expr $c + 1`
    fi
#    printf "no!\n "
done
now=`date +'%Y-%m-%d %H:%M:%S'`
end_time=$(date --date="$now" +%s);
echo "transfered $c json files about greatwall to dataset, used time:"$((end_time-start_time))"s"
```

The result of execution will generate a folder corresponding to the picture, which includes four files: img, info, label, label_viz

<p align="left">
  <img src="picture/colosseum.PNG"/>
</p>

<p align="left">
  <img src="picture/eiffeltower.PNG"/>
</p>

<p align="left">
  <img src="picture/great.PNG"/>
</p>

<p align="center">
  <img src="picture/colosseum (14).jpg" width=300/>
  <img src="picture/colosseumlabel.png" width=300/>
  <img src="picture/colosseum_viz.png" width=300/>
</p>

<p align="center">
  <img src="picture/eiffel.png" width=300/>
  <img src="picture/eiffellabel.png" width=300/>
  <img src="picture/eiffel_viz.png" width=300/>
</p>

<p align="center">
  <img src="picture/wall.png" width=300/>
  <img src="picture/wall_label.png" width=300/>
  <img src="picture/wall_viz.png" width=300/>
</p>

<p align="center">
  <img src="picture/py.png" width=300/>
  <img src="picture/py_label.png" width=300/>
  <img src="picture/py_viz.png" width=300/>
</p>

Another step is needed to extract label.png file from single \_json directory. I wrote a [python script](extract.py) to extract and rename the file.

<p align="left">
  <img src="picture/direc.PNG" width=300/>
</p>

<p align="left">
  <img src="picture/content.PNG" width=600/>
</p>
