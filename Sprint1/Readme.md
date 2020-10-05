## Product mission
This product is to answer the background-related question about image.

## MVP
The product can recognize the type of background object.

For example, it can recognize the background is a mountain but may not exactly know which exact mountain it is.

## User Story
There are always many silly criminals who want to show off what they have down and post pictures on the social media. As a police officer, I can use the background recognition product to analyze the picture. Find their location and catch them.

When you walk along the street and see a movie poster. The scenery in the poster is so fascinating that you really want to know where it is and might take vacation there some day. So you can ask our product, “Where is it located?” . And then you will get the answer you want.

## Literature Review

<p align="left">
    <img src="https://github.com/Zihang97/Visual-Question-Answering/blob/main/Sprint1/Picture/four%20categories.PNG" width="500"/>
</p>

### Joint Embedding Approaches
Joint embedding is a classic idea when dealing with multi-modal problems. Here it refers to the joint coding of images and problems. The schematic diagram of this method is:

<p align="left">
    <img src="https://github.com/Zihang97/Visual-Question-Answering/blob/main/Sprint1/Picture/Joint.PNG" width="600"/>
</p>

First, the image and question are first encoded by CNN and RNN to obtain their respective features, and then jointly input to another encoder to obtain joint embedding, and finally the answer is output through the decoder. It is worth noting that some works treat VQA as a sequence generation problem, while other works simplify VQA to a classification problem with a predictable answer range. In the former setting, the decoder is an RNN, which outputs sequences of unequal length; the latter decoder is a classifier, which selects the answer from a predefined vocabulary.

### Models using an external knowledge base
Although VQA is to solve the task of answering questions by looking at pictures, in fact, many questions often require certain prior knowledge to answer. For example, in order to answer the question "Are there many trees on the mountain?", the model must know the definition of "mountain" instead of simply understanding the content of the image.
A relatively good job in this area is a paper published by Wu Qi as a work "Ask Me Anything: Free-form Visual Question Answering Based on Knowledge from External Sources". The model framework of this work is as follows.

<p align="left">
    <img src="https://github.com/Zihang97/Visual-Question-Answering/blob/main/Sprint1/Picture/knowledge.PNG" width="600"/>
</p>

The red part indicates that the image is classified with multiple labels to obtain the image label (attribute).

The blue part means that the 5 most obvious tags in the above image tags are input into the associated DBpedia to retrieve the relevant content, and then encoded using Doc2Vec.

The green part means that multiple image descriptions (titles) are generated using the above image tags, and this group of image descriptions are coded.

The above three items are simultaneously input into a Seq2Seq model as its initial state, and then the Seq2Seq model encodes the question, decodes the final answer, and uses the MLE method for training.

## Techniques
For any VQA problem, it has two main focus: image and question.

The aim of our product is to recognize the background of image, so the questions will be quite similar and not complex, such as ‘ What’s the background of the image?’ and ‘ Where are the person in the image?’. 

The image processing part is the key point of our product.

### Which part is the background?
#### Image Matting
Image matting is the problem of accurate foreground estimation in images and videos.

They can separate the foreground and background, but the focus is foreground estimation.

They need trimap or scribble as part of dataset to help analyze, which is unrealistic for VQA. 

<p align="left">
    <img src="https://github.com/Zihang97/Visual-Question-Answering/blob/main/Sprint1/Picture/mat.PNG" width="400"/>
</p>

#### Scene Labelling
Scene labelling is the combination of semantic segmentation and classification

The task of scene labeling is to densely predict every pixel of an image into one of the pre-defined classes.

Semantic segmentation can help us find which part is background.

Classification can help us define what’s in the background.



