# Deep learning Yugioh card classifier
One shot learning (image recogntion) on 10,856 unique Yugioh cards

This is a program that uses deep learning to recognize different cards from among all 10,856 cards in the game. From a dataset 

Got inspiration to try this out after seeing when Konami made:

https://ygorganization.com/konami-staff-develop-powerful-image-recognition-technology/

I wanted to try to replicate as close to Konami’s results as much as possible, but tried a different approach to the problem.

_______________________________________________________________

Here are some results of it:

![GitHub Logo](/images/0.png)

Real life Yugioh card photos recognized by AI (with or without sleeves)
_______________________________________________________________

# **__Method__**

Here, I will explain some of the methods I used. Traditional machine learning / deep learning methods would not do so well with a massive amount of cards with each one having one or a few art designs, so I ended up using a method known as one-shot learning. One-shot learning is basically just a method to compare two different cards together and calculate how similar they are. One would feed two images (target image and a comparison image) into the same network and calculate the distance between the output feature maps to get a simularity score. The architecture that was used is resnet101 that used triplet loss function to capture features from the card images and use positive / negative anchors to move simular cards closer along with moving different cards further from each other. 

The ORB algorithm is used as a ranking system to deal with "borderline" predictions where the correct card not predicted as rank 0 (1st place), but got relatively close. The number of ORB points are then counted between two compared images and resulting number is used in a weighted formula with the one-shot learning similarity score to get a final score.

One-shot learning generates the first wave of rankings (to work despite lighting / contrast conditions), then ORB is then used to resolve any borderline cases.

# **__Process__**

Sample comparison (rank 0 is the most confident prediction):

The closer the dissimilarity is to 0, the more similar an image is with another

![GitHub Logo](/images/1.png)

Comparing similarity of input image with database of cards. Left is input image while right are card arts from database.

![GitHub Logo](/images/2.png)

Even if input image (left) was under very dark conditions, Dark Magician still gets recognized.

![GitHub Logo](/images/3.png)

Blue Eyes White Dragon gets recognized even with different lighting conditions

_______________________________________________________________

The AI / machine learning model was tested on real photos of cards (cards with and without sleeves)

![GitHub Logo](/images/4.png)
Left card has card sleeve, right one is without

These types of images are what I ultimately want my AI classifer to be successful on: having a camera point down on your card and be able to recognize it.

_______________________________________________________________

However, since buying over 10,000 cards and taking pictures of them wasn’t a realistic scenario for me, so I tried the next best thing, which was to test it on an online database of Yugioh cards and artificially add challenging modifications to it. Modifications included changing brightness, contrast, and shear to simulate Yugioh cards under different lighting / photo quality scenarios in real life.

Here is some of the (right is the modified simulated image and the right the original art):

![GitHub Logo](/images/5.png)
Batch of different images under different contrast / lighting conditions. Left of each pair is input image, right is the card art from database

# **__Results__**

And here are the results of testing on this different dataset:

![GitHub Logo](/images/6.png)

The AI classifier managed to achieve around 99% accuracy on all the cards in the game of Yugioh.

This was meant to be a quick project, so I am happy with progress that was made. I may try to see if I can gather more Yugioh cards and try to improve the system.

# **__Dataset__**
The card dataset was retrieved from an API. The full size version of the cards were used:

https://db.ygoprodeck.com/api-guide/

You can also download the dataset and trained model here as well:
https://drive.google.com/drive/folders/1JZCt7hHf4NYgEp2XiE1SpNboAbgBxwDY?usp=sharing

# **__Current improvements to be made__**

The dataset used for training were official card art images used from ygoprodeck (dataset A) and not real life photos of cards in the wild / pictures of cards taken by a camera (dataset B). 

The 99% accuracy results were from training and testing on dataset A while the trained model also was tested on a handful of cards on dataset B. However we don't have a lot of data for dataset B to preform actual training on it or even mass-evaluation. This repo proves that our model is capable of learning Yugioh cards through dataset A and has potential on succeeding with dataset B, which is the more realistic and natural set of images we are aiming to succeed on. Setting up a data collection infrastructure to mass-collect image samples for dataset B would greatly advance this project and help confirm the strength of the model.


Also, this program does not have a proper object detector and just uses simple image processing methods (4 point transformation) to get the bounding box of the card and align it. Using a proper object detector like YOLO would be ideal, which would also aid in being able to detect multiple cards in the demo.
