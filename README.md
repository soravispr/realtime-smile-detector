# realtime-smile-detector
The simplest implementation of real-time facial smile detection for webcam you'll ever find.

This small project demonstrates how to simply implement a real-time smile detection via a PC webcam. Face is first detected by OpenCV's Haar Cascade algorithm, then the image of the face is fed to a simple CNN-based model to recognize a smiley face. I've trained the classification model using images from [Kaggle dataset](https://www.kaggle.com/datasets/ghousethanedar/smiledetection).\
\
Requirements:
```
opencv-contrib-python==4.5.5.64
tensorflow-gpu==2.7.0
```
Usage:
```
python main.py
```
\
![not_smile](https://user-images.githubusercontent.com/43654034/211880559-093f0fd2-6d3d-4039-a52c-fe18d363ce62.JPG)


![smile](https://user-images.githubusercontent.com/43654034/211880566-4ba609cc-d157-4b42-80a7-ef17a49578c1.JPG)

