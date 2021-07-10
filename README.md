# Digits-OCR
**A Machine Learning Web App For Hand-written Digits Recognition**
Link : http://digits-ocr.herokuapp.com/

# Background
* A simple feed-forward neural network model, trained from sratch on MNIST dataset
* The code was written from scratch using numpy, as tensorflow/pytorch couldn't be used due to slug size limit of Heroku
* It takes input image, pre-processes it & feeds to the model to generate the output
* The test accuracy acheived was 97.3 % (It could be above 99% if CNNs were used)
* It was deployed on Heroku using Flask and Python
