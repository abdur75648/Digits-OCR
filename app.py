import flask
import pickle
import numpy as np
from PIL import Image

# Use pickle to load in the pre-trained model.
with open(f'digits_OCR.pickle', 'rb') as f:
    parameters= pickle.load(f)

#### Update
# 1. Sigmoid Activation FUnction
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache
# 3. Softmax Activation FUnction
def softmax(Z):
    expZ = np.exp(Z)
    A = expZ / expZ.sum(axis=0, keepdims=True)
    cache = Z
    return A, cache
# Linear_activation_forward : :Implements the forward propagation for a single layer
def linear_activation_forward(A_prev, W, b, activation):
    Z = np.dot(W,A_prev) + b
    linear_cache = (A_prev, W, b,)       # This "linear_cache" contains (A_prev, W, b)
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z) # This "activation_cache" contains "Z"
    elif activation == "softmax":
        A, activation_cache = softmax(Z)    # This "activation_cache" contains "Z"
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache
# Linear_model_forward: # Implements the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters)//2  # number of layers in the neural network
    # Implement [LINEAR -> RELU]*(L-1).
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "sigmoid")
        # Add "cache" to the "caches" list
        caches.append(cache) # HERE -> cache = (linear_cache, activation_cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "softmax")
    # Addcompilecache" to the "caches" list
    caches.append(cache) # HERE -> cache = (linear_cache, activation_cache)
    return AL, caches
def max_index(arr):
    return arr.argmax()
def predict_digit(X, parameters):
    probas, caches = L_model_forward(X, parameters)
    confidence_score = np.round((max(probas)/sum(probas)),2)
    return max_index(probas), confidence_score

app = flask.Flask(__name__, template_folder='templates')
app = flask.Flask(__name__, static_folder="static")
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        image= flask.request.files['myfile']
        fileImage = (Image.open(image).convert("P").resize([28,28],Image.ANTIALIAS))
        fileImage=np.asarray(fileImage.getdata(),dtype=np.float64).reshape(784,1)
        example = np.squeeze(fileImage)
        example = example.reshape(example.shape[0], -1)
        my_image= (example)/255.
        my_predicted_image,confidence_score = predict_digit(my_image, parameters)
        prediction = str(np.squeeze(my_predicted_image))
        confidence_score = str(np.squeeze(confidence_score))
        return flask.render_template("result.html",prediction=prediction,confidence_score=confidence_score)
def main():
    return(flask.render_template('main.html'))
if __name__ == '__main__':
    app.run()