from flask import Flask, request, jsonify,send_file
from flask_cors import CORS
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

prototxt_path = 'models/colorization_deploy_v2.prototxt'
model_path = 'models/colorization_release_v2.caffemodel'
kernel_path = 'models/pts_in_hull.npy'

# Load the colorization model and the cluster points
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
points = np.load(kernel_path)

# Reshape and assign the cluster points to the model
points = points.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def colorize_image(image):
    # Normalize the image and convert it to Lab color space
    normalized = image.astype("float32") / 255.0
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)

    # Resize the image and extract the L channel
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Feed the L channel to the model and get the predicted a and b channels
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize the a and b channels to the original image size
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L = cv2.split(lab)[0]

    # Combine the L, a, and b channels and convert them to RGB color space
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = (255.0 * colorized).astype("uint8")
 
    return colorized


@app.route('/colorize', methods=['POST'])
def colorize_image_route():
    
    if 'image' not in request.files:
        return jsonify({'result': 'error', 'message': 'No image provided'})

    file = request.files['image']
    
    # Save the image to a temporary file
    original_image_path ='Original_' + file.filename
    file.save(original_image_path)

    # Read the image using OpenCV
    image = cv2.imread(original_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Colorize the image using the function defined above
    colorized_image = colorize_image(image)

    # Save the colorized image to a file
    colorized_image_path =  'Colored_' + file.filename
    cv2.imwrite(colorized_image_path, cv2.cvtColor(colorized_image, cv2.COLOR_RGB2BGR))

    return send_file(colorized_image_path, mimetype=file.content_type)
        

app.run(debug=True, port=5000)