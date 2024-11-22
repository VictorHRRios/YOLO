from flask import Flask, render_template, request, jsonify
from io import BytesIO
import torch
from PIL import Image
from flask import Response
app = Flask(__name__)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)  # yolov5n - yolov5x6 or custommodel.load_weights('model.h5')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    image = Image.open(BytesIO(file.read()))

    results = model(image)
    r_img = results.render()
    img_with_boxes = r_img[0]

    image = Image.fromarray(img_with_boxes)

    img_io = BytesIO()
    image.save(img_io, format="PNG")
    img_io.seek(0)

    return Response(img_io, mimetype="image/png")


if __name__ == '__main__':
    app.run(debug=True)
