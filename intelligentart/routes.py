from flask import jsonify
from flask import request, render_template
from flask_cors import cross_origin
from intelligentart import app
from intelligentart.guesstime import run_guess
from intelligentart.generate import run_generate
from intelligentart.generate import run_test
from intelligentart.generate import run_rate
import uuid


@app.route("/")
def home():
    return render_template('home.html')


@app.route("/about")
def about():
    return render_template('about.html')


@app.route("/create")
def create():
    return render_template('create.html')


@app.route("/guess", methods=["GET", "POST"])
@cross_origin()
def guess_time():
    if request.method == "GET":
        return render_template('create.html')
    elif request.method == "POST":
        message = request.get_json(force=True)
        image = message['image']

        result = run_guess(image)

        response = {
            'result': {
                'time': result
            }
        }
        return jsonify(response)


@app.route("/test", methods=["GET", "POST"])
@cross_origin()
def test():
    if request.method == "GET":
        return render_template('create.html')
    elif request.method == "POST":
        # generating id
        id1 = uuid.uuid1()
        # 64 128 256 384 512
        max_dim = 128
        # getting passed variables from message
        message = request.get_json(force=True)
        encoded_content_image = message['content_image']
        encoded_style_images = message['style_images']
        epochs = int(message['epochs'])
        steps_per_epoch = int(message['steps'])
        content_weight = int(message['content_weight'])
        style_layers = message['style_layers']
        content_layers = message['content_layer']

        response = run_test(id1, encoded_content_image, encoded_style_images
                               , epochs, steps_per_epoch, content_weight, content_layers, style_layers)

        return jsonify(response)


@app.route("/generate", methods=["GET", "POST"])
@cross_origin()
def generate():
    if request.method == "GET":
        return render_template('create.html')
    elif request.method == "POST":
        # getting passed variables from message
        message = request.get_json(force=True)
        file_name = message['file_name']

        result = run_generate(file_name)

        response = {
            'result': {
                'generated_image': result
            }
        }
        return jsonify(response)


@app.route("/review", methods=["GET", "POST"])
@cross_origin()
def review():
    if request.method == "GET":
        return render_template('create.html')
    elif request.method == "POST":
        # getting passed variables from message
        message = request.get_json(force=True)
        rating = int(message['rating'])
        file_name = message['file_name']

        result = run_rate(file_name, rating)

        response = {
            'result': {
                'done': result
            }
        }
        return jsonify(response)
