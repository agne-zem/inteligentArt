from __future__ import absolute_import
from tensorflow.keras import optimizers
from PIL import Image
from intelligentart import app
from intelligentart.models import Content, Style, Generated
from intelligentart import db

import numpy as np
import base64
import tensorflow as tf
import os
import PIL
import io
import time
import random


tf.compat.v1.enable_eager_execution()

style_weight = 1e-2
opt = optimizers.Adam(learning_rate=0.09, beta_1=0.99, epsilon=1e-1)

# =================== HELPERS ====================


# prepares image
def preprocess_image(image_path, max_dim):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


# converts tensor to image
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


# creates vgg that returns a list of intermediate output values
def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

# USAGE      - calculates gramian matrix
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.style_layers_count = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.style_layers_count],
                                          outputs[self.style_layers_count:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


# USAGE      - keeps the pixel values between 0 and 1
# PARAMETERS - image (Tensor)
# RETURNS    - clipped image (Tensor)
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


# USAGE      - shrinks image to maximum dimensions allowed
# PARAMETERS - image (Image), maximum dimensions (array)
# RETURNS    - shrunken image (Image)
def shrink(img, max_dim):
    base_width = max_dim
    width_percent = (base_width/float(img.size[0]))
    height_size = int((float(img.size[1])*float(width_percent)))
    img = img.resize((base_width, height_size), Image.ANTIALIAS)
    return img


def merge_and_save(identifier, style_image_count, resample=Image.BICUBIC):
    merged = Image
    img1 = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], identifier.hex + 'style0' + '.png'))
    img2 = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], identifier.hex + 'style1' + '.png'))
    img2 = img2.resize((int(img2.width * img1.height / img2.height), img1.height), resample=resample)

    merged = Image.new('RGB', (img1.width + img2.width, img1.height))
    merged.paste(img1, (0, 0))
    merged.paste(img2, (img1.width, 0))

    if style_image_count == 3:
        img3 = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], identifier.hex + 'style2' + '.png'))

        img3 = img3.resize((merged.width, int(img3.height * merged.width / img3.width)), resample=resample)

        merged3 = Image.new('RGB', (merged.width, merged.height + img3.height))
        merged3.paste(merged, (0, 0))
        merged3.paste(img3, (0, merged.height))
        merged = merged3

    if style_image_count == 4:
        img3 = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], identifier.hex + 'style2' + '.png'))
        img4 = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], identifier.hex + 'style3' + '.png'))

        img4 = img4.resize((int(img4.width * img3.height / img4.height), img3.height), resample=resample)

        merged2 = Image.new('RGB', (img3.width + img4.width, img3.height))
        merged2.paste(img3, (0, 0))
        merged2.paste(img4, (img3.width, 0))

        merged2 = merged2.resize((merged.width, int(merged2.height * merged.width / merged2.width)), resample=resample)

        merged4 = Image.new('RGB', (merged.width, merged.height + merged2.height))
        merged4.paste(merged, (0, 0))
        merged4.paste(merged2, (0, merged.height))
        merged = merged4

    style_image_name = identifier.hex + 'style_m' + '.png'
    merged.save(os.path.join(app.config['UPLOAD_FOLDER'], style_image_name))


def run_test(identifier, encoded_content_image, encoded_style_images, epochs, steps_per_epoch
                       , content_weight, content_layers, style_layers):
    # layer count
    content_layers_count = len(content_layers)
    style_layers_count = len(style_layers)
    # style image count
    style_image_count = len(encoded_style_images)
    max_dim = 512
    total_variation_weight = 30

    # decoding json content image
    decoded_content_image = base64.b64decode(encoded_content_image)
    decoded_style_images = []
    # decoding json style images
    # getting array of decoded image data
    for i in range(0, style_image_count, 1):
        decoded_style_images.append(base64.b64decode(encoded_style_images[i]))

    # converting back to image
    # this is needed in order to save images in folder for further use

    # converting content image and saving it
    content_image = Image.open(io.BytesIO(decoded_content_image))
    content_image = shrink(content_image, max_dim)
    content_img_name = identifier.hex + 'content' + '.png'
    content_image.save(os.path.join(app.config['UPLOAD_FOLDER'], content_img_name))

    # creating content image query for database
    c_image = Content(file_name=content_img_name, processing_status=1)
    db.session.add(c_image)
    db.session.commit()


    # converting style images and saving them
    for i in range(0, style_image_count, 1):
        style_image = Image.open(io.BytesIO(decoded_style_images[i]))
        style_image = shrink(style_image, max_dim)
        style_img_name = identifier.hex + 'style' + str(i) + '.png'
        style_image.save(os.path.join(app.config['UPLOAD_FOLDER'], style_img_name))

        # creating style image query for database
        s_image = Style(file_name=style_img_name, fk_content_id=c_image.id)
        db.session.add(s_image)
        db.session.commit()


    max_dim = 128

    # preparing content image
    content_image = preprocess_image(os.path.join(app.config['UPLOAD_FOLDER'], content_img_name)
                                     , max_dim)



    # merging style images together into one
    if style_image_count > 1:
        merge_and_save(identifier, style_image_count)
        style_image = preprocess_image(os.path.join(app.config['UPLOAD_FOLDER'], identifier.hex + 'style_m' + '.png')
                                       , max_dim)
    else:
        style_image = preprocess_image(os.path.join(app.config['UPLOAD_FOLDER'], identifier.hex + 'style0' + '.png')
                                       , max_dim)

    extractor = StyleContentModel(style_layers, content_layers)
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    def style_content_loss(outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                               for name in style_outputs.keys()])
        style_loss *= style_weight / style_layers_count

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                                 for name in content_outputs.keys()])
        content_loss *= content_weight / content_layers_count
        loss = style_loss + content_loss
        return loss

    @tf.function()
    def train_step(img):
        with tf.GradientTape() as tape:
            outputs = extractor(img)
            loss = style_content_loss(outputs)
            loss += total_variation_weight * tf.image.total_variation(image)
        grad = tape.gradient(loss, img)
        opt.apply_gradients([(grad, img)])
        img.assign(clip_0_1(img))

    # initializing optimization variable
    image = tf.Variable(content_image)

    start = time.time()

    i = 1

    ep_random1 = random.randint(1, 5)
    ep_random2 = random.randint(1, 5)
    stp_random1 = random.randint(1, 50)
    stp_random2 = random.randint(1, 50)
    file_name1 = identifier.hex + 'test_generated1' + '.png'
    file_name2 = identifier.hex + 'test_generated2' + '.png'
    file_name3 = identifier.hex + 'test_generated3' + '.png'

    ep_max = max(ep_random1, ep_random2, epochs)
    stp_max = max(stp_random1, stp_random2, steps_per_epoch)
    step = 0
    for n in range(ep_max):
        for m in range(stp_max):
            step += 1
            train_step(image)
            # save as custom
            if steps_per_epoch - 1 == m and epochs - 1 == n:
                tensor_to_image(image).save(os.path.join(app.config['UPLOAD_FOLDER'], file_name1))
            if stp_random1 - 1 == m and ep_random1 - 1 == n:
                tensor_to_image(image).save(os.path.join(app.config['UPLOAD_FOLDER'], file_name2))
            if stp_random2 - 1 == m and ep_random2 - 1 == n:
                tensor_to_image(image).save(os.path.join(app.config['UPLOAD_FOLDER'], file_name3))
            print(".", end='')
        print("Train step: {}".format(step))

    end = time.time()
    print("Total time: {:.1f}".format(end - start))

    if style_layers_count > 3:
        type_var = 1
    else:
        type_var = 2

    # creating generated image query for database
    g_image = Generated(file_name=file_name1, type=type_var, content_weight=content_weight, custom=True, epochs=epochs
                        , steps=steps_per_epoch, fk_content_id=c_image.id)
    g_image2 = Generated(file_name=file_name2, type=type_var, content_weight=content_weight, custom=False
                         , epochs=ep_random1, steps=stp_random1, fk_content_id=c_image.id)
    g_image3 = Generated(file_name=file_name3, type=type_var, content_weight=content_weight, custom=False
                         , epochs=ep_random2, steps=stp_random2, fk_content_id=c_image.id)

    db.session.add(g_image)
    db.session.add(g_image2)
    db.session.add(g_image3)
    c_image.processing_status = 2
    db.session.commit()

    response = {
        'result': {
            'test_images': [file_name1, file_name2, file_name3]
        }
    }
    return response


def run_generate(file):
    selected_image = Generated.query.filter_by(file_name=file).first()
    content_image_id = selected_image.fk_content_id
    style_image_count = Style.query.filter_by(fk_content_id=content_image_id).count()
    content_img = Content.query.filter_by(id=content_image_id).first()
    content_img.processing_status = 3
    selected_image_id = selected_image.id
    type_var = selected_image.type
    db.session.commit()

    if type_var == 2:
        content_layers = ['block2_conv2']

        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1']
    else:
        content_layers = ['block5_conv2']

        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1',
                        'block4_conv1',
                        'block5_conv1']

    epochs = selected_image.epochs
    steps_per_epoch = selected_image.steps
    content_weight = selected_image.content_weight

    identifier = file[0:32]

    # layer count
    content_layers_count = len(content_layers)
    style_layers_count = len(style_layers)
    # style image count
    max_dim = 512
    total_variation_weight = 30
    content_img_name = identifier + 'content' + '.png'

    # preparing content image
    content_image = preprocess_image(os.path.join(app.config['UPLOAD_FOLDER'], content_img_name), max_dim)

    if style_image_count > 1:
        style_image = preprocess_image(os.path.join(app.config['UPLOAD_FOLDER'], identifier + 'style_m' + '.png')
                                       , max_dim)
    else:
        style_image = preprocess_image(os.path.join(app.config['UPLOAD_FOLDER'], identifier + 'style0' + '.png')
                                       , max_dim)

    extractor = StyleContentModel(style_layers, content_layers)
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    def style_content_loss(outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                               for name in style_outputs.keys()])
        style_loss *= style_weight / style_layers_count

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                                 for name in content_outputs.keys()])
        content_loss *= content_weight / content_layers_count
        loss = style_loss + content_loss
        return loss

    @tf.function()
    def train_step(img):
        with tf.GradientTape() as tape:
            outputs = extractor(img)
            loss = style_content_loss(outputs)
            loss += total_variation_weight * tf.image.total_variation(image)
        grad = tape.gradient(loss, img)
        opt.apply_gradients([(grad, img)])
        img.assign(clip_0_1(img))

    # initializing optimization variable
    image = tf.Variable(content_image)

    start = time.time()

    i = 1

    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(image)
            print(".", end='')
        print("Train step: {}".format(step))

    end = time.time()
    print("Total time: {:.1f}".format(end - start))

    file_name = identifier + 'generated' + '.png'
    g_image = Generated(file_name=file_name, type=type_var, content_weight=content_weight, selected=True,
                          custom=selected_image.custom, epochs=epochs, steps=steps_per_epoch,
                          fk_content_id=selected_image_id)

    db.session.add(g_image)
    selected_image.selected = True
    content_img.processing_status = 4
    db.session.commit()


    tensor_to_image(image).save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))

    return file_name


def run_rate(file_name, rating):
    selected_image = Generated.query.filter_by(file_name=file_name).first()
    selected_image.rating = rating
    db.session.commit()

    return 'ok'

