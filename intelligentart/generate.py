from __future__ import absolute_import
from tensorflow.keras import optimizers
from PIL import Image
from intelligentart import app
from intelligentart.models import Content, Style, Generated
from intelligentart import db
from sklearn.cluster import KMeans
from os import path
from io import BytesIO
from time import time
from random import randint
from numpy import histogram, std, shape, array, arange, unique, ndim, uint8, zeros
from pickle import load
from base64 import b64decode
import tensorflow as tf
import cv2

tf.compat.v1.enable_eager_execution()

style_weight = 1e-2
opt = optimizers.Adam(learning_rate=0.01, beta_1=0.99, epsilon=1e-1)

# =================== HELPERS ====================


def unique_count_app(a):
    colors, count = unique(a.reshape(-1, a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]


def make_histogram(cluster):
    """
    Count the number of pixels in each cluster
    :param: KMeans cluster
    :return: numpy histogram
    """
    numLabels = arange(0, len(unique(cluster.labels_)) + 1)
    hist, _ = histogram(cluster.labels_, bins=numLabels)
    hist = hist.astype('float32')
    hist /= hist.sum()
    return hist


def make_bar(height, width, color):
    """
    Create an image of a given color
    :param: height of the image
    :param: width of the image
    :param: BGR pixel values of the color
    :return: tuple of bar, rgb values, and hsv values
    """
    bar = zeros((height, width, 3), uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    hsv_bar = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv_bar[0][0]
    return bar, (red, green, blue), (hue, sat, val)


def sort_hsvs(hsv_list):
    """
    Sort the list of HSV values
    :param hsv_list: List of HSV tuples
    :return: List of indexes, sorted by hue, then saturation, then value
    """
    bars_with_indexes = []
    for index, hsv_val in enumerate(hsv_list):
        bars_with_indexes.append((index, hsv_val[0], hsv_val[1], hsv_val[2]))
    bars_with_indexes.sort(key=lambda elem: (elem[1], elem[2], elem[3]))
    return [item[0] for item in bars_with_indexes]


def get_rgb_disp(img_name):
    rgb_array = []
    disp = 0
    img = cv2.imread('./intelligentart/static/generated/' + img_name)
    height, width, _ = shape(img)

    image = img.reshape((height * width, 3))

    num_clusters = 5
    clusters = KMeans(n_clusters=num_clusters)
    clusters.fit(image)

    # count the dominant colors and put them in "buckets"
    histogram = make_histogram(clusters)
    # then sort them, most-common first
    combined = zip(histogram, clusters.cluster_centers_)
    combined = sorted(combined, key=lambda x: x[0], reverse=True)

    # finally, we'll output a graphic showing the colors in order
    bars = []
    hsv_values = []
    for index, rows in enumerate(combined):
        bar, rgb, hsv = make_bar(100, 100, rows[1])
        # print(f'Bar {index + 1}')
        print(f'  RGB values: {rgb}')
        rgb_array.append(rgb[0])
        rgb_array.append(rgb[1])
        rgb_array.append(rgb[2])
        # print(f'  HSV values: {hsv}')
        hsv_values.append(hsv)
        bars.append(bar)

    # sort the bars[] list so that we can show the colored boxes sorted
    # by their HSV values -- sort by hue, then saturation
    sorted_bar_indexes = sort_hsvs(hsv_values)
    sorted_bars = [bars[idx] for idx in sorted_bar_indexes]

    # cv2.imshow('Sorted by HSV values', np.hstack(sorted_bars))
    # cv2.imshow(f'{num_clusters} Most Common Colors', np.hstack(bars))
    color = unique_count_app(img)
    stand_nuo = std(img)
    disp = stand_nuo**2
    return rgb_array, disp

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
    tensor = array(tensor, dtype=uint8)
    if ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


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
    img1 = Image.open(path.join(app.config['UPLOAD_FOLDER'], identifier.hex + 'style0' + '.png'))
    img2 = Image.open(path.join(app.config['UPLOAD_FOLDER'], identifier.hex + 'style1' + '.png'))
    img2 = img2.resize((int(img2.width * img1.height / img2.height), img1.height), resample=resample)

    merged = Image.new('RGB', (img1.width + img2.width, img1.height))
    merged.paste(img1, (0, 0))
    merged.paste(img2, (img1.width, 0))

    if style_image_count == 3:
        img3 = Image.open(path.join(app.config['UPLOAD_FOLDER'], identifier.hex + 'style2' + '.png'))

        img3 = img3.resize((merged.width, int(img3.height * merged.width / img3.width)), resample=resample)

        merged3 = Image.new('RGB', (merged.width, merged.height + img3.height))
        merged3.paste(merged, (0, 0))
        merged3.paste(img3, (0, merged.height))
        merged = merged3

    if style_image_count == 4:
        img3 = Image.open(path.join(app.config['UPLOAD_FOLDER'], identifier.hex + 'style2' + '.png'))
        img4 = Image.open(path.join(app.config['UPLOAD_FOLDER'], identifier.hex + 'style3' + '.png'))

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
    merged.save(path.join(app.config['UPLOAD_FOLDER'], style_image_name))
    return style_image_name


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
    decoded_content_image = b64decode(encoded_content_image)
    decoded_style_images = []
    # decoding json style images
    # getting array of decoded image data
    for i in range(0, style_image_count, 1):
        decoded_style_images.append(b64decode(encoded_style_images[i]))

    # converting back to image
    # this is needed in order to save images in folder for further use

    # converting content image and saving it
    content_image = Image.open(BytesIO(decoded_content_image))
    content_image = shrink(content_image, max_dim)
    content_img_name = identifier.hex + 'content' + '.png'
    content_image.save(path.join(app.config['UPLOAD_FOLDER'], content_img_name))

    # creating content image query for database
    c_image = Content(file_name=content_img_name, processing_status=1)
    db.session.add(c_image)
    db.session.commit()


    # converting style images and saving them
    for i in range(0, style_image_count, 1):
        style_image = Image.open(BytesIO(decoded_style_images[i]))
        style_image = shrink(style_image, max_dim)
        style_img_name = identifier.hex + 'style' + str(i) + '.png'
        style_image.save(path.join(app.config['UPLOAD_FOLDER'], style_img_name))

        # creating style image query for database
        s_image = Style(file_name=style_img_name, fk_content_id=c_image.id)
        db.session.add(s_image)
        db.session.commit()


    max_dim = 128

    # preparing content image
    content_image = preprocess_image(path.join(app.config['UPLOAD_FOLDER'], content_img_name)
                                     , max_dim)



    # merging style images together into one
    if style_image_count > 1:
        style_image_name = merge_and_save(identifier, style_image_count)
        s_image = Style(file_name=style_image_name, used=True, fk_content_id=c_image.id)
        db.session.add(s_image)
        db.session.commit()
        style_image = preprocess_image(path.join(app.config['UPLOAD_FOLDER'], identifier.hex + 'style_m' + '.png')
                                       , max_dim)
    else:
        style_image_name = identifier.hex + 'style0' + '.png'
        s_image = Style.query.filter_by(file_name=style_image_name).first()
        s_image.used = True
        db.session.commit()
        style_image = preprocess_image(path.join(app.config['UPLOAD_FOLDER'], identifier.hex + 'style0' + '.png')
                                       , max_dim)

    file_name3 = run_prediction(identifier, content_image, style_image, content_img_name, style_image_name,  c_image)

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

    start = time()

    i = 1

    ep_random1 = randint(1, 5)
    stp_random1 = randint(1, 50)
    file_name1 = identifier.hex + 'test_generated1' + '.png'
    file_name2 = identifier.hex + 'test_generated2' + '.png'

    ep_max = max(ep_random1, epochs)
    stp_max = max(stp_random1, steps_per_epoch)
    step = 0
    for n in range(ep_max):
        for m in range(stp_max):
            step += 1
            train_step(image)
            # save
            if steps_per_epoch - 1 == m and epochs - 1 == n:
                tensor_to_image(image).save(path.join(app.config['UPLOAD_FOLDER'], file_name1))
            if stp_random1 - 1 == m and ep_random1 - 1 == n:
                tensor_to_image(image).save(path.join(app.config['UPLOAD_FOLDER'], file_name2))
            print(".", end='')
        print("Train step: {}".format(step))

    end = time()
    print("Total time: {:.1f}".format(end - start))

    if style_layers_count > 3:
        type_var = 1
    else:
        type_var = 2

    # creating generated image query for database
    g_image = Generated(file_name=file_name1, type=type_var, content_weight=content_weight, configuration_type=1,
                        epochs=epochs, steps=steps_per_epoch, fk_content_id=c_image.id)
    g_image2 = Generated(file_name=file_name2, type=type_var, content_weight=content_weight, configuration_type=2
                         , epochs=ep_random1, steps=stp_random1, fk_content_id=c_image.id)

    db.session.add(g_image)
    db.session.add(g_image2)
    c_image.processing_status = 2
    db.session.commit()

    response = {
        'result': {
            'test_images': [file_name1, file_name2, file_name3]
        }
    }
    return response


def run_prediction(identifier, content_image, style_image, content_img_name, style_image_name, c_image):
    file_name = identifier.hex + 'test_generated3' + '.png'

    style_result = get_rgb_disp(style_image_name)
    style_rgb = style_result[0]
    style_disp = style_result[1]

    content_result = get_rgb_disp(content_img_name)
    content_rgb = content_result[0]
    content_disp = content_result[1]

    data_in = [[content_rgb[0], content_rgb[1], content_rgb[2], content_rgb[3], content_rgb[4],
              content_rgb[5], content_rgb[6], content_rgb[7], content_rgb[8], content_rgb[9],
              content_rgb[10], content_rgb[11], content_rgb[12], content_rgb[13], content_rgb[14], style_disp,
              style_rgb[0], style_rgb[1], style_rgb[2], style_rgb[3], style_rgb[4], style_rgb[5],
              style_rgb[6], style_rgb[7], style_rgb[8], style_rgb[9], style_rgb[10], style_rgb[11],
              style_rgb[12], style_rgb[13], style_rgb[14], content_disp, 10]]

    loaded_model = load(open("KNN_model.sav", 'rb'))
    print(data_in)
    yhat = loaded_model.predict(data_in)

    epochs = int(round(yhat[0][0]))
    steps = int(round(yhat[0][1]))
    content_weight = int(round(yhat[0][2]))
    type = int(round(yhat[0][3]))

    print("epochs predicted: " + str(epochs))
    print("steps predicted: " + str(steps))
    print("content_weight predicted: " + str(content_weight))
    print("type predicted: " + str(type))

    if type == 2:
        content_layers = ['block2_conv2']

        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1']
    else:
        content_layers = ['block4_conv2']

        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1',
                        'block4_conv1',
                        'block5_conv1']

    style_layers_count = len(style_layers)
    content_layers_count = len(content_layers)
    total_variation_weight = 30



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

    start = time()

    step = 0
    for n in range(epochs):
        for m in range(steps):
            step += 1
            train_step(image)
            print(".", end='')
        print("Train step: {}".format(step))

    end = time()
    print("Total time: {:.1f}".format(end - start))

    tensor_to_image(image).save(path.join(app.config['UPLOAD_FOLDER'], file_name))

    # creating generated image query for database
    g_image3 = Generated(file_name=file_name, type=type, content_weight=content_weight, configuration_type=3,
                        epochs=epochs, steps=steps, fk_content_id=c_image.id)

    db.session.add(g_image3)
    db.session.commit()

    return file_name


def run_generate(file):
    selected_image = Generated.query.filter_by(file_name=file).first()
    content_image_id = selected_image.fk_content_id
    style_image_count = Style.query.filter_by(fk_content_id=content_image_id).count()
    content_img = Content.query.filter_by(id=content_image_id).first()
    content_img.processing_status = 3
    selected_image_id = selected_image.fk_content_id
    type_var = selected_image.type
    db.session.commit()

    if type_var == 2:
        content_layers = ['block2_conv2']

        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1']
    else:
        content_layers = ['block4_conv2']

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
    content_image = preprocess_image(path.join(app.config['UPLOAD_FOLDER'], content_img_name), max_dim)

    if style_image_count > 1:
        style_image = preprocess_image(path.join(app.config['UPLOAD_FOLDER'], identifier + 'style_m' + '.png')
                                       , max_dim)
    else:
        style_image = preprocess_image(path.join(app.config['UPLOAD_FOLDER'], identifier + 'style0' + '.png')
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

    start = time()

    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(image)
            print(".", end='')
        print("Train step: {}".format(step))

    end = time()
    print("Total time: {:.1f}".format(end - start))

    file_name = identifier + 'generated' + '.png'
    g_image = Generated(file_name=file_name, type=type_var, content_weight=content_weight, selected=True,
                        configuration_type=selected_image.configuration_type, epochs=epochs, steps=steps_per_epoch,
                        fk_content_id=selected_image_id)

    db.session.add(g_image)
    selected_image.selected = True
    content_img.processing_status = 4
    db.session.commit()

    tensor_to_image(image).save(path.join(app.config['UPLOAD_FOLDER'], file_name))

    return file_name


def run_rate(file_name, rating):
    selected_image = Generated.query.filter_by(file_name=file_name).first()
    selected_image.rating = rating
    db.session.commit()

    return 'ok'