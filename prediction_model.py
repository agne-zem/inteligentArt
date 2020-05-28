from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from intelligentart import db
from intelligentart.models import Content, Style, Generated
import numpy as np
import pickle
from cv2 import cvtColor, imread, COLOR_BGR2HSV


def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]


def make_histogram(cluster):
    """
    Count the number of pixels in each cluster
    :param: KMeans cluster
    :return: numpy histogram
    """
    numLabels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    hist, _ = np.histogram(cluster.labels_, bins=numLabels)
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
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    hsv_bar = cvtColor(bar, COLOR_BGR2HSV)
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
    img = imread('./intelligentart/static/generated/' + img_name)
    height, width, _ = np.shape(img)

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
    stand_nuo = np.std(img)
    disp = stand_nuo**2
    return rgb_array, disp


X = []
y = []
data_rows_used = 0
generated_images = Generated.query.filter(Generated.rating != None)

for generated in generated_images:
    print(generated)
    style = Style.query.filter(Style.fk_content_id == generated.fk_content_id).filter(Style.used == True).first()
    print(style)
    content = Content.query.filter(Content.id == generated.fk_content_id).first()
    print(content)
    if style != None:
        style_name = style.file_name
        content_name = content.file_name

        style_result = get_rgb_disp(style_name)
        style_rgb = style_result[0]
        style_disp = style_result[1]
        print("style_rgb = " + str(style_result[0]))
        print("style_disp = " + str(style_result[1]))

        content_result = get_rgb_disp(content_name)
        content_rgb = content_result[0]
        content_disp = content_result[1]
        print("content_rgb = " + str(content_result[0]))
        print("content_disp = " + str(content_result[1]))

        X.append([content_rgb[0], content_rgb[1], content_rgb[2], content_rgb[3], content_rgb[4],
                  content_rgb[5], content_rgb[6], content_rgb[7], content_rgb[8], content_rgb[9],
                  content_rgb[10], content_rgb[11], content_rgb[12], content_rgb[13], content_rgb[14], style_disp,
                  style_rgb[0], style_rgb[1], style_rgb[2], style_rgb[3], style_rgb[4], style_rgb[5],
                  style_rgb[6], style_rgb[7], style_rgb[8], style_rgb[9], style_rgb[10], style_rgb[11],
                  style_rgb[12], style_rgb[13], style_rgb[14], content_disp, generated.rating])
        y.append([generated.epochs, generated.steps, generated.content_weight, generated.type])
        data_rows_used = data_rows_used + 1

print("x =" + str(X))
print("y =" + str(y))
print("model fitting")
# define model
model = KNeighborsRegressor()
# fit model
model.fit(X, y)
print("done")
filename = 'KNN_model.sav'
pickle.dump(model, open(filename, 'wb'))
print("saved")
print("data rows used: " + str(data_rows_used))
data_in = [[128, 144, 160, 224, 217, 204, 131, 96, 68, 69, 42, 27, 202, 153, 98,
            3222.785511159022, 80, 80, 45, 179, 87, 125, 231, 138, 106, 44, 62, 111, 104, 110, 169, 3693.4577149483, 10]]
yhat = model.predict(data_in)
print(int(round(yhat[0][0])))
print(int(round(yhat[0][1])))
print(int(round(yhat[0][2])))
print(int(round(yhat[0][3])))
