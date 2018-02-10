from sklearn.ensemble import ExtraTreesRegressor
from selenium import webdriver
import time
from bs4 import BeautifulSoup
import pickle
import traceback
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, LeakyReLU
from keras.optimizers import RMSprop, Adam, Adadelta
import urllib.request
import io
import numpy as np
import random
import scipy.misc
from keras import optimizers
from PIL import Image
import glob

driver_loc = 'C:/Users/tdelforge/Documents/project_dbs/attractiveness_rater/'
image_loc = 'C:/Users/tdelforge/Documents/project_dbs/attractiveness_rater/images/'
image_size = 256

def scrape_images():
    driver = webdriver.PhantomJS(executable_path=driver_loc + 'phantomjs')
    try:
        with open(driver_loc + 'link_dict.plk', 'rb') as f:
            link_dict = pickle.load(f)

        driver.get('http://www.rankmyphotos.com/index.php')

        driver.find_element_by_css_selector('#sel_women').click()
        #driver.find_element_by_css_selector('#sel_bikini_pics').click()


        while True:
            link =  driver.find_element_by_css_selector('body > table:nth-child(3) > tbody > tr > td > table > tbody > tr > td:nth-child(2) > table > tbody > tr > td:nth-child(2) > img').get_attribute('src')
            time.sleep(2)
            driver.find_element_by_css_selector('#sel_six').click()

            time.sleep(2)

            rating = driver.find_element_by_css_selector('body > table:nth-child(3) > tbody > tr > td > table > tbody > tr > td:nth-child(3) > table.smallfont > tbody > tr:nth-child(5) > td:nth-child(2) > span').text

            if link in link_dict.keys():
                continue
            link_dict[link] = rating
            print(rating, link, len(link_dict.keys()))

            with open(driver_loc + 'link_dict.plk', 'wb') as f:
                pickle.dump(link_dict, f)
            # soup = BeautifulSoup(driver.page_source)
            # soup.find
    except:
        driver.close()
        raise Exception('Driver failed')


def store_pics():
    with open(driver_loc + 'link_dict.plk', 'rb') as f:
        link_dict = pickle.load(f)
    with open(driver_loc + 'storage_dict.plk', 'rb') as f:
        storage_dict = pickle.load(f)
    for link, rating in link_dict.items():
        try:
            file_name = link.split('/')[-1]
            if file_name in storage_dict.keys():
                continue

            urllib.request.urlretrieve(link, image_loc + file_name)
            print('stored:', file_name, rating)
            storage_dict[file_name] = rating

            with open(driver_loc + 'storage_dict.plk', 'wb') as f:
                pickle.dump(storage_dict, f)
        except:
            print('error with:', link)


def process_image_1(location):
    image_size = (256,256)

    image = Image.open(image_loc + location).convert('LA')
    np_image = np.array(image.getdata())[:,0]
    np_image = np.reshape(np_image, (image.size[1], image.size[0]))
    np_image = scipy.misc.imresize(np_image, image_size)
    np_image = np.reshape(np_image, (256, 256, 1))
    print(np_image.shape)

    return np_image



def process_images(test_size = .2):
    with open(driver_loc + 'storage_dict.plk', 'rb') as f:
        storage_dict = pickle.load(f)

    input_dict = dict()
    for count, (i, j) in enumerate(storage_dict.items()):
        if count > 25:
            break
        input_dict[i] = {'input_array': process_image_1(i), 'result': float(j)}

    testing_keys = random.sample(input_dict.keys(), int(test_size*len(input_dict.keys())))
    training_keys = [i for i in input_dict.keys() if i not in testing_keys]

    test_dict = {i: j for i, j in input_dict.items() if i in testing_keys}
    training_dict = {i: j for i, j in input_dict.items() if i in training_keys}



    return test_dict, training_dict



def create_model():
    nb_filters = 16
    nb_conv = 1

    model = Sequential()

    model.add(Conv2D(128, (3, 3), input_shape=(256, 256, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(16, 16)))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(8, 8)))

    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.add(Activation('softmax'))

    sgd = optimizers.SGD(lr=0.5, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model



def run_model():
    test_dict, training_dict = process_images()
    model = create_model()

    x_train = np.array([ i['input_array'] for i in training_dict.values()])
    y_train = np.vstack([ i['result'] for i in training_dict.values()])

    x_test = np.array([ i['input_array'] for i in test_dict.values()])
    y_test = np.vstack([ i['result'] for i in test_dict.values()])

    print(x_train.shape)
    model.fit(x_train, y_train, epochs=5)

    score = model.evaluate(x_test, y_test, verbose=0)
    print(score)

    for i, j in test_dict.items():
        print(i, model.predict(np.reshape(j['input_array'], (1, 256, 256, 1))), j['result'])



if __name__ == '__main__':
    run_model()