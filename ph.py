from sklearn.ensemble import ExtraTreesRegressor
from selenium import webdriver
import time
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pickle
import traceback
import re
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
from keras import backend as K


driver_loc = 'C:/Users/tdelforge/Documents/project_dbs/attractiveness_rater/'
image_loc = 'C:/Users/tdelforge/Documents/project_dbs/attractiveness_rater/images/'
image_size = 256
training_image_size_len = 200

min_votes = 100


def get_image_links():
    try:
        with open(driver_loc + 'ph_image_url_list.plk', 'rb') as out_file:
            pic_links = pickle.load(out_file)
    except:
        pic_links = []


    link = 'https://www.pornhub.com/albums/female?o=mv&t=a'
    options = webdriver.ChromeOptions()
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36')
    driver = webdriver.Chrome(executable_path=driver_loc + 'chromedriver', chrome_options=options)

    driver.get(link)
    album_links = []
    for _ in range(5):
        try:
            albums = driver.find_elements_by_css_selector('li[class="photoAlbumListContainer"]')


            for a in albums:
                album_links.append(a.find_element_by_css_selector('a').get_attribute('href'))
            print(len(album_links), len(set(album_links)))
            time.sleep(2)
            driver.find_element_by_css_selector('li[class="page_next omega"]').find_element_by_css_selector('a').click()
            time.sleep(4)
        except:
            pass
            break

    for a in album_links:
        driver.get(a)
        photos = driver.find_elements_by_css_selector('li[class="photoAlbumListContainer"]')

        for p in photos:
            pic_links.append(p.find_element_by_css_selector('a').get_attribute('href'))
        print(len(pic_links), len(set(pic_links)))

    with open(driver_loc + 'ph_image_url_list.plk', 'wb') as out_file:
        pickle.dump(list(set(pic_links)), out_file)
    driver.close()

def url_to_file_name(url):
    extention = url.split('.')[-1]
    file_name = '.'.join(url.split('.')[0:-1])
    return  re.sub(r'[^a-zA-Z\d]', '_', file_name) + '.' + extention

def scrape_images():
    get_image_links()
    with open(driver_loc + 'ph_image_url_list.plk', 'rb') as out_file:
        pic_links = pickle.load(out_file)
        random.shuffle(pic_links)
    try:
        with open(driver_loc + 'ph_score_dict.plk', 'rb') as out_file:
            score_dict = pickle.load(out_file)
    except:
        score_dict = dict()

    options = webdriver.ChromeOptions()
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36')
    driver = webdriver.Chrome(executable_path=driver_loc + 'chromedriver', chrome_options=options)


    for count, i in enumerate(pic_links):
        try:
            driver.get(i)
            soup = BeautifulSoup(driver.page_source)
            vote_count = int(soup.find('span', {'id':'voteCountNumber'}).text)
            if vote_count < min_votes:
                continue
            score = int(soup.find('span', {'id':'votePercentageNumber'}).text)
            url = soup.find('div', {'id':'photoImageSection'}).find('img')['src']
            file_name = url_to_file_name(url)
            if 'gif' in url or file_name in score_dict.keys():
                continue
            print(score, vote_count, url)


            urllib.request.urlretrieve(url, image_loc + file_name)
            print('stored:', file_name, score, len(score_dict.keys()))
            score_dict[file_name] = score

            with open(driver_loc + 'ph_score_dict.plk', 'wb') as out_file:
                pickle.dump(score_dict, out_file)

        except:
            traceback.print_exc()


def process_image_1(location):
    image_size = (training_image_size_len,training_image_size_len)

    image = Image.open(image_loc + location).convert('LA')
    np_image = np.array(image.getdata())[:,0]
    np_image = np.reshape(np_image, (image.size[1], image.size[0]))

    print(np_image.shape)

    min_len = min(image.size[1], image.size[0])
    np_image=np_image[0:min_len, 0:min_len]

    np_image = scipy.misc.imresize(np_image, image_size)
    np_image = np.reshape(np_image, (training_image_size_len, training_image_size_len, 1))

    np_images = []
    image_versions = []
    image_versions.append(np.fliplr(np_image))
    image_versions.append(np.flipud(np_image))
    image_versions.append(np_image)

    for i in image_versions:
        np_images.append(np.rot90(i, 1))
        np_images.append(np.rot90(i, 2))
        np_images.append(np.rot90(i, 3))
        np_images.append(np.rot90(i, 4))

    return np_images



def process_images(test_size = .1):
    with open(driver_loc + 'ph_score_dict.plk', 'rb') as f:
        storage_dict = pickle.load(f)

    input_dict = dict()
    input_list = []
    for count, (i, j) in enumerate(storage_dict.items()):
        try:
            np_images = process_image_1(i)
            for k in np_images:
                input_list.append({'input_array': k, 'result': float(j)})
        except:
            traceback.print_exc()
    random.shuffle(input_list)
    test = input_list[0:int(test_size*len(input_list))]
    train = input_list[int(test_size * len(input_list)):]

    return test, train


def create_model():
    nb_filters = 16
    nb_conv = 1

    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(training_image_size_len, training_image_size_len, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Dense(1))

    #model.add(Activation('softmax'))

    sgd = optimizers.SGD(lr=0.01, decay=1e-9, momentum=0.0, nesterov=False)

    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mae', coeff_determination])
    return model


def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def run_model():
    test, train = process_images()
    model = create_model()

    x_train = np.array([ i['input_array'] for i in train])
    y_train = np.vstack([ i['result'] for i in train])

    x_test = np.array([ i['input_array'] for i in test])
    y_test = np.vstack([ i['result'] for i in test])

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    model.fit(x_train, y_train, epochs=1000, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print(score)

if __name__ == '__main__':
    scrape_images()
    run_model()








