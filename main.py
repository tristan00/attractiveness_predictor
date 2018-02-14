from sklearn.ensemble import ExtraTreesRegressor
from selenium import webdriver
import time
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
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

from keras import backend as K

driver_loc = 'C:/Users/tdelforge/Documents/project_dbs/attractiveness_rater/'
image_loc = 'C:/Users/tdelforge/Documents/project_dbs/attractiveness_rater/images/'
image_size = 256
training_image_size_len = 200


def add_manual_images():
    #site blocking me
    with open(driver_loc + 'link_dict.plk', 'rb') as f:
        link_dict = pickle.load(f)

    solutions = {'http://www.rankmyphotos.com/pics/img5689c818d419b.jpg':'6.01',
                 'http://www.rankmyphotos.com/pics/img55f35067eee1a.jpg':'7.39',
                 'http://www.rankmyphotos.com/pics/img59cae1a811e38.jpg':'6.11',
                 'http://www.rankmyphotos.com/pics/img58276e0205dc9.jpg':'6.89',
                 'http://www.rankmyphotos.com/pics/img5765726bd04b7.jpg':'5.17',
                 'http://www.rankmyphotos.com/pics/img5a21ccd550eef.jpg':'5.01',
                 'http://www.rankmyphotos.com/pics/img52941916b8f3c.jpg':'6.03',
                 'http://www.rankmyphotos.com/pics/img56c1ddba60446.jpg':'6.06',
                 'http://www.rankmyphotos.com/pics/img5779cc4f3c0ad.jpg':'5.96',
                 'http://www.rankmyphotos.com/pics/img56c2810d0d061.jpg':'4.46',
                 'http://www.rankmyphotos.com/pics/img50e249701736a.jpg':'6.56',
                 'http://www.rankmyphotos.com/pics/img57921e19d2d7a.jpg':'6.04',
                 'http://www.rankmyphotos.com/pics/img53e63d03671c1.jpg':'7.31',
                 'http://www.rankmyphotos.com/pics/img5480cc49d4ca0.jpg':'5.22',
                 'http://www.rankmyphotos.com/pics/img5398d20b55b6b.jpg':'7.46',
                 'http://www.rankmyphotos.com/pics/img579936d4967ed.jpg':'7.11',
                 'http://www.rankmyphotos.com/pics/img587d23b1be0a8.jpg':'6.35',
                 'http://www.rankmyphotos.com/pics/img5940f2838e4e3.jpg':'6.23',
                 'http://www.rankmyphotos.com/pics/img531d30a1deefd.jpg':'6.4',
                 'http://www.rankmyphotos.com/pics/img577c2af8a955a.jpg':'5.98',
                 'http://www.rankmyphotos.com/pics/img5723b769cefb9.jpg':'5.34',
                 'http://www.rankmyphotos.com/pics/img561dc9b30156d.jpg':'6.96',
                 'http://www.rankmyphotos.com/pics/img54201df5e98e5.jpg':'6.74',
                 'http://www.rankmyphotos.com/pics/img57426e28bf6da.jpg':'5.83',
                 'http://www.rankmyphotos.com/pics/img57cbb8a3ae0d3.jpg':'7.06',
                 'http://www.rankmyphotos.com/pics/img5603bfc3827b9.jpg':'7.24',
                 'http://www.rankmyphotos.com/pics/img535a659f76f95.jpg':'6.89',
                 'http://www.rankmyphotos.com/pics/img59c0a5077044c.jpg':'6.09',
                 'http://www.rankmyphotos.com/pics/img5738fada0231f.jpg':'4.67',
                 'http://www.rankmyphotos.com/pics/img575c03c3425f4.jpg':'6.76',
                 'http://www.rankmyphotos.com/pics/img58a1dee091ac9.jpg':'5.36',
                 'http://www.rankmyphotos.com/pics/img50523e105b51a.jpg':'6.78',
                 'http://www.rankmyphotos.com/pics/img56a49a98cdea5.jpg':'5.78',
                 'http://www.rankmyphotos.com/pics/img56e0835e968d8.jpg':'5.77',
                 'http://www.rankmyphotos.com/pics/img5807eb4c504c1.jpg': '5.67',
                 'http://www.rankmyphotos.com/pics/img5663f5108ddbb.jpg': '6.2',
                 'http://www.rankmyphotos.com/pics/img56c69dbb3fc8f.jpg': '7.2',
                 'http://www.rankmyphotos.com/pics/img5615e0eb4de67.jpg': '4.56',
                 'http://www.rankmyphotos.com/pics/img55ecdff44b524.jpg': '6.21',
                 'http://www.rankmyphotos.com/pics/img56be61e5d87c8.jpg': '5.7'
             }
    link_dict.update(solutions)
    with open(driver_loc + 'link_dict.plk', 'wb') as f:
        pickle.dump(link_dict, f)


def scrape_images():
    options = webdriver.ChromeOptions()
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36')
    driver = webdriver.Chrome(executable_path=driver_loc + 'chromedriver', chrome_options=options)
    #driver = webdriver.PhantomJS(executable_path=driver_loc + 'phantomjs')
    try:
        try:
            with open(driver_loc + 'link_dict.plk', 'rb') as f:
                link_dict = pickle.load(f)
        except:
            link_dict = dict()

        driver.get('http://www.rankmyphotos.com/index.php')

        if random.randint(1, 2) == 1:
            driver.find_element_by_css_selector('#sel_women').click()
        else:
            driver.find_element_by_css_selector('#sel_bikini_pics').click()


        for _ in range(500):
            time.sleep(random.randint(10, 30))
            link = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR,
                                                "body > table:nth-child(3) > tbody > tr > td > table > tbody > tr > td:nth-child(2) > table > tbody > tr > td:nth-child(2) > img"))).get_attribute(
                'src')
            time.sleep(random.randint(10, 30))
            try:
                if random.randint(1,2) == 1:
                    driver.find_element_by_css_selector('#sel_five').click()
                else:
                    driver.find_element_by_css_selector('#sel_seven').click()
            except:
                if random.randint(1,2) == 1:
                    driver.find_element_by_css_selector('#sel_women').click()
                else:
                    driver.find_element_by_css_selector('#sel_bikini_pics').click()
                continue

            time.sleep(random.randint(10, 30))


            rating = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR,
                                                'body > table:nth-child(3) > tbody > tr > td > table > tbody > tr > td:nth-child(3) > table.smallfont > tbody > tr:nth-child(5) > td:nth-child(2) > span'))).text

            if link in link_dict.keys():
                print('already found. ', link)
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
    driver.close()


def scrape_images2():
    driver = webdriver.Chrome(executable_path=driver_loc + 'chromedriver')
    #driver = webdriver.PhantomJS(executable_path=driver_loc + 'phantomjs')
    try:
        try:
            with open(driver_loc + 'link_dict.plk', 'rb') as f:
                link_dict = pickle.load(f)
        except:
            link_dict = dict()

        driver.get('http://www.pictures2rate.com')

        for _ in range(10000):
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR,
                                                'body > table > tbody > tr:nth-child(1) > td > a.headerbuttongirls'))).click()

            link = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR,
                                                'img[name="mainimg"]'))).get_attribute('src')


            rating = driver.find_element_by_css_selector(
                'body > table > tbody > tr:nth-child(2) > td > table:nth-child(4) > tbody > tr > td:nth-child(2) > table > tbody > tr:nth-child(2) > td:nth-child(4)').text.split(
                ':')[1]

            if link in link_dict.keys():
                print('already found. ', link)
                continue
            link_dict[link] = rating
            print(rating, link, len(link_dict.keys()))

            with open(driver_loc + 'link_dict.plk', 'wb') as f:
                pass
                pickle.dump(link_dict, f)
            # soup = BeautifulSoup(driver.page_source)
            # soup.find
    except:
        driver.close()
        raise Exception('Driver failed')
    driver.close()


def store_pics():
    with open(driver_loc + 'link_dict.plk', 'rb') as f:
        link_dict = pickle.load(f)

    try:
        with open(driver_loc + 'storage_dict.plk', 'rb') as f:
            storage_dict = pickle.load(f)
    except:
        storage_dict = dict()
    for link, rating in link_dict.items():
        try:
            extention = link.split('.')[-1]
            file_name = '.'.join(link.split('.')[0:-1])
            file_name = file_name.replace(r'/','_').replace(r':','_').replace(r'.','_') + '.' + extention
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
    with open(driver_loc + 'storage_dict.plk', 'rb') as f:
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
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(3, 3)))

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

    sgd = optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.0, nesterov=False)

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
    add_manual_images()
    store_pics()
    run_model()

    # store_pics()

    #
    # while True:
    #     print(1)
    #     try:
    #         scrape_images()
    #     except:
    #         pass
    #         traceback.print_exc()
    #         time.sleep(1800)
        # print(2)
        # try:
        #     scrape_images2()
        # except:
        #     pass
        #     traceback.print_exc()
