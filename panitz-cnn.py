###################################################################################
### 1. Tensorflow
###################################################################################
'''
Stellen Sie zuerst sicher dass Tensorflow mindestens in Version 1.10
vorhanden ist.
'''
import tensorflow as tf

print(tf.__version__)

###################################################################################
### 2. Daten einlesen
###################################################################################
'''
Prof. Panitz macht von sich tägliche Selfies. Diese lesen wir ein und verwenden Sie 
für PanitzNetz. In der Datei

    panitznet.zip 

die Sie aus dem Read.MI heruntergeladen haben befinden sich die Selfies unter 

    imgs/small/*

Führen Sie den folgenden Code aus. Passen Sie vorher ggfs. die Variable PATH an. 
Es sollten ca. 1800 Bilder der Dimension 32×32×3 eingelesen werden. Am Ende 
wird eines der Bilder als Beispiel geplottet.

HINWEIS: Sollten Sie auf dem Server supergpu arbeiten wollen, werden beim Plotten
         von Daten evtl. noch Fehler auftauchen. Wir besprechen dies im Praktikum.
'''

from datetime import timedelta, date
from PIL import Image
import os
import numpy as np
import random
import matplotlib.pyplot as plt

PATH = 'imgs/small'
D = 32


def read_jpg(path):
    '''liest ein JPEG ein und gibt ein DxDx3-Numpy-Array zurück.'''
    img = Image.open(path)
    w, h = img.size
    # schneide etwas Rand ab.
    img = img.crop((5, 24, w - 5, h - 24))
    # skaliere das Bild
    img = img.resize((D, D), Image.ANTIALIAS)
    img = np.asarray(img)
    return img


def read_panitz(directory):
    def daterange(start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    start_date = date(2010, 10, 30)
    end_date = date(2019, 1, 1)

    imgs = []

    for date_ in daterange(start_date, end_date):
        img_path = '%s/small-b%s.jpg' % (directory, date_.strftime("%Y%m%d"))
        if os.path.exists(img_path):
            img = read_jpg(img_path)
            imgs.append(img)

    return np.array(imgs)


imgs = read_panitz(PATH)

print('Dimension der gelesenen Bilder:', imgs.shape)

test_imgs = []
while len(test_imgs) <= 10:
    test_imgs.append(imgs[random.randint(0, len(imgs))])

# zeigt ein Bild
# plt.imshow(imgs[17])
# plt.show()


###################################################################################
### 3. Hifsmethode zum Plotten
###################################################################################
'''
Während wir PanitzNet trainieren, möchten wir beobachten wie die Rekonstruktionen
des Netzes den Eingabebildern immer ähnlicher werden. Hierzu können Sie die 
folgende Methode verwenden: Übergeben Sie eine Liste von z.B. 10 Bildern (imgs) 
und die  zugehörigen Rekonstruktionen Ihres Netzes (recs) als Listen von 
numpy-Arrays. Es sollte ein Plot erstellt werden, in dem Sie neben jedem Bild 
die Rekonstruktion sehen, ähnlich dem Bild

   panitzplot.png

Überprüfen Sie kurz die Methode, indem Sie 10 zufällige Bilder und (anstelle der 
noch nicht vorhandenen Rekonstruktionen) noch einmal dieselben Bilder übergeben. 
'''


def plot_reconstructions(imgs, recs):
    # Erstellt ein NxN-Grid zum Plotten der Bilder
    N = int(np.ceil(np.sqrt(2 * len(imgs))))
    f, axarr = plt.subplots(nrows=N, ncols=N, figsize=(18, 18))

    # Fügt die Bilder in den Plot ein
    for i in range(min(len(imgs), 100)):
        axarr[2 * i // N, 2 * i % N].imshow(imgs[i].reshape((D, D, 3)),
                                            interpolation='nearest')
        axarr[(2 * i + 1) // N, (2 * i + 1) % N].imshow(recs[i].reshape((D, D, 3)),
                                                        interpolation='nearest')
    f.tight_layout()
    plt.show()


###################################################################################
### 4. Vorverarbeitung
###################################################################################
'''
Momentan ist jedes der Bild noch ein D×D×3-Tensor. Machen Sie hieraus einen 
eindimensionalen Vektor. Skalieren Sie den Pixelbereich außerdem von 0,...,255 
auf [0,1].
'''

imgs = imgs / 255.0  # normalize values to [0,1]
#imgs = imgs.reshape((len(imgs), np.prod(imgs.shape[1:])))  # flatten tensor to 1d array
train_imgs = imgs[:1000]
test_imgs = imgs[len(train_imgs):len(imgs)]  # split test values from dataset

#5

model = tf.keras.Sequential()
#model.add(tf.keras.layers.Conv2D(32,(3,3), padding='same', input_shape=(32,32,3), activation='relu'))
#model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(8,(3,3), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(16,(3,3), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(24,(3,3), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(4*4*24, activation='relu'))
model.add(tf.keras.layers.Reshape((4,4,24)))

model.add(tf.keras.layers.UpSampling2D(size=(2,2))) # 8,8,24
model.add(tf.keras.layers.Conv2D(16,(3,3), padding='same', activation='relu'))

model.add(tf.keras.layers.UpSampling2D(size=(2,2))) # 16,16,16
model.add(tf.keras.layers.Conv2D(8,(3,3), padding='same', activation='relu'))

model.add(tf.keras.layers.UpSampling2D(size=(2,2))) # 32,32,8
model.add(tf.keras.layers.Conv2D(16,(3,3), padding='same', activation='relu'))

model.add(tf.keras.layers.Conv2D(3,(3,3), padding='same', activation='sigmoid'))
#model.add(tf.keras.layers.UpSampling2D(size=(2,2)))
#model.add(tf.keras.layers.Conv2D(3,(3,3), padding='same', activation='relu'))

#model.summary()

model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])


EPOCHS = 5
_ = model.predict(train_imgs)
model.summary()
history = model.fit(train_imgs,train_imgs,epochs=EPOCHS, validation_data=(test_imgs,test_imgs))
results = model.evaluate(test_imgs,test_imgs)
print("Results:",results)

anomaly = []
a_orig = []
a_dec = []
decoded = model.predict(test_imgs)
for i in range(len(test_imgs)):
    diff = test_imgs[i]-decoded[i]
    loss = np.sum(diff*diff)
    anomaly.append((test_imgs[i],decoded[i],loss))

anomaly = sorted(anomaly,key=lambda tup: tup[2], reverse=True)[:20]

for x,y,z in anomaly:
    a_orig.append(x)
    a_dec.append(y)

plot_reconstructions(a_orig,a_dec)




history_dict = history.history
acc = history_dict['acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1,len(acc)+1)

plt.plot(epochs,loss,'bo',label = 'Training loss')
plt.plot(epochs,val_loss,'b',label= 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()






