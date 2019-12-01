from keras.models import load_model
import pickle
import cv2
import uuid
from keras import backend as K

K.clear_session()
model = load_model('D:/ProjectsPy/passport/output2/smallvggnet.model')
lb = pickle.loads(open("D:/ProjectsPy/passport/output2/smallvggnet_lb.pickle", "rb").read())
image = cv2.imread('1.jpg')

# output = image.copy()
image = cv2.resize(image, (64, 64))
# масштабируем значения пикселей к диапазону [0, 1]
image = image.astype("float") / 255.0
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# загружаем модель и бинаризатор меток
print("[INFO] loading network and label binarizer...")
# делаем предсказание на изображении
preds = model.predict(image)
print(preds)
if preds[0][1] > 0.8:
    print('Это оригинал паспорта')
else:
    print('Это копия паспорта')




