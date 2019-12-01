from flask import Flask,request
import uuid
import cv2
from keras.models import load_model
from keras import backend as K

app = Flask(__name__)

@app.route("/file", methods=['GET', 'POST'])
def file():
    K.clear_session()
    if request.method == 'GET':
        return 'Не верный запрос'
    if request.method == 'POST':
        model = load_model('./output/smallvggnet.model')

        key = str(uuid.uuid4())
        data = request.files[''].read()

        with open("./new_photo/test"+key+".jpg", mode="wb") as new:
            new.write(data)

        image = cv2.imread("./new_photo/test"+key+".jpg")

        image = cv2.resize(image, (64, 64))

        # масштабируем значения пикселей к диапазону [0, 1]
        image = image.astype("float") / 255.0
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

        # загружаем модель
        preds = model.predict(image)

        if preds[0][1] > 0.8:
            print('Это оригинал паспорта')
            label = 'original'
        else:
            print('Это копия паспорта')
            label = 'copy'

    return label

if __name__ == "__main__":
    app.run(host="0.0.0.0")