from keras.models import Sequential
from keras.models import load_model

classifier = load_model("model_ofaaaa_train.h5")
import numpy as np
from keras.preprocessing import image
def predict(dire):
    test_image = image.load_img(dire, target_size = (224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)

    print(result)
    if result[0][1] > 0.5:
        prediction = 'Pneumonia'
    else:
        prediction = 'Normal'
    print()
    print()
    print()
    print(prediction)


#predict('test/d.jpeg')


predict('test/N.jpeg')


