from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(64, (3, 3), input_shape = (224, 224, 3), activation = 'relu'))
# Step 2 - Pooling


classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))    
classifier.add(Dropout(0.2))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(units = 2, activation = 'softmax'))
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator
training_set = ImageDataGenerator().flow_from_directory('chest_xray/train',target_size = (224, 224),batch_size = 32)
valid_set =ImageDataGenerator().flow_from_directory('chest_xray/val',target_size = (224, 224),batch_size = 1)
testing_set =ImageDataGenerator().flow_from_directory('chest_xray/test',target_size = (224, 224),batch_size = 26)
imgs, labels= next(training_set)


classifier.fit_generator(training_set,steps_per_epoch = 163 ,epochs = 1,validation_data = valid_set,validation_steps = 16)
classifier.save("model_of_train.h5")
#classifier.fit(training_set,batch_size = 32,epochs = 1, verbose = 1)
#classifier.fit_generator(training_set,steps_per_epoch = 224,epochs = 25,validation_data = test_set,validation_steps = 20    )
# Part 3 - Making new predictions
'''test_imgs, test_label = next(testing_set)
test_label = test_label[:,0]
 
predictions = classifier.predict_generator(testing_set, steps = 24)
print(test_label  , predictions[:,0])
cm = confusion_matrix(test_label, predictions[:,0])

def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion Matrix', cmap = plt.cm.Blues):
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.average(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float')/cm.sum(axis = 1)[:, np.newaxis]
        print("Normalized Confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    print(cm)
    thresh = cm.max()/2.
    for i,j in itertools.product(range(cm,shape[0],range(cm,shape[1]))):
        plt.text(j,i,cm[i,j],
            horizontalalignment-"center",
            color-"white" if cm[i,j]>thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicated label')

cm_plot_labels = ['normal','pneumonia']

plot_confusion_matrix(cm,cm_plot_labels, title-'Confusion Matrix')'''