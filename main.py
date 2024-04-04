import os
import numpy as np
from yoga_pose_classification import Model
from keras.models import load_model
from keras.preprocessing import image

# Train and save CNN model to classify yoga poses
def train_pose_classifier_cnn(model_filename):
  cnn_model = Model()
  cnn_model.build_model()
  cnn_model.summary()
  cnn_model.compile()
  train_generator, validation_generator = cnn_model.generate_img_data()
  history = cnn_model.train(train_generator, validation_generator)

  cnn_model.save(model_filename)
  

# Predict yoga pose given 
def predict_pose(image_path, model_path):
  loaded_model = load_model('best_model.h5')

  img_path = 'goddess_test_2_stick.jpg'
  img = image.load_img(img_path, target_size=(300, 300))
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array /= 255.0


  predictions = loaded_model.predict(img_array)
  np.set_printoptions(suppress=True, precision=4)
  print(predictions)
  label = np.argmax(predictions)

  entries = os.listdir('./data/test')
  labels = {0: 'downdog', 1: 'goddess', 2: 'mountain', 3: 'plank', 4: 'tree', 5: 'warrior2'}
  # labels = {i: label for i, label in enumerate(entries) if label != '.DS_Store'}
  # labels = {i - 1 if i > 1 else i: label for i, label in labels.items()}
  print(labels)
  print(labels[label])


if __name__ == "__main__":
  model_filename = 'yoga_pose_classification_model.h5'
  image_filename = 'goddess_test_2_stik.jpg'
  
  # train_pose_classifier_cnn(model_filename)
  # predict_pose(image_filename, model_filename) 
  