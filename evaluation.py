import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import confusion_matrix

# Import the resize_to_fit function from the helpers module
from helpers import resize_to_fit

# Load the trained model
MODEL_FILENAME = "captcha_model.hdf5"
model = load_model(MODEL_FILENAME)

LETTER_IMAGES_FOLDER = "extracted_letter_images"
MODEL_LABELS_FILENAME = "model_labels.dat"

# Load the label binarizer
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Initialize data and labels for the test set
test_data = []
true_labels = []

# Loop over the input test images
for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = resize_to_fit(image, 20, 20)
    image = np.expand_dims(image, axis=2)
    label = image_file.split(os.path.sep)[-2]
    test_data.append(image)
    true_labels.append(label)

# Scale the raw pixel intensities to the range [0, 1]
test_data = np.array(test_data, dtype="float") / 255.0

# Convert the true labels into one-hot encodings
true_labels = lb.transform(true_labels)

# Make predictions on the test data
predicted_labels = model.predict(test_data)

# Convert one-hot encoded predictions to class labels
predicted_labels = lb.inverse_transform(predicted_labels)


# Convert the predicted labels to one-hot encodings
predicted_labels_onehot = lb.transform(predicted_labels)

# Calculate and display evaluation metrics
accuracy = accuracy_score(true_labels, predicted_labels_onehot)  #accuracy = tn+tp/(total instances)
precision = precision_score(true_labels, predicted_labels_onehot, average='weighted')    #precision = tp/(tp+fp)
recall = recall_score(true_labels, predicted_labels_onehot, average='weighted')         #recall = tp/(tp+fn)
f1 = f1_score(true_labels, predicted_labels_onehot, average='weighted')                #f1score = 2*(prec*recall)/(prec+recall)

# Convert the true labels and predicted labels to their respective classes
true_labels_class = lb.inverse_transform(true_labels)
predicted_labels_class = lb.inverse_transform(predicted_labels_onehot)

# Calculate the confusion matrix
confusion = confusion_matrix(true_labels_class, predicted_labels_class)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Confusion Matrix:")
print(confusion)
