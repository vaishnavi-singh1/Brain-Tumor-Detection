import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
def adaptive_median_filtering(image):
    return cv2.medianBlur(image, 3)  
def gmm_segmentation(image):
    # Reshape image into a 2D array where each row is a pixel and each column is a channel
    img_reshaped = image.reshape((-1, 1))
def gmm_segmentation(image):
    # Reshape image into a 2D array where each row is a pixel and each column is a channel
    img_reshaped = image.reshape((-1, 1))
    gmm = GaussianMixture(n_components=2)
    gmm.fit(img_reshaped)
    segmented_img = segmented_img.reshape(image.shape)
    return segmented_img
 
def extract_features(image):
     glcm = greycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast').flatten()
    correlation = greycoprops(glcm, 'correlation').flatten()
    energy = greycoprops(glcm, 'energy').flatten()
    homogeneity = greycoprops(glcm, 'homogeneity').flatten()
    return np.hstack([contrast, correlation, energy, homogeneity])
def train_classifier(features, labels):
    classifier = SVC(kernel='linear', probability=True)
    cv_scores = cross_val_score(classifier, features, labels, cv=5)
    classifier.fit(features, labels)
    return classifier, cv_scores.mean()
  
if __name__ == "__main__":
       image = cv2.imread('brain_tumor_image.jpg', cv2.IMREAD_GRAYSCALE)  # Replace with your image path
 
       filtered_image = adaptive_median_filtering(image)
       segmented_img = gmm_segmentation(filtered_image)
       features = extract_features(segmented_img)
   
       labels = np.array([1, 0])  # Dummy labels: 1 for malignant, 0 for benign
       features = np.array([features, features])  # Dummy features for example purposes
   
       classifier, validation_accuracy = train_classifier(features, labels)
       print(f"Validation Accuracy: {validation_accuracy}")
   
       cv2.imshow('Original Image', image)
       cv2.imshow('Filtered Image', filtered_image)
       cv2.imshow('Segmented Image', segmented_img * 255)
       cv2.waitKey(0)
       cv2.destroyAllWindows()
