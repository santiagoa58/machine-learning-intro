"""
Face Similarity Checker using SVM and LFW dataset.

This module trains an SVM model on the LFW dataset for face similarity checks. 
Features are extracted from raw pixel values
"""
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import numpy as np
from skimage import io
from skimage.transform import resize
from skimage.feature import hog
from collections import Counter
import argparse


def read_image(filename):
    """
    Read an image from a file, convert to grayscale, and vectorize.
    """
    return io.imread(filename, as_gray=True)


def preprocess_image(image):
    """
    Preprocess an image by resizing and extracting HOG features.
    """
    # resize image to 128x64 to standardize size
    image_resized = resize(image, (128, 64), anti_aliasing=True)
    features, _ = hog(
        image_resized,
        orientations=9,  # number of gradient orientations
        pixels_per_cell=(8, 8),  # size of cell for HOG
        cells_per_block=(2, 2),  # number of cells per block
        visualize=True,  # return HOG image
    )
    # Reshape HOG features to a 2D array where 1st dimension is number of samples and 2nd dimension is number of features
    return features.reshape(1, -1)


def preprocess_images(images):
    """
    Preprocess a set of images.
    """
    preprocessed = [preprocess_image(image.reshape(62, 47)) for image in images]
    return np.vstack(preprocessed)


def filter_underrepresented_classes(X, y, n):
    count_per_class = Counter(y)
    # Find classes that have fewer than 'n' samples
    underrepresented_classes = [
        cls for cls, count in count_per_class.items() if count < n
    ]
    # Remove these classes from X and y
    mask = np.isin(y, underrepresented_classes, invert=True)
    return X[mask], y[mask]


def load_data():
    """
    Load and preprocess LFW dataset.
    """
    lfw_people = fetch_lfw_people(
        min_faces_per_person=5, download_if_missing=True, data_home="sklearn-datasets"
    )
    X = preprocess_images(lfw_people.images)
    y = lfw_people.target
    X, y = filter_underrepresented_classes(X, y, 2)
    target_names = np.array([lfw_people.target_names[i] for i in np.unique(y)])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "target_names": target_names,
        "class_distribution": Counter(y),
    }


def create_model():
    """
    Create SVM model with dimensionality reduction to improve performance
    """
    svc = SVC(kernel="linear", class_weight="balanced")
    pca = PCA(n_components=250, whiten=True)
    model = make_pipeline(pca, svc)
    return model


def train_model(model, X, y):
    """
    Train SVM model on LFW dataset.
    """
    param_grid = {
        "svc__C": [
            0.0001,
            0.001,
            0.01,
            0.1,
            1,
            5,
            10,
            50,
            100,
        ],
    }

    grid = GridSearchCV(model, param_grid, cv=StratifiedKFold(n_splits=2))
    grid.fit(X, y)
    print(f"Best cross-validation accuracy: {grid.best_score_}")
    print(f"Best parameters: {grid.best_params_}")
    return grid.best_estimator_


def evaluate_model(model, X, y, target_names=None):
    """
    Evaluate model on LFW dataset.
    """
    score = model.score(X, y)
    y_pred = model.predict(X)
    return score, classification_report(
        y, y_pred, target_names=target_names, zero_division=1
    )


def predict_similarity(model, user_upload, n=3):
    """
    Predict 1-3 most similar faces to uploaded image.
    """
    result = model.predict(user_upload)
    distances = model.decision_function(user_upload)
    # Find top 3 similar faces based on distance metric
    most_similar = np.argsort(distances[0])[-n:]
    return most_similar


def main(user_upload):
    """
    Make predictions on uploaded image using SVM model and print results.
    """
    print("Loading model data...")
    processed_user_upload = preprocess_image(user_upload)
    data = load_data()
    print("Creating the model...")
    model = create_model()
    print("Training model...")
    model = train_model(model, data["X_train"], data["y_train"])
    print("Evaluating the model...")
    train_score, _ = evaluate_model(
        model, data["X_train"], data["y_train"], data["target_names"]
    )
    print(f"Training set model score: {train_score}")
    test_score, _ = evaluate_model(
        model, data["X_test"], data["y_test"], data["target_names"]
    )
    print(f"Testing set model score: {test_score}")
    print("Predicting similarity...")
    most_similar = predict_similarity(model, processed_user_upload)
    most_similar_names = data["target_names"][most_similar]
    print(f"Most similar faces: {most_similar_names}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Face Similarity Checker using SVM and LFW dataset."
    )
    parser.add_argument(
        "filename", type=str, help="Path to image file to check similarity against."
    )

    args = parser.parse_args()
    print(f"Parsing image: {args.filename}")
    user_upload = read_image(args.filename)
    main(user_upload)
