"""
Face Similarity Checker using SVM and LFW dataset.

This module trains an SVM model on the LFW dataset for face similarity checks. 
Features are extracted from raw pixel values. The model uses an RBF kernel.
"""
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import numpy as np
from skimage import io
from skimage.transform import resize
import argparse


def read_and_preprocess_image(filename):
    """
    Read an image from a file and preprocess it to be compatible with the LFW dataset.
    """
    image = io.imread(filename, as_gray=True)
    image_resized = resize(
        image, (62, 47), anti_aliasing=True
    )  # Resize to match LFW dimensions
    image_vectorized = image_resized.reshape(1, -1)  # Flatten the image
    return image_vectorized


def load_data():
    """
    Load and preprocess LFW dataset.
    """
    lfw_people = fetch_lfw_people(
        min_faces_per_person=5, download_if_missing=True, data_home="sklearn-datasets"
    )
    X = lfw_people.data
    y = lfw_people.target
    target_names = lfw_people.target_names
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "target_names": target_names,
    }


def create_model():
    """
    Create SVM model with dimensionality reduction to improve performance and
    RBF kernel for non-linear decision boundary.
    """
    svc = SVC(kernel="rbf", class_weight="balanced", C=30, gamma=0.001)
    pca = PCA(n_components=150, whiten=True)
    model = make_pipeline(pca, svc)
    return model


def train_model(model, X, y):
    """
    Train SVM model on LFW dataset.
    """
    param_grid = {
        "svc__C": [1, 10, 20, 30, 50],
        "svc__gamma": [0.001, 0.01, 0.1],
    }
    grid = GridSearchCV(model, param_grid, cv=2)
    grid.fit(X, y)
    print(f"Best cross-validation accuracy: {grid.best_score_}")
    print(f"Best parameters: {grid.best_params_}")
    return grid.best_estimator_


def evaluate_model(model, X, y):
    """
    Evaluate model on LFW dataset.
    """
    score = model.score(X, y)
    return score


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
    data = load_data()
    print("Creating the model...")
    model = create_model()
    print("Training model...")
    model = train_model(model, data["X_train"], data["y_train"])
    print("Evaluating the model...")
    train_score = evaluate_model(model, data["X_train"], data["y_train"])
    print(f"Training set model score: {train_score}")
    test_score = evaluate_model(model, data["X_test"], data["y_test"])
    print(f"Testing set model score: {test_score}")
    print("Predicting similarity...")
    most_similar = predict_similarity(model, user_upload)
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
    user_upload = read_and_preprocess_image(args.filename)
    main(user_upload)
