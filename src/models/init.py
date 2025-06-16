from .base_model import CustomerSegmentationModel
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


def main():
    segmentation_model = CustomerSegmentationModel(
        csv_file_path='data/external/customer/Train.csv')

    X, y, feature_names = segmentation_model.load_and_preprocess_data()

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {len(y)}")
    print(f"Number of unique segments: {len(np.unique(y))}")
    print(f"Segment distribution: {np.bincount(y)}")

    # Train the model
    print("\nTraining the model...")
    X_test, y_test, y_pred = segmentation_model.train_model(X, y, epochs=100)

    # Evaluate model performance
    print("\nModel Performance:")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    print("\nFeature Importance:")
    importance_df = segmentation_model.get_feature_importance(feature_names)
    print(importance_df)

    # Example prediction on new data
    print("\nExample predictions on test data:")
    probabilities = segmentation_model.predict(X_test[:5])[1]
    for i, prob in enumerate(probabilities):
        print(f"Sample {i+1}: Segment {y_pred[i]}, Probabilities: {prob}")

    # Show segment mapping if available
    if 'target' in segmentation_model.label_encoders:
        print(
            f"\nSegment mapping: {dict(zip(segmentation_model.label_encoders['target'].transform(segmentation_model.label_encoders['target'].classes_), segmentation_model.label_encoders['target'].classes_))}")


if __name__ == "__main__":
    main()
