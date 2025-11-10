import PreProcessing
import FeaturesExtraction


if __name__ == "__main__":
    IMAGE_PATH = ""

    processed_image_array = PreProcessing.load_and_preprocess_image(IMAGE_PATH)
    features = FeaturesExtraction.extract_features(processed_image_array)