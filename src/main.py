from calories_db import CalorieDatabaseAndInteractor
from calorie_estimator import CalorieEstimator

def main():
    DATA_FILE_PATH = '../utils/calories_database.json'
    MODEL_PATH = '../model/best.pt'
    OUTPUT_IMG_DIR = '../outputs/'

    db = CalorieDatabaseAndInteractor(DATA_FILE_PATH)

    estimator = CalorieEstimator(MODEL_PATH,OUTPUT_IMG_DIR ,db)

    image_path = '../data/test/images/IMG_20230909_201044_jpg.rf.2b538e316b4324e6409a9d0d27dff2d9.jpg'
    estimator.pipeline(img_path=image_path,visualize=True)
    

main()