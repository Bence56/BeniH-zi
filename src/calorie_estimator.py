import cv2
import numpy as np
from ultralytics import YOLO
import json
from calories_db import CalorieDatabaseAndInteractor


class CalorieEstimator:
    def __init__(self, model_path:str,img_output_dir_path:str,calorie_db_interactor:CalorieDatabaseAndInteractor):
        """
        Initialize the calorie estimator with YOLO model and calories database
        
        :param model_path: Path to the trained YOLOv8 model
        :param calories_database_path: Path to JSON file with calorie information
        """
    
        self.model = YOLO(model_path)
        self.img_output_dir_path = img_output_dir_path
        self.calories_db = calorie_db_interactor
    
    def _estimate_calories(self, image_path):
        """
        Estimate calories for fruits and vegetables in an image
        
        :param image_path: Path to the input image
        :return: Dictionary with detection results and calorie information
        """
        image = cv2.imread(image_path)
        
        results = self.model(image)
        
        calorie_results = {
            'total_calories': 0,
            'detections': []
        }
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[cls]
                
                # Estimate calories (simplified approach)
                # In a real-world scenario, you'd estimate portion size and use more precise calculations
                calories = self._get_calories(class_name)
                
                # Crop the detected object
                cropped_object = image[y1:y2, x1:x2]
                
                # Estimate approximate weight/volume (simplified)
                estimated_weight = self._estimate_weight(cropped_object)
                
                # Calculate calories based on estimated weight
                adjusted_calories = calories * (estimated_weight / 100)  # Assuming calories are per 100g
                
                # Populate detection results
                detection = {
                    'class': class_name,
                    'confidence': conf,
                    'bounding_box': {
                        'x1': x1, 'y1': y1,
                        'x2': x2, 'y2': y2
                    },
                    'estimated_weight_g': round(estimated_weight, 2),
                    'calories': round(adjusted_calories, 2)
                }
                
                calorie_results['detections'].append(detection)
                calorie_results['total_calories'] += detection['calories']
        
        return calorie_results
    
    def _get_calories(self, class_name):
        """
        Retrieve calorie information for a given class
        
        :param class_name: Name of the fruit/vegetable
        :return: Calories per 100g
        """
        return self.calories_db.get_calorie_for_class_name(class_name.lower())
    
    def _estimate_weight(self, cropped_object):
        """
        Estimate weight of the detected object
        
        :param cropped_object: Cropped image of the detected object
        :return: Estimated weight in grams
        """
        volume = cropped_object.shape[0] * cropped_object.shape[1]
        estimated_weight = volume / 500  # Calibration factor
        
        return max(10, min(estimated_weight, 500))
    
    def _visualize_results(self, image_path, results):
        """
        Visualize detection results on the image
        
        :param image_path: Path to the original image
        :param results: Calorie estimation results
        """
        image = cv2.imread(image_path)
        
        for detection in results['detections']:
            x1, y1 = detection['bounding_box']['x1'], detection['bounding_box']['y1']
            x2, y2 = detection['bounding_box']['x2'], detection['bounding_box']['y2']
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"{detection['class']}: {detection['calories']} cal"
            cv2.putText(image, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        total_label = f"Total Calories: {results['total_calories']:.2f}"
        cv2.putText(image, total_label, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imwrite(f"{self.img_output_dir_path}result_with_calories.jpg", image)
        return image

    def pipeline(self,img_path:str,visualize:bool=True):    
    
        results = self._estimate_calories(img_path)
        
        image = self._visualize_results(img_path, results)
        
        return results, image

