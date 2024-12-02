import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk

class FruitCalorieApp:
    def __init__(self, calorie_estimator_function):
        """
        Initialize the Tkinter application
        
        :param calorie_estimator_function: Your calorie estimation function
        """
        # Root window
        self.root = tk.Tk()
        self.root.title("Fruit Calorie Estimator")
        self.root.geometry("1200x800")

        # Select Image Button
        self.select_button = tk.Button(self.root, 
                                       text="Select Image", 
                                       command=self.open_file_dialog, 
                                       font=("Arial", 14))
        self.select_button.pack(pady=10)

        # Main container
        self.main_container = tk.Frame(self.root)
        self.main_container.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        # Image frame
        self.image_frame = tk.Frame(self.main_container)
        self.image_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10)

        # Image label with a title
        self.image_title = tk.Label(self.image_frame, text="Processed Image", font=("Arial", 12, "bold"))
        self.image_title.pack()
        
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack(expand=True, fill=tk.BOTH)

        # Results frame
        self.results_frame = tk.Frame(self.main_container)
        self.results_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Results table
        self.results_tree = ttk.Treeview(self.results_frame, 
                                         columns=('Class', 'Confidence', 'Weight (g)', 'Calories'), 
                                         show='headings')
        self.results_tree.heading('Class', text='Fruit/Vegetable')
        self.results_tree.heading('Confidence', text='Confidence')
        self.results_tree.heading('Weight (g)', text='Weight (g)')
        self.results_tree.heading('Calories', text='Calories')
        self.results_tree.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        # Total calories label
        self.total_calories_label = tk.Label(self.results_frame, 
                                             text="Total Calories: 0", 
                                             font=("Arial", 12, "bold"))
        self.total_calories_label.pack(pady=10)

        # Store the calorie estimation function
        self.calorie_estimator = calorie_estimator_function

    def open_file_dialog(self):
        """Open file dialog to select an image"""
        filepath = filedialog.askopenfilename(
            title="Select an Image", 
            filetypes=[
                ('Image files', '*.png *.jpg *.jpeg *.bmp *.gif'),
                ('All files', '*.*')
            ]
        )
        if filepath:
            self.process_image(filepath)

    def process_image(self, filepath):
        """
        Process the selected image
        
        :param filepath: Path to the selected image file
        """
        try:
            # Call your calorie estimation function
            results, image = self.calorie_estimator(filepath)
            print(results)
            # Display processed image
            self.display_image(image)

            # Clear previous results
            for i in self.results_tree.get_children():
                self.results_tree.delete(i)

            # Populate results table
            for detection in results['detections']:
                self.results_tree.insert('', 'end', values=(
                    detection['class'],
                    f"{detection['confidence']:.2f}",
                    f"{detection['estimated_weight_g']:.2f}",
                    f"{detection['calories']:.2f}"
                ))

            # Update total calories
            self.total_calories_label.config(
                text=f"Total Calories: {results['total_calories']:.2f}"
            )

        except Exception as e:
            messagebox.showerror("Error", str(e))
            # Clear previous image and results if processing fails
            self.image_label.config(image='')
            for i in self.results_tree.get_children():
                self.results_tree.delete(i)
            self.total_calories_label.config(text="Total Calories: 0")

    def display_image(self, numpy_image):
        """
        Display the image in the GUI
        
        :param numpy_image: Numpy array of the image (BGR format from OpenCV)
        """
        try:
            # Ensure the image is not empty
            if numpy_image is None or numpy_image.size == 0:
                raise ValueError("Empty or invalid image")

            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_img = Image.fromarray(rgb_image)
            
            # Resize image to fit GUI while maintaining aspect ratio
            pil_img.thumbnail((600, 400))
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_img)
            
            # Update image label
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference
        
        except Exception as e:
            messagebox.showerror("Image Display Error", str(e))
            self.image_label.config(image='')

    def run(self):
        """Start the Tkinter event loop"""
        self.root.mainloop()

# Main execution
def main():
    # Import your existing calorie estimation function
    from calories_db import CalorieDatabaseAndInteractor
    from calorie_estimator import CalorieEstimator

    DATA_FILE_PATH = '../utils/calories_database.json'
    MODEL_PATH = '../model/best.pt'
    OUTPUT_IMG_DIR = '../outputs/'

    db = CalorieDatabaseAndInteractor(DATA_FILE_PATH)

    estimator = CalorieEstimator(MODEL_PATH,OUTPUT_IMG_DIR ,db)

    app = FruitCalorieApp(estimator.pipeline)
    app.run()

if __name__ == "__main__":
    main()