import json

class CalorieDatabaseAndInteractor:
    def __init__(self, json_data_filepath: str):
        try:
            with open(json_data_filepath, 'r') as json_file:
                self.calories = json.load(json_file)
        except FileNotFoundError:
            print(f"Error: File '{json_data_filepath}' not found.")
            self.calories = {}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            self.calories = {}

    def print_db(self)->None:
        print(self.calories)

    def get_calorie_for_class_name(self,class_name:str)->str:
        return self.calories[class_name]