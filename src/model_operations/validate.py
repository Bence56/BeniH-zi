from ultralytics import YOLO

model = YOLO('./best.pt')

results = model.val(data='/Users/bencesomogyi/Documents/BeniHázi/data/data.yaml')

print(results)

results.plot_confusion_matrix()