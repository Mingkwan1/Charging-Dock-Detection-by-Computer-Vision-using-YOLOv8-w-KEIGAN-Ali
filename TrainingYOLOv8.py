from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.pt") #Desired pre trained model. The size increases from n < s < m < l

# Train the model   
train_results = model.train(
    task = "detect",
    data=" ",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    freeze = 10,
    lr0 = 0.005,
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model.predict("Enter path to video or image",show =True, save=True)
results[0].show()