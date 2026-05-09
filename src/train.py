import yaml
from ultralytics import YOLO

# Load parameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Path to dataset yaml
DATA_YAML_PATH = "data/raw/Wildfire-Smoke-1/data.yaml"

# Load pretrained model
model = YOLO(params['model_type'])

# Train
model.train(
    data=DATA_YAML_PATH,
    imgsz=params['imgsz'],
    batch=params['batch'],
    epochs=params['epochs'],
    optimizer=params['optimizer'],
    lr0=params['lr0'],
    seed=params['seed'],
    pretrained=params['pretrained'],
    name=params['name']
)