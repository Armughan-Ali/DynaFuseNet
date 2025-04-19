CONFIG = {
    "image_size": (224, 224),
    "batch_size": 16,
    "num_epochs": 30,
    "learning_rate": 0.0001,
    "skeleton_points": 21,
    "fusion_output_dim": 256,
    "num_classes": 20,
    "data_path": "./data/",
    "save_model_path": "./outputs/model.pt",
    "device": "cuda"  # change to "cpu" if not using GPU
}