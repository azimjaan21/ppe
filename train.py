from ultralytics import YOLO

def train_model():

    model = YOLO('yolov8m.pt')

    # Training the modell
    model.train(
        data=r'C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\ppe\ppe_data.yaml',
        epochs=100,
        batch=16,
        imgsz=640,
        device='cuda',
        lr0=0.01,
        weight_decay=0.0005,
        patience=10,  # Early stopping patience
        optimizer='SGD',
        momentum=0.937,
        augment=True,
        plots=True,
        verbose=True
    )

if __name__ == '__main__':
    train_model()
