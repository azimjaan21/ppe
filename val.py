from ultralytics import YOLO

def validate_model():
    model = YOLO(r'C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\ppe\ppe.pt')

    # Run validation with additional parameters
    results = model.val(
        data=r'C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\ppe\ppe_data.yaml',
        batch=16,
        imgsz=640,
        device='cuda',
        verbose=True,
        save=True,  # Save validation predictions with bounding boxes
        conf=0.25  # Set confidence threshold for validation
    )

    # Print detailed validation summary
    print("\nValidation Summary:")
    print(f"mAP@0.5: {results['map_50']:.2f}")
    print(f"mAP@0.5:0.95: {results['map_50_95']:.2f}")
    print(f"Precision: {results['precision']:.2f}")
    print(f"Recall: {results['recall']:.2f}")

    # Save results to a log file
    with open("validation_results.log", "a") as log_file:
        log_file.write(f"\nValidation Results:\n{results}\n")

    # If available, plot PR curve
    if hasattr(results, "plot_pr_curve"):
        results.plot_pr_curve()

if __name__ == '__main__':
    validate_model()
