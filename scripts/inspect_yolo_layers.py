from ultralytics import YOLO

# Load the YOLO model we want to inspect
# You can change this to another model path if needed
model = YOLO("yolo11s.pt")

# The actual PyTorch model is inside model.model
print("\nYOLO model layer structure:\n")

for i, layer in enumerate(model.model.model):
    # Print layer number, layer type, and whether parameters are trainable
    num_params = sum(p.numel() for p in layer.parameters())
    trainable_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)

    print(
        f"Layer {i:2d} | "
        f"{layer.__class__.__name__:20s} | "
        f"params: {num_params:,} | "
        f"trainable: {trainable_params:,}"
    )
