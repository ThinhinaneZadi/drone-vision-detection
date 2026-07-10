# ------------------------------------------------------------
# Custom Partial Fine-Tuning Script for YOLO11s on VisDrone
# ------------------------------------------------------------
# Purpose:
# This script loads a pretrained YOLO11s model, freezes most of
# the model, trains only the last deep layer + Detect head, and
# validates the result on VisDrone.
#
# Main experiment:
# - Pretrained model: YOLO11s
# - Frozen layers: 0 to 21
# - Trainable layers: 22 and 23
# - Batch size: 8
# - Image size: 640
# - Optimizer: AdamW
# - Learning rate: 0.005
#
# This is more advanced than only using the YOLO command line
# because we inspect the model, count trainable parameters, and
# explain the training algorithm directly in code.
# ------------------------------------------------------------


# Import YOLO from Ultralytics.
# YOLO gives us access to model loading, training, validation,
# detection, and tracking tools.
from ultralytics import YOLO


def count_parameters(model):
    """
    Count total parameters and trainable parameters.

    A parameter is a learnable number inside the neural network.
    Examples:
    - convolution weights
    - bias values
    - detection head weights

    requires_grad=True:
        PyTorch computes gradients for this parameter.
        The optimizer can update it.

    requires_grad=False:
        The parameter is frozen.
        It will not be updated during training.
    """

    # Store total number of parameters in the full model.
    total_params = 0

    # Store only the parameters that are trainable.
    trainable_params = 0

    # Loop through every parameter tensor in the model.
    for param in model.parameters():

        # numel() returns how many numbers are inside the tensor.
        # Example: shape [64, 3, 3, 3] has 64*3*3*3 parameters.
        total_params += param.numel()

        # Count this parameter only if it is trainable.
        if param.requires_grad:
            trainable_params += param.numel()

    # Return both values.
    return total_params, trainable_params


def freeze_early_layers(yolo_model, freeze_until_layer):
    """
    Freeze early YOLO layers and keep later layers trainable.

    freeze_until_layer = 22 means:
    - Layers 0 through 21 are frozen.
    - Layers 22 and 23 are trainable.

    Why this experiment is useful:
    - Early layers learn general features like edges, colors, and textures.
    - Later layers learn more task-specific features.
    - Layer 22 is a deep feature layer with many parameters.
    - Layer 23 is the Detect head.
    - Training only layers 22 and 23 gives more adaptation than Detect-only
      training, but still keeps most of the model frozen.
    """

    # ------------------------------------------------------------
    # Step 1: Freeze every parameter first.
    # ------------------------------------------------------------
    # This makes the whole model frozen at the beginning.
    for param in yolo_model.model.parameters():
        param.requires_grad = False

    # ------------------------------------------------------------
    # Step 2: Unfreeze only the last selected layers.
    # ------------------------------------------------------------
    # yolo_model.model.model is the internal list of YOLO layers.
    # Each layer has an index: 0, 1, 2, ..., 23.
    for layer_index, layer in enumerate(yolo_model.model.model):

        # If the layer index is 22 or higher,
        # we want this layer to be trainable.
        if layer_index >= freeze_until_layer:

            # Loop through all parameters inside this layer.
            for param in layer.parameters():

                # Allow PyTorch to compute gradients for this parameter.
                param.requires_grad = True

    # ------------------------------------------------------------
    # Step 3: Print layer-by-layer freezing summary.
    # ------------------------------------------------------------
    print("\nLayer freezing summary before training:")
    print("------------------------------------------------------------")

    # Inspect each YOLO layer.
    for layer_index, layer in enumerate(yolo_model.model.model):

        # Count all parameters in this layer.
        layer_params = sum(p.numel() for p in layer.parameters())

        # Count only trainable parameters in this layer.
        trainable_params = sum(
            p.numel() for p in layer.parameters() if p.requires_grad
        )

        # Decide whether the layer is frozen or trainable.
        if trainable_params > 0:
            status = "TRAINABLE"
        else:
            status = "FROZEN"

        # Print layer information.
        print(
            f"Layer {layer_index:02d} | "
            f"{layer.__class__.__name__:20s} | "
            f"{status:10s} | "
            f"total params={layer_params:,} | "
            f"trainable={trainable_params:,}"
        )


def main():
    """
    Main experiment algorithm.

    Algorithm:
    1. Define hyperparameters.
    2. Load pretrained YOLO11s.
    3. Freeze layers 0-21.
    4. Keep layers 22-23 trainable.
    5. Count trainable parameters.
    6. Train on VisDrone.
    7. Validate on VisDrone validation set.
    """

    # ------------------------------------------------------------
    # 1. Experiment setup / hyperparameters
    # ------------------------------------------------------------

    # Pretrained YOLO11s weights.
    # This starts from COCO-pretrained knowledge instead of random weights.
    pretrained_model = "yolo11s.pt"

    # Dataset configuration file.
    # This file tells YOLO where the train/val images are
    # and what the 10 VisDrone classes are.
    data_yaml = "visdrone.yaml"

    # Freeze index.
    #
    # freeze_until_layer = 22 means:
    # - Freeze layers 0 through 21.
    # - Train layers 22 and 23.
    #
    # This trains only:
    # - Layer 22: deep C3k2 feature layer
    # - Layer 23: Detect head
    freeze_until_layer = 22

    # Number of epochs.
    # One epoch means the model sees the whole training dataset once.
    epochs = 20

    # Input image size.
    # YOLO will resize images to 640 during training and validation.
    image_size = 640

    # Batch size.
    # This means YOLO processes 8 images before one optimizer update.
    #
    # Important:
    # batch_size = 8 does NOT mean only 8 batches total.
    # With 6,471 training images:
    # 6471 / 8 ≈ 809 mini-batches per epoch.
    batch_size = 8

    # Initial learning rate.
    # This controls how large the parameter updates are.
    learning_rate = 0.005

    # Final learning rate fraction.
    # Ultralytics uses this for its LR schedule.
    # Final LR is approximately lr0 * lrf.
    # Example: 0.005 * 0.01 = 0.00005 near the end.
    final_lr_fraction = 0.01

    # Optimizer.
    # AdamW uses adaptive updates and decoupled weight decay.
    # We set it explicitly so Ultralytics does not use optimizer=auto.
    optimizer_name = "AdamW"

    # Loss weights.
    # These control how much each part of the YOLO loss matters.
    #
    # box:
    #   Weight for bounding box localization loss.
    #
    # cls:
    #   Weight for classification loss.
    #
    # dfl:
    #   Weight for Distribution Focal Loss.
    #
    # These are kept at common/default-style values for this run.
    box_loss_weight = 7.5
    cls_loss_weight = 0.5
    dfl_loss_weight = 1.5

    # Experiment name.
    # This should match the actual setup.
    experiment_name = "custom_last2_layers_img640_batch8_adamw_lr005"

    # ------------------------------------------------------------
    # 2. Print experiment setup
    # ------------------------------------------------------------

    print("\nCustom YOLO11s Last-Layers Fine-Tuning Experiment")
    print("============================================================")
    print(f"Pretrained model: {pretrained_model}")
    print(f"Dataset YAML: {data_yaml}")
    print(f"Frozen layers: 0 to {freeze_until_layer - 1}")
    print(f"Trainable layers: {freeze_until_layer} to 23")
    print(f"Epochs: {epochs}")
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate lr0: {learning_rate}")
    print(f"Final LR fraction lrf: {final_lr_fraction}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Box loss weight: {box_loss_weight}")
    print(f"Class loss weight: {cls_loss_weight}")
    print(f"DFL loss weight: {dfl_loss_weight}")
    print(f"Experiment name: {experiment_name}")
    print("============================================================")

    # ------------------------------------------------------------
    # 3. Load pretrained YOLO11s
    # ------------------------------------------------------------

    # Create a YOLO object from pretrained YOLO11s weights.
    model = YOLO(pretrained_model)

    # ------------------------------------------------------------
    # 4. Manually freeze/unfreeze layers for inspection
    # ------------------------------------------------------------

    # This manually sets requires_grad before training so we can inspect
    # exactly which layers are intended to be frozen/trainable.
    freeze_early_layers(model, freeze_until_layer)

    # Count parameters after our manual freezing.
    total_params, trainable_params = count_parameters(model.model)

    # Calculate frozen parameter count.
    frozen_params = total_params - trainable_params

    # Calculate percentage of model that will be trainable.
    trainable_percentage = 100 * trainable_params / total_params

    print("\nParameter summary before training:")
    print("------------------------------------------------------------")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Trainable percentage: {trainable_percentage:.2f}%")
    print("------------------------------------------------------------")

    # ------------------------------------------------------------
    # 5. Train the model
    # ------------------------------------------------------------
    #
    # Training algorithm:
    #
    # For each epoch:
    #     For each mini-batch (x, y):
    #
    #         1. Forward pass:
    #              y_hat = model(x)
    #
    #         2. Compute YOLO detection loss:
    #              L_total = λ_box * L_box
    #                      + λ_cls * L_cls
    #                      + λ_dfl * L_dfl
    #
    #         3. Backpropagation:
    #              Compute gradients of L_total.
    #
    #         4. Optimizer step:
    #              Update only trainable parameters.
    #
    # In this experiment:
    #     Layers 0-21 are frozen.
    #     Layers 22-23 are trainable.
    #
    # Very important:
    #     We also pass freeze=freeze_until_layer into model.train().
    #     This forces Ultralytics trainer to apply the same freeze rule.
    # ------------------------------------------------------------

    print("\nStarting training...")
    print("------------------------------------------------------------")

    model.train(
        # Dataset YAML file.
        data=data_yaml,

        # Number of epochs.
        epochs=epochs,

        # Image size.
        imgsz=image_size,

        # Mini-batch size.
        batch=batch_size,

        # Use GPU 0.
        device=0,

        # Number of dataloader workers.
        workers=2,

        # Initial learning rate.
        lr0=learning_rate,

        # Final learning rate fraction for LR schedule.
        lrf=final_lr_fraction,

        # Explicit optimizer.
        # This prevents Ultralytics from using optimizer=auto
        # and ignoring our chosen learning rate.
        optimizer=optimizer_name,

        # Force Ultralytics trainer to freeze layers 0 to 21.
        # With freeze=22:
        # - layers 0-21 are frozen
        # - layers 22-23 are trainable
        freeze=freeze_until_layer,

        # Loss weights.
        box=box_loss_weight,
        cls=cls_loss_weight,
        dfl=dfl_loss_weight,

        # Disable AMP because GTX 1650 previously showed AMP warnings.
        # This makes training safer, although sometimes a little slower.
        amp=False,

        # If the folder already exists, continue without crashing.
        exist_ok=True,

        # Save run under runs/detect.
        project="runs/detect",

        # Name of this specific experiment.
        name=experiment_name,
    )

    # ------------------------------------------------------------
    # 6. Validate the trained model
    # ------------------------------------------------------------
    #
    # Validation reports:
    # - Precision
    # - Recall
    # - mAP50
    # - mAP50-95
    #
    # These metrics tell us how well the fine-tuned model performs
    # on the VisDrone validation set.
    # ------------------------------------------------------------

    print("\nStarting final validation...")
    print("------------------------------------------------------------")

    metrics = model.val(
        # Same dataset YAML.
        data=data_yaml,

        # Same image size as training.
        imgsz=image_size,

        # Same batch size.
        batch=batch_size,

        # Use GPU 0.
        device=0,
    )

    # Print validation result object.
    print("\nValidation finished.")
    print("------------------------------------------------------------")
    print(metrics)


# ------------------------------------------------------------
# Python entry point
# ------------------------------------------------------------
# This means:
# Run main() only when this script is executed directly.
#
# Example:
# python scripts/custom_partial_finetune_yolo11s.py
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
