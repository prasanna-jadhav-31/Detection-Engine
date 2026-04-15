from torchvision import transforms

def get_transform(is_train: bool = True):
    """
    Returns the torchvision transform pipeline configured for the model.

    Args:
        is_train (bool): Whether the transform is for training or inference.
            
    Returns:
        torchvision.transforms.Compose: The transformation pipeline.
    """
    # Define the base transform pipeline (common for both)
    pipeline = [
        # Resize all images to 224x224 as required by standard CNNs / ViTs
        transforms.Resize((224, 224)),
    ]

    # Add training-specific augmentations
    if is_train:
        pipeline.extend([
            # Stronger augmentations as requested
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])

    pipeline.extend([
        # Convert PIL Image or numpy.ndarray to tensor.
        # This scales the pixel values from [0, 255] to [0.0, 1.0].
        transforms.ToTensor(),
        # Normalize the image with ImageNet mean and standard deviation
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])

    if is_train:
        pipeline.append(transforms.RandomErasing(p=0.2))

    return transforms.Compose(pipeline)
