import numpy as np
import torch
import cv2


def add_elapsed_steps_overlay(obs, display_value):
    """Add visual overlay showing a value on the observation images

    Args:
        obs: observation dictionary
        display_value: the value to display on the overlay
    """
    if "sensor_data" in obs and "base_camera" in obs["sensor_data"]:
        images = obs["sensor_data"]["base_camera"]["rgb"]

        # Handle both single image and batched images
        if isinstance(images, torch.Tensor):
            images_np = images.cpu().numpy()
        else:
            images_np = np.array(images)

        # Get display value
        value = int(display_value)

        # Process each image in the batch
        original_shape = images_np.shape
        if len(original_shape) == 3:  # Single image (H, W, C)
            images_np = images_np[np.newaxis, ...]

        processed_images = []
        for img in images_np:
            # Convert from float [0, 1] to uint8 [0, 255] if needed
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = (img * 255).astype(np.uint8)

            # Add text overlay using cv2
            img_with_text = img.copy()
            text = f"Steps: {value}"
            color = (255, 255, 255)  # White

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            bg_color = (0, 0, 0)  # Black background

            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # Position at bottom-left corner with padding
            img_height = img_with_text.shape[0]
            x = 10
            y = img_height - 15

            # Draw black background rectangle
            cv2.rectangle(img_with_text, (x - 5, y - text_height - 5),
                        (x + text_width + 5, y + baseline + 5), bg_color, -1)

            # Draw text
            cv2.putText(img_with_text, text, (x, y), font, font_scale, color, thickness)

            # Convert back to float [0, 1] if original was float
            if original_shape[-1] == 3 and (img.dtype == np.float32 or img.dtype == np.float64):
                img_with_text = img_with_text.astype(np.float32) / 255.0

            processed_images.append(img_with_text)

        # Convert back to tensor and restore original shape
        processed_images = np.array(processed_images)
        if len(original_shape) == 3:  # Was single image
            processed_images = processed_images[0]

        obs["sensor_data"]["base_camera"]["rgb"] = torch.from_numpy(processed_images).to(images.device)

    return obs
