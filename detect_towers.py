import os
import glob
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def detect_towers_in_images(input_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check for device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load processor and model
    model_id = "IDEA-Research/grounding-dino-tiny"
    print(f"Loading processor and model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # Prompt for grounding dino
    # Needs to be lowercased and end with a dot
    text_prompt = "an electricity transmission tower. a power transmission pylon."

    # Find all JPEGs (handling case insensitivity)
    image_paths = glob.glob(os.path.join(input_dir, "*.[jJ][pP][gG]"))
    if not image_paths:
        print(f"No JPG images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images to process.")

    for img_path in sorted(image_paths):
        print(f"\nProcessing {os.path.basename(img_path)}...")
        try:
            image = Image.open(img_path).convert("RGB")
            orig_width, orig_height = image.size
            print(f"Original size: {orig_width}x{orig_height}")

            # Prepare inputs
            inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

            # Inference
            with torch.no_grad():
                outputs = model(**inputs)

            # Post-process detection results
            # The signature has been standardized: 'threshold' instead of 'box_threshold'
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=0.35,
                text_threshold=0.3,
                target_sizes=[(orig_height, orig_width)]
            )[0]

            # Extract results
            boxes = results["boxes"].cpu().numpy()
            scores = results["scores"].cpu().numpy()
            labels = results["labels"]

            print(f"Detected {len(boxes)} tower(s) in {os.path.basename(img_path)}")

            # Annotate image
            draw = ImageDraw.Draw(image)

            # Since the original image is very high resolution, use a thick line
            line_width = max(5, int(min(orig_width, orig_height) * 0.005))

            # Attempt to get a default/large font if possible
            try:
                # Use a larger font for high-res images
                font = ImageFont.load_default()
            except Exception:
                font = None

            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                xmin, ymin, xmax, ymax = box
                # Draw bounding box
                draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=line_width)

                # Draw text background and text
                label_text = f"{label} {score:.2f}"
                print(f"  Box {i+1}: {label_text} -> [{int(xmin)}, {int(ymin)}, {int(xmax)}, {int(ymax)}]")

                # Draw label text slightly above the box
                draw.text((xmin + 5, max(0, ymin - 35)), label_text, fill="red", font=font)

            # Save annotated full-res image
            out_filename = os.path.basename(img_path)
            out_path = os.path.join(output_dir, out_filename)
            image.save(out_path, quality=90)
            print(f"Saved full-res annotated image to {out_path}")

            # Save a resized/preview version (width 1280) for easy visualization
            preview_width = 1280
            preview_height = int(orig_height * (preview_width / orig_width))
            preview_image = image.resize((preview_width, preview_height), Image.Resampling.LANCZOS)
            preview_out_path = os.path.join(output_dir, "preview_" + out_filename)
            preview_image.save(preview_out_path, quality=85)
            print(f"Saved preview annotated image to {preview_out_path}")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    detect_towers_in_images("input_images", "output_images")
