import argparse
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from model import ResNetModel
import os

"""
USAGE:
image directoey:
python inference.py --model_name resnet50 --image_path TLDataset/ALL/PKG-C-NMC2019/C-NMC_test_prelim_phase_data --model_path model_checkpoints/ALL/resnet50_best.pth --num_classes 2 --image_save output_folder
single image:
python inference.py --model_name resnet50 --image_path TLDataset/ALL/PKG-C-NMC2019/C-NMC_test_prelim_phase_data/test.png --model_path model_checkpoints/ALL/resnet50_best.pth --num_classes 2 --image_save out.png
"""

def load_image(image_path, device):
    """Load and preprocess an image for inference."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Match training normalization
    ])
    image = Image.open(image_path).convert('RGB')  # Ensure image is RGB
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    return image


def predict(model, image_path, device):
    """Perform inference using the given model and image."""
    image = load_image(image_path, device)
    model.eval()

    # Perform inference
    with torch.no_grad():  # Disable gradients for faster computation
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)  # Get probabilities
        confidence, predicted = torch.max(probabilities, 0)  # Get class index and confidence
    return predicted.item(), confidence.item()


def save_image_with_label(image_path, label, confidence, save_path):
    """Save the image with predicted label and confidence displayed."""
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    text = f"Label: {label}, Confidence: {confidence:.2f}"
    draw.text((10, 10), text, fill='red', font=font)
    image.save(save_path)


def is_valid_image(filename):
    """Check if the file has a valid image extension."""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}
    return os.path.splitext(filename)[1].lower() in valid_extensions


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Inference with a trained model')
    parser.add_argument('--model_name', type=str, default='resnet18', help='Model architecture used for training')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image or directory for inference')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes in the model')
    parser.add_argument('--image_save', type=str, default=None, help='Directory to save images with label and confidence')

    args = parser.parse_args()

    # Set device (use GPU if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = ResNetModel(model_name=args.model_name, num_classes=args.num_classes, pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))  # Load model weights
    model.to(device)

    # Class names mapping
    class_names = ['CANCER', 'NO_CANCER']

    # Perform inference
    if os.path.isdir(args.image_path):
        for img_name in os.listdir(args.image_path):
            img_path = os.path.join(args.image_path, img_name)
            if os.path.isfile(img_path) and is_valid_image(img_name):
                predicted_class, confidence = predict(model, img_path, device)
                class_label = class_names[predicted_class]
                print(f'{img_name}: Predicted class: {class_label} ({predicted_class}), Confidence: {confidence:.2f}')
                if args.image_save:
                    os.makedirs(args.image_save, exist_ok=True)
                    save_path = os.path.join(args.image_save, img_name)
                    save_image_with_label(img_path, f"{class_label} ({predicted_class})", confidence, save_path)
    elif os.path.isfile(args.image_path) and is_valid_image(args.image_path):
        predicted_class, confidence = predict(model, args.image_path, device)
        class_label = class_names[predicted_class]
        print(f'Predicted class: {class_label} ({predicted_class}), Confidence: {confidence:.2f}')
        if args.image_save:
            save_image_with_label(args.image_path, f"{class_label} ({predicted_class})", confidence, args.image_save)
    else:
        print("Invalid image path or unsupported file format.")
