import argparse
import torch
from torchvision import transforms
from PIL import Image
from model import ResNetModel


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
        _, predicted = torch.max(outputs, 1)  # Get class index with highest probability
    return predicted.item()


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Inference with a trained model')
    parser.add_argument('--model_name', type=str, default='resnet18', help='Model architecture used for training')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image for inference')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes in the model')

    args = parser.parse_args()

    # Set device (use GPU if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    # Initialize model
    model = ResNetModel(model_name=args.model_name, num_classes=args.num_classes, pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))  # Load model weights
    model.to(device)

    # Perform inference
    prediction = predict(model, args.image_path, device)
    print(f'Predicted class: {prediction}')
