from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(data_dir, batch_size=32, train=True, shuffle=True, num_workers=4):
    # transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),          # Resize images to 224x224
        transforms.ToTensor(),                 # Convert images to tensors
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize for pre-trained models
    ])

    # load dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


