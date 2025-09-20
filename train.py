import argparse
import torch
import pytorch_lightning as pl
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from PIL import Image
from model import MultiClassCNN

class ImageDataset(datasets.ImageFolder):
    """Custom dataset class for loading images and labels from a directory structure."""
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        
    def __len__(self) -> int:
        return super().__len__()
    
    def Classes(self):
        return self.classes
        
    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class TrainDataModule(pl.LightningDataModule):
    """DataModule for handling data loading and preprocessing."""
    def __init__(self, data_dir, batch_size=32, num_workers=4, val_split=0.2,augment=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        full_dataset = ImageDataset(root=self.data_dir, transform=transform)
        loader = DataLoader(full_dataset, batch_size=len(full_dataset), shuffle=False)
        data = next(iter(loader))[0]
        self.mean = torch.mean(data, dim=[0, 2, 3])
        self.std = torch.std(data, dim=[0, 2, 3])


    def setup(self, stage=None, augment=False):
        if augment:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),transforms.Normalize(mean=self.mean, std=self.std)])
        full_dataset = ImageDataset(root=self.data_dir, transform=transform)
        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True,persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True,persistent_workers=True)
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="data/Animals", help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--max_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    args = parser.parse_args()
    
    device = torch.device('cuda' if args.gpus else 'cpu')
    batch_size ,height ,width = args.batch_size , 224 , 224
    data_module = TrainDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, augment=args.augment)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    classes = data_module.train_dataset.dataset.classes
    
    model = MultiClassCNN(num_classes=args.num_classes)
    
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() and args.gpus > 0 else "cpu",
        devices=args.gpus if torch.cuda.is_available() and args.gpus > 0 else 1,
        max_epochs=args.max_epochs
    )

    trainer.fit(model, train_loader, val_loader)