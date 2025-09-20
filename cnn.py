import argparse
import os
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from torch import optim
from torchinfo import summary
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

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


class MultiClassCNN(pl.LightningModule):
    """A simple CNN for multi-class image classification."""
    def __init__(self, num_classes=3):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 56 * 56, 128)  
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(128, num_classes)
        
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, x, *args, **kwargs):
        # Convolutional + pooling
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        # Fully connected
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        
        # Output layer (logits, no softmax here)
        x = self.fc2(x)
        return x
    
    def _common_step(self,batch,batch_idx):
        images,labels = batch
        outputs = self(images)
        loss = self.loss_fn(outputs,labels)
        predictions = torch.argmax(outputs,dim=1)
        acc = (predictions == labels).float().mean()
        return loss, outputs, labels, acc
    
    def training_step(self,batch,batch_idx):
        loss, outputs, labels, acc = self._common_step(batch,batch_idx)
        self.log('train_loss',loss)
        return loss
    
    def validation_step(self,batch,batch_idx):
        loss, outputs, labels, acc = self._common_step(batch,batch_idx)
        self.log('val_loss',loss,prog_bar=True)
        self.log('val_acc',acc,prog_bar=True)
        
    
    def test_step(self,batch,batch_idx):
        loss, outputs, labels, acc = self._common_step(batch,batch_idx)
        self.log('test_loss',loss,prog_bar=True)
        self.log('test_acc',acc,prog_bar=True)
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
    
    def predict_step(self, batch, batch_idx):
        images, _ = batch   # ignore labels during prediction
        outputs = self(images)
        preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        return preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="data/Animals", help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--max_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--show_accuracy', action='store_true', help='Show accuracy after each epoch')
    parser.add_argument('--summary', action='store_true', help='Print model summary and exit')
    args = parser.parse_args()

    device = torch.device('cuda' if args.gpus else 'cpu')
    batch_size ,height ,width = args.batch_size , 224 , 224
    data_module = TrainDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, augment=args.augment)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    classes = data_module.train_dataset.dataset.classes
    
    model = MultiClassCNN(num_classes=args.num_classes)
    
    if args.summary:
        summary(model, input_size=(batch_size, 3, height, width))
        exit()

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() and args.gpus > 0 else "cpu",
        devices=args.gpus if torch.cuda.is_available() and args.gpus > 0 else 1,
        max_epochs=args.max_epochs
    )

    trainer.fit(model, train_loader, val_loader)
    
    if args.show_accuracy:
        # Collect true labels
        y_true = []
        for _, labels in val_loader:
            y_true.extend(labels.tolist())
        y_true = np.array(y_true)

        # Collect predictions
        predictions_batches = trainer.predict(model, val_loader)
        y_pred = torch.cat(predictions_batches).cpu().numpy()

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        print("Training Accuracy:")
        trainer.validate(model, train_loader)
        print("Validation Accuracy:")
        trainer.validate(model, val_loader)
        print("Confusion Matrix:")
        print("Overall Accuracy:", accuracy_score(y_true, y_pred))
        disp.plot(cmap="Blues", xticks_rotation=45)
        plt.title("Confusion Matrix - Validation Set")
        plt.tight_layout()
        plt.show()
        exit()