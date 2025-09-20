import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch import optim

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
        
    def forward(self, x):
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