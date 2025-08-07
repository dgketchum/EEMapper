import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from torchmetrics.classification import Accuracy

from unet import UNet1D

class NDVIDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        df = pd.read_parquet(self.file_list[idx])
        df.fillna(0, inplace=True)
        ndvi = torch.tensor(df.values, dtype=torch.float32).T
        label = int(os.path.basename(self.file_list[idx]).split('_')[-1].split('.')[0])
        if label == 4:
            label = 1
        return ndvi, torch.tensor(label, dtype=torch.long)

class NDVIClassifier(pl.LightningModule):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = UNet1D(in_channels, out_channels)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_accuracy = Accuracy(task="multiclass", num_classes=4)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=4)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def train_model(past_processed_dir, modern_processed_dir, batch_size=32, epochs=10):
    all_files = [os.path.join(past_processed_dir, f) for f in os.listdir(past_processed_dir) if f.endswith('.parquet')]
    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=42)

    with open('train_test_split.txt', 'w') as f:
        f.write('Training Files:\n')
        for item in train_files:
            f.write("%s\n" % item)
        f.write('\nValidation Files:\n')
        for item in val_files:
            f.write("%s\n" % item)
        f.write('\nTest Files:\n')
        for item in test_files:
            f.write("%s\n" % item)

    train_dataset = NDVIDataset(train_files)
    val_dataset = NDVIDataset(val_files)
    test_dataset = NDVIDataset(test_files)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    model = NDVIClassifier(in_channels=1, out_channels=4)

    trainer = pl.Trainer(max_epochs=epochs, accelerator='gpu', devices=1)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, dataloaders=test_loader)

    predict_modern_data(model, modern_processed_dir, batch_size)

def predict_modern_data(model, modern_processed_dir, batch_size):
    modern_dataset = NDVIDataset([os.path.join(modern_processed_dir, f) for f in
                                  os.listdir(modern_processed_dir) if f.endswith('.parquet')])
    modern_loader = DataLoader(modern_dataset, batch_size=batch_size, num_workers=4)

    predictions = []
    with torch.no_grad():
        for batch in modern_loader:
            x, _ = batch
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())

    fids = [os.path.basename(f).split('_')[0] for f in modern_dataset.file_list]
    results = pd.DataFrame({'FID': fids, 'predicted_class': predictions})
    results.to_csv('modern_predictions.csv', index=False)
    print("Modern data predictions saved to modern_predictions.csv")

if __name__ == '__main__':
    past_dir = '/data/ssd2/irrmapper/ndvi/past_processed/'
    modern_dir = '/data/ssd2/irrmapper/ndvi/modern_processed/'
    train_model(past_dir, modern_dir)

# ========================= EOF ====================================================================