import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
from sklearn.model_selection import train_test_split
from torchmetrics.classification import Accuracy, ConfusionMatrix
import torch.nn as nn
import json
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from training_redevelopment.combined_dataset import load_and_preprocess_data, _load_and_preprocess_worker
from training_redevelopment.combined_dataset import PreloadedDataset

torch.set_float32_matmul_precision('medium')

from unet import UNet1D

NUMBER_WORKERS = 16


class CombinedModel(nn.Module):
    def __init__(self, input_dim_continuous, cat_dims, unet_out_channels, output_dim):
        super(CombinedModel, self).__init__()

        self.mlp_embedding_layers = nn.ModuleList(
            [nn.Embedding(num_embeddings, min(50, (num_embeddings + 1) // 2)) for num_embeddings in cat_dims])
        embedding_output_size = sum([min(50, (num_embeddings + 1) // 2) for num_embeddings in cat_dims])
        self.mlp_continuous = nn.Sequential(
            nn.Linear(input_dim_continuous, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        self.mlp_feature_size = 256 + embedding_output_size

        self.unet = UNet1D(in_channels=1, out_channels=unet_out_channels)
        self.unet_feature_size = unet_out_channels

        self.fusion_head = nn.Sequential(
            nn.Linear(self.mlp_feature_size + self.unet_feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, output_dim)
        )

    def forward(self, x_continuous, x_categorical, x_ndvi):
        embedded_features = [emb(x_categorical[:, i]) for i, emb in enumerate(self.mlp_embedding_layers)]
        embedded_features = torch.cat(embedded_features, 1)
        continuous_output = self.mlp_continuous(x_continuous)
        mlp_features = torch.cat([continuous_output, embedded_features], 1)

        unet_features = self.unet(x_ndvi)
        combined_features = torch.cat([mlp_features, unet_features], dim=1)
        output = self.fusion_head(combined_features)

        return output


class CombinedClassifier(pl.LightningModule):
    def __init__(self, input_dim_continuous, cat_dims, unet_out_channels, output_dim):
        super().__init__()
        self.model = CombinedModel(input_dim_continuous, cat_dims, unet_out_channels, output_dim)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_accuracy = Accuracy(task="multiclass", num_classes=output_dim)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=output_dim)
        self.test_conf_matrix = ConfusionMatrix(task="multiclass", num_classes=output_dim)

    def forward(self, x):
        return self.model(x[0], x[1], x[2])

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
        self.test_conf_matrix.update(preds, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        conf_matrix = self.test_conf_matrix.compute()
        print("\nConfusion Matrix:")
        pprint(conf_matrix)
        df_cm = pd.DataFrame(conf_matrix.cpu().numpy(), index=range(4), columns=range(4))
        df_cm.to_csv('confusion_matrix.csv')
        print("\nConfusion matrix saved to confusion_matrix.csv")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


def get_files(bands, ndvi):
    ndvi_files = {f for f in os.listdir(ndvi) if f.endswith('.parquet')}
    all_files = []
    for f in os.listdir(bands):
        if not f.endswith('.parquet'):
            continue
        parts = f.split('_')
        ndvi_f = '_'.join(parts[:3] + parts[4:])
        if ndvi_f in ndvi_files:
            all_files.append((f, ndvi_f))
    return all_files


def train_model(past_bands_dir, past_ndvi_dir, categorical_json_path, batch_size=64, epochs=10, checkpoint_dir=None,
                debug=False):
    all_files = get_files(past_bands_dir, past_ndvi_dir)

    with open(categorical_json_path, 'r') as f:
        categorical_data = json.load(f)

    sample_df = pd.read_parquet(os.path.join(past_bands_dir, all_files[0][0]))
    all_feature_names = list(sample_df.columns)
    categorical_features = sorted(list(categorical_data.keys()))
    continuous_feature_count = len([f for f in all_feature_names if f not in categorical_features])

    categorical_mappings = {}
    categorical_dims = []
    for cat in categorical_features:
        unique_values = sorted(categorical_data[cat])
        mapping = {val: i for i, val in enumerate(unique_values)}
        categorical_mappings[cat] = mapping
        categorical_dims.append(len(unique_values))

    all_data = load_and_preprocess_data(all_files, past_bands_dir, past_ndvi_dir, all_feature_names,
                                        categorical_mappings, num_workers=NUMBER_WORKERS, debug=debug)

    train_val_data, _ = train_test_split(all_data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=42)

    train_dataset = PreloadedDataset(train_data)
    val_dataset = PreloadedDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUMBER_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=NUMBER_WORKERS, pin_memory=True)

    model = CombinedClassifier(input_dim_continuous=continuous_feature_count, cat_dims=categorical_dims,
                               unet_out_channels=4, output_dim=4)

    callbacks = []
    if checkpoint_dir:
        datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f'model_{datestamp}',
            save_top_k=1,
            verbose=True,
            monitor='val_loss',
            mode='min'
        )
        callbacks.append(checkpoint_callback)

        metadata = {
            'input_dim_continuous': continuous_feature_count,
            'cat_dims': categorical_dims,
            'unet_out_channels': 4,
            'output_dim': 4,
            'all_features': all_feature_names,
            'categorical_mappings': categorical_mappings
        }
        metadata_path = os.path.join(checkpoint_dir, f'metadata_{datestamp}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

    trainer = pl.Trainer(max_epochs=epochs, accelerator='gpu', devices=1, callbacks=callbacks)
    trainer.fit(model, train_loader, val_loader)


def evaluate_and_compare(past_bands_dir, past_ndvi_dir, checkpoint_dir, batch_size=64, debug=False):
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    latest_ckpt = max([os.path.join(checkpoint_dir, f) for f in ckpts], key=os.path.getmtime)
    checkpoint_path = latest_ckpt

    datestamp = os.path.basename(latest_ckpt).replace('model_', '').replace('.ckpt', '')
    checkpoint_metadata = os.path.join(checkpoint_dir, f'metadata_{datestamp}.json')

    with open(checkpoint_metadata, 'r') as f:
        metadata = json.load(f)

    all_files = get_files(past_bands_dir, past_ndvi_dir)
    all_data = load_and_preprocess_data(all_files, past_bands_dir, past_ndvi_dir,
                                        metadata['all_features'], metadata['categorical_mappings'], debug=debug)

    train_val_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
    train_data, _ = train_test_split(train_val_data, test_size=0.2, random_state=42)

    model = CombinedClassifier.load_from_checkpoint(
        checkpoint_path,
        input_dim_continuous=metadata['input_dim_continuous'],
        cat_dims=metadata['cat_dims'],
        unet_out_channels=metadata['unet_out_channels'],
        output_dim=metadata['output_dim']
    )

    test_dataset = PreloadedDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=NUMBER_WORKERS, pin_memory=True)

    trainer = pl.Trainer(accelerator='gpu', devices=1)
    print("\n--- Combined Classifier Test ---")
    trainer.test(model, dataloaders=test_loader)

    print("\n--- RandomForest Comparison ---")

    def prepare_data_for_sklearn(data):
        X, y = [], []
        for (continuous, categorical, _), label in data:
            feature_vector = np.concatenate((continuous, categorical.astype(np.float32)))
            X.append(feature_vector)
            y.append(label)
        return np.array(X), np.array(y)

    X_train, y_train = prepare_data_for_sklearn(train_data)
    X_test, y_test = prepare_data_for_sklearn(test_data)

    rf = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    print("\nRandomForest Confusion Matrix:")
    pprint(sk_confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    past_bands_dir_ = '/data/ssd2/irrmapper/states/bands/past_processed/'
    past_ndvi_dir_ = '/data/ssd2/irrmapper/states/timeseries/past_processed/'

    modern_bands_dir_ = '/data/ssd2/irrmapper/states/bands/modern_processed/'
    modern_ndvi_dir_ = '/data/ssd2/irrmapper/states/timeseries/modern_processed/'

    checkpoint_dir_ = '/data/ssd2/irrmapper/states/combined_classifier'
    categorical_meta_ = os.path.join(checkpoint_dir_, 'categorical_20250811.json')

    debug_mode = False
    train_model(past_bands_dir_, past_ndvi_dir_, categorical_meta_,
                batch_size=128, epochs=5, checkpoint_dir=checkpoint_dir_, debug=debug_mode)
    # evaluate_and_compare(past_bands_dir_, past_ndvi_dir_, checkpoint_dir_, debug=debug_mode)

# ========================= EOF ====================================================================
