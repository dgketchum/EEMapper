import json
import os
from datetime import datetime
from pprint import pprint

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, ConfusionMatrix
from pytorch_lightning.callbacks import EarlyStopping

from combined_dataset import PreloadedDataset
from combined_dataset import load_and_preprocess_data
from training_redevelopment.get_files import get_files
from training_redevelopment import RANDOM_SEED, NUMBER_WORKERS, NUMBER_CLASSES

torch.set_float32_matmul_precision('medium')

from unet import UNet1D


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
        self.save_hyperparameters()
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
        df_cm = pd.DataFrame(conf_matrix.cpu().numpy(), index=range(NUMBER_CLASSES), columns=range(NUMBER_CLASSES))
        df_cm.to_csv('confusion_matrix.csv')
        print("\nConfusion matrix saved to confusion_matrix.csv")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)


def train_model(past_bands_dir, past_ndvi_dir, categorical_json_path, batch_size=64, epochs=10, checkpoint_dir=None,
                sample=None, debug=False):
    pl.seed_everything(RANDOM_SEED, workers=True)
    all_files = get_files(past_bands_dir, past_ndvi_dir, sample)

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

    train_val_data, test_data = train_test_split(all_data, test_size=0.2, random_state=RANDOM_SEED)
    train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=RANDOM_SEED)

    train_dataset = PreloadedDataset(train_data)
    val_dataset = PreloadedDataset(val_data)
    test_dataset = PreloadedDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUMBER_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=NUMBER_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=NUMBER_WORKERS, pin_memory=True)

    model = CombinedClassifier(input_dim_continuous=continuous_feature_count, cat_dims=categorical_dims,
                               unet_out_channels=NUMBER_CLASSES, output_dim=NUMBER_CLASSES)

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
            'unet_out_channels': NUMBER_CLASSES,
            'output_dim': NUMBER_CLASSES,
            'all_features': all_feature_names,
            'categorical_mappings': categorical_mappings
        }
        metadata_path = os.path.join(checkpoint_dir, f'metadata_{datestamp}.json')
        print(f'Saving Model Metadata {metadata_path}\n')

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True,
        mode='min'
    )
    callbacks.append(early_stop_callback)

    trainer = pl.Trainer(max_epochs=epochs, accelerator='gpu', devices=1, callbacks=callbacks)
    trainer.fit(model, train_loader, val_loader)

    if checkpoint_dir:
        print(f'Loading best model from {callbacks[0].best_model_path}')
        model = CombinedClassifier.load_from_checkpoint(callbacks[0].best_model_path)

    trainer.test(model, dataloaders=test_loader)


def evaluate_and_compare(past_bands_dir, past_ndvi_dir, checkpoint_dir, checkpoint_filename=None,
                         batch_size=64, sample=None, debug=False):
    """"""
    pl.seed_everything(RANDOM_SEED, workers=True)

    if not checkpoint_filename:
        ckpts = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        latest_ckpt = max([os.path.join(checkpoint_dir, f) for f in ckpts], key=os.path.getmtime)
        checkpoint_filename = latest_ckpt
    else:
        checkpoint_filename = os.path.join(checkpoint_dir, checkpoint_filename)

    datestamp = os.path.basename(checkpoint_filename).replace('model_', '').replace('.ckpt', '')
    checkpoint_metadata = os.path.join(checkpoint_dir, f'metadata_{datestamp}.json')

    with open(checkpoint_metadata, 'r') as f:
        metadata = json.load(f)

    # recast mapping keys to int
    for cat, map in metadata['categorical_mappings'].items():
        metadata['categorical_mappings'][cat] = {int(k): v for k, v in map.items()}

    all_files = get_files(past_bands_dir, past_ndvi_dir, sample)
    all_data = load_and_preprocess_data(all_files, past_bands_dir, past_ndvi_dir,
                                        metadata['all_features'], metadata['categorical_mappings'], debug=debug)

    train_val_data, test_data = train_test_split(all_data, test_size=0.2, random_state=RANDOM_SEED)
    train_data, _ = train_test_split(train_val_data, test_size=0.2, random_state=RANDOM_SEED)

    print(f'Loading Checkpoint: {checkpoint_filename}')
    print(f'Loading Model Metadata: {checkpoint_metadata}')

    model = CombinedClassifier.load_from_checkpoint(checkpoint_filename)
    model.eval()

    test_dataset = PreloadedDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=NUMBER_WORKERS, pin_memory=True)

    trainer = pl.Trainer(accelerator='gpu', devices=1)
    print("\n--- Combined Classifier Test ---")
    trainer.test(model, dataloaders=test_loader)

    print("\n--- RandomForest Comparison ---")

    def prepare_data_for_sklearn(data):
        X, y = [], []
        for (continuous, categorical, _), label in data:
            feature_vector = np.concatenate((continuous, categorical.float()))
            X.append(feature_vector)
            y.append(label)
        return np.array(X), np.array(y)

    X_train, y_train = prepare_data_for_sklearn(train_data)
    X_test, y_test = prepare_data_for_sklearn(test_data)

    rf = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nRandomForest Accuracy: {acc:.4f}")

    print("\nRandomForest Confusion Matrix:")
    pprint(sk_confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    past_bands_dir_ = '/data/ssd2/irrmapper/states/bands/past_processed/'
    past_ndvi_dir_ = '/data/ssd2/irrmapper/states/timeseries/past_processed/'

    modern_bands_dir_ = '/data/ssd2/irrmapper/states/bands/modern_processed/'
    modern_ndvi_dir_ = '/data/ssd2/irrmapper/states/timeseries/modern_processed/'

    checkpoint_dir_ = '/data/ssd2/irrmapper/states/combined_classifier'
    categorical_meta_ = os.path.join(checkpoint_dir_, 'categorical_20250811.json')

    batch_sz = 128
    debug_mode = False
    sample_size = None

    # current /data/ssd2/irrmapper/states/combined_classifier/model_20250811_124746.ckpt

    # train_model(past_bands_dir_, past_ndvi_dir_, categorical_meta_, sample=sample_size,
    #            batch_size=batch_sz, epochs=100, checkpoint_dir=checkpoint_dir_, debug=debug_mode)

    # all points
    checkpoint_ = 'model_20250812_100850.ckpt'

    evaluate_and_compare(past_bands_dir_, past_ndvi_dir_, checkpoint_dir_, checkpoint_filename=None,
                         batch_size=batch_sz, sample=sample_size, debug=debug_mode)

# ========================= EOF ====================================================================
