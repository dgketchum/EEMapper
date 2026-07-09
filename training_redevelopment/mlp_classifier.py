import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from torchmetrics.classification import Accuracy, ConfusionMatrix
import torch.nn as nn
import json
from pprint import pprint


torch.set_float32_matmul_precision('medium')

CATEGORICAL_FEATURES = ['nlcd', 'cdl']


class MLPWithEmbeddings(nn.Module):
    def __init__(self, input_dim_continuous, cat_dims, output_dim, emb_dropout=0.1):
        super(MLPWithEmbeddings, self).__init__()

        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(num_embeddings, min(50, (num_embeddings + 1) // 2)) for num_embeddings in cat_dims])
        self.embedding_dropout = nn.Dropout(emb_dropout)

        embedding_output_size = sum([min(50, (num_embeddings + 1) // 2) for num_embeddings in cat_dims])

        self.continuous_mlp = nn.Sequential(
            nn.Linear(input_dim_continuous, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )

        self.final_mlp = nn.Sequential(
            nn.Linear(embedding_output_size + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

    def forward(self, x_continuous, x_categorical):
        embedded_features = [emb(x_categorical[:, i]) for i, emb in enumerate(self.embedding_layers)]
        embedded_features = torch.cat(embedded_features, 1)
        embedded_features = self.embedding_dropout(embedded_features)

        continuous_output = self.continuous_mlp(x_continuous)

        concatenated = torch.cat([continuous_output, embedded_features], 1)
        return self.final_mlp(concatenated)


class BandsDataset(Dataset):
    def __init__(self, file_list, all_features, categorical_mappings):
        self.file_list = file_list
        self.all_features = all_features
        self.categorical_mappings = categorical_mappings
        self.continuous_features = [f for f in all_features if f not in CATEGORICAL_FEATURES]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        df = pd.read_parquet(self.file_list[idx])

        for cat, mapping in self.categorical_mappings.items():
            df[cat] = df[cat].map(mapping).fillna(0).astype(int)

        continuous_data = torch.tensor(df[self.continuous_features].values, dtype=torch.float32).squeeze()
        categorical_data = torch.tensor(df[CATEGORICAL_FEATURES].values, dtype=torch.long).squeeze()

        label = int(os.path.basename(self.file_list[idx]).split('_')[-1].split('.')[0])
        if label == 4:
            label = 1
        return (continuous_data, categorical_data), torch.tensor(label, dtype=torch.long)


class BandsClassifier(pl.LightningModule):

    def __init__(self, input_dim_continuous, cat_dims, output_dim):
        super().__init__()
        self.model = MLPWithEmbeddings(input_dim_continuous, cat_dims, output_dim)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_accuracy = Accuracy(task="multiclass", num_classes=output_dim)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=output_dim)
        self.test_conf_matrix = ConfusionMatrix(task="multiclass", num_classes=output_dim)

    def forward(self, x):
        return self.model(x[0], x[1])

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


def train_model(past_processed_dir, categorical_json_path, batch_size=64, epochs=10):
    all_files = [os.path.join(past_processed_dir, f) for f in os.listdir(past_processed_dir) if f.endswith('.parquet')]

    sample_df = pd.read_parquet(all_files[0])
    all_feature_names = list(sample_df.columns)
    continuous_feature_count = len([f for f in all_feature_names if f not in CATEGORICAL_FEATURES])

    with open(categorical_json_path, 'r') as f:
        categorical_data = json.load(f)

    categorical_mappings = {}
    categorical_dims = []
    for cat in CATEGORICAL_FEATURES:
        unique_values = sorted(categorical_data[cat])
        mapping = {val: i for i, val in enumerate(unique_values)}
        categorical_mappings[cat] = mapping
        categorical_dims.append(len(unique_values))

    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=42)

    train_dataset = BandsDataset(train_files, all_feature_names, categorical_mappings)
    val_dataset = BandsDataset(val_files, all_feature_names, categorical_mappings)
    test_dataset = BandsDataset(test_files, all_feature_names, categorical_mappings)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

    model = BandsClassifier(input_dim_continuous=continuous_feature_count, cat_dims=categorical_dims, output_dim=4)

    trainer = pl.Trainer(max_epochs=epochs, accelerator='gpu', devices=1)
    trainer.fit(model, train_loader, val_loader)

    checkpoint_path = "best_mlp_model.ckpt"
    trainer.save_checkpoint(checkpoint_path)

    trainer.test(model, dataloaders=test_loader)

    return checkpoint_path, all_feature_names, categorical_mappings, train_files, test_files

def predict_modern_data(checkpoint_path, modern_processed_dir, batch_size, all_features, categorical_mappings):
    model = BandsClassifier.load_from_checkpoint(checkpoint_path)
    modern_dataset = BandsDataset([os.path.join(modern_processed_dir, f) for f in
                                   os.listdir(modern_processed_dir) if f.endswith('.parquet')], all_features,
                                  categorical_mappings)
    modern_loader = DataLoader(modern_dataset, batch_size=batch_size, num_workers=8)

    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in modern_loader:
            x, _ = batch
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())

    fids = [os.path.basename(f).split('_')[0] for f in modern_dataset.file_list]
    results = pd.DataFrame({'FID': fids, 'predicted_class': predictions})
    results.to_csv('modern_bands_predictions_with_embeddings.csv', index=False)
    print("Modern data predictions saved to modern_bands_predictions_with_embeddings.csv")




if __name__ == '__main__':
    past_dir = '/data/ssd2/irrmapper/bands/past_processed/'
    modern_dir = '/data/ssd2/irrmapper/bands/modern_processed/'

    categorical_map = '/data/ssd2/irrmapper/bands/past_processed/categorical.json'

    checkpoint, features, mappings, train_fs, test_fs = train_model(past_dir, categorical_map)
    # predict_modern_data(checkpoint, modern_dir, 64, features, mappings)

# ========================= EOF ====================================================================
