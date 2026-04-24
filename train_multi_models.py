import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import numpy as np
import pandas as pd
from scipy.stats import beta
from tqdm import tqdm
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


class Config:
    num_classes = 45
    epochs = 30
    batch_size = 16
    lr = 3e-4
    data_path = '/home/featurize/data/Ilex_data46/Ilex_data'
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    img_size = 224
    device = 'cuda:0'
    random_state = 42
   
    models_to_compare = [
        'GoogleNet', 'ResNet50', 'ResNet101',
        'DenseNet121', 'DenseNet169', 'EfficientNet-B3'
    ]
   
    model_params = {
        'GoogleNet': 6.8, 'ResNet50': 25.6, 'ResNet101': 44.5,
        'DenseNet121': 8.0, 'DenseNet169': 14.2, 'EfficientNet-B3': 12.0,
    }
   
    freeze_epochs = 5
    patience = 10
    use_weighted_sampler = True
    use_label_smoothing = True
    label_smoothing_eps = 0.1
    gradient_clip_norm = 1.0
    use_mixup = True
    mixup_alpha = 0.2

cfg = Config()


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ All random seeds fixed to {seed}")


device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
print(f"🧠 Using device: {device}")

timestamp = time.strftime("%Y-%m-%d_%H-%M")
base_save_dir = f'./results/model_comparison_unified_{cfg.img_size}_{timestamp}'
os.makedirs(base_save_dir, exist_ok=True)
data_dir = os.path.join(base_save_dir, 'data')
os.makedirs(data_dir, exist_ok=True)


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(cfg.img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(cfg.img_size + 20),
            transforms.CenterCrop(cfg.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1):
        super().__init__()
        self.eps = eps
       
    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_preds = torch.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.eps / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.eps)
        return torch.mean(torch.sum(-true_dist * log_preds, dim=1))


def mixup_data(x, y, alpha=0.2):
    lam = beta.rvs(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def create_model(model_name, num_classes):
    print(f" Loading {model_name} with pretrained weights...")
   
    if model_name == 'GoogleNet':
        model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.aux_logits = False
    elif model_name == 'ResNet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'ResNet101':
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'DenseNet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'DenseNet169':
        model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'EfficientNet-B3':
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
   
    return model


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.mode = mode
       
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
            return False
       
        if self.mode == 'max':
            improved = val_score > self.best_score + self.min_delta
        else:
            improved = val_score < self.best_score - self.min_delta
           
        if improved:
            self.best_score = val_score
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def train_model(model_name, train_loader, val_loader, test_loader,
                train_dataset, val_dataset, test_dataset, class_names, save_dir):
   
    print(f"\n{'='*60}")
    print(f"🚀 Training {model_name}")
    print(f"{'='*60}")
   
    model = create_model(model_name, cfg.num_classes)
    model = model.to(device)
   
    for param in model.parameters():
        param.requires_grad = False
   
    if model_name.startswith('ResNet'):
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_name.startswith('DenseNet'):
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif model_name == 'GoogleNet':
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_name.startswith('EfficientNet'):
        for param in model.classifier.parameters():
            param.requires_grad = True
   
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
   
    criterion = LabelSmoothingCrossEntropy(eps=cfg.label_smoothing_eps) if cfg.use_label_smoothing else nn.CrossEntropyLoss()
   
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=cfg.patience, mode='max')
   
    best_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
   
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
   
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg.epochs} [Train]', leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
   
            if cfg.use_mixup and epoch >= cfg.freeze_epochs:
                x, y_a, y_b, lam = mixup_data(x, y, cfg.mixup_alpha)
                outputs = model(x)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                preds = outputs.argmax(dim=1)
                train_correct += preds.eq(y).sum().item()
            else:
                outputs = model(x)
                loss = criterion(outputs, y)
                preds = outputs.argmax(dim=1)
                train_correct += preds.eq(y).sum().item()
   
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
            optimizer.step()
   
            train_loss += loss.item()
            train_total += y.size(0)
   
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
   
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
   
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                preds = outputs.argmax(dim=1)
                val_loss += loss.item()
                val_correct += preds.eq(y).sum().item()
                val_total += y.size(0)
   
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
   
        print(f"Epoch {epoch+1:2d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
   
        if epoch + 1 == cfg.freeze_epochs:
            print(f"🔓 Unfreezing all layers for {model_name}")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=cfg.lr * 0.1)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
   
        scheduler.step(val_loss)
   
        if early_stopping(val_acc):
            print(f"⛔ Early stopping at epoch {epoch+1}")
            break
   
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{save_dir}/best_model.pth")

    model.load_state_dict(torch.load(f"{save_dir}/best_model.pth"))
    model.eval()
   
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
   
    test_acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
   
    curve_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    })
    curve_df.to_csv(f"{save_dir}/training_curves.csv", index=False)
   
    cm = confusion_matrix(all_labels, all_preds)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(f"{save_dir}/confusion_matrix_raw.csv")
   
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, None]
    pd.DataFrame(cm_norm, index=class_names, columns=class_names).to_csv(f"{save_dir}/confusion_matrix_normalized.csv")
   
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    f1_df = pd.DataFrame({'class_name': class_names, 'f1_score': f1_per_class})
    f1_df = f1_df.sort_values('f1_score', ascending=False)
    f1_df.to_csv(f"{save_dir}/per_class_f1.csv", index=False)
   
    per_class_metrics = []
    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
       
        per_class_metrics.append({
            'class_name': cls, 'precision': precision, 'recall': recall, 'f1_score': f1,
            'support': int(cm[i, :].sum()), 'correct': int(tp), 'errors': int(cm[i, :].sum() - tp)
        })
    pd.DataFrame(per_class_metrics).to_csv(f"{save_dir}/per_class_detailed_metrics.csv", index=False)
   
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    pd.DataFrame(report_dict).transpose().to_csv(f"{save_dir}/classification_report.csv")
   
    errors = cm.copy()
    np.fill_diagonal(errors, 0)
    error_pairs = [
        {'true_class': class_names[i], 'predicted_class': class_names[j],
         'count': int(errors[i, j]), 'proportion': errors[i, j] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0}
        for i in range(len(class_names)) for j in range(len(class_names))
        if i != j and errors[i, j] > 0
    ]
    if error_pairs:
        pd.DataFrame(error_pairs).sort_values('count', ascending=False).to_csv(f"{save_dir}/top_confusion_pairs.csv", index=False)
   
    summary_df = pd.DataFrame({
        'model': [model_name], 'test_accuracy': [test_acc],
        'macro_f1': [f1_macro], 'weighted_f1': [f1_weighted],
        'actual_epochs': [len(train_losses)], 'input_size': [cfg.img_size]
    })
    summary_df.to_csv(f"{save_dir}/model_summary.csv", index=False)
   
    print(f"\n✅ {model_name} Test Results: Acc={test_acc:.4f}, Macro F1={f1_macro:.4f}")
   
    return {
        'model': model_name,
        'test_acc': test_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'actual_epochs': len(train_losses)
    }


def main():
    print("\n" + "="*60)
    print("🌿 Multi-Model Comparison (Unified 224×224, Fully Reproducible)")
    print("="*60)
   
    set_all_seeds(cfg.random_state)
   
    train_tf = get_transforms(train=True)
    val_tf = get_transforms(train=False)
   
    full_dataset = datasets.ImageFolder(cfg.data_path, transform=val_tf)
    class_names = full_dataset.classes
    labels = [s[1] for s in full_dataset.samples]
   
    train_idx, temp_idx, _, temp_labels = train_test_split(
        list(range(len(full_dataset))), labels, stratify=labels,
        test_size=(1 - cfg.train_ratio), random_state=cfg.random_state)
   
    val_ratio_adjusted = cfg.val_ratio / (cfg.val_ratio + cfg.test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, stratify=temp_labels,
        test_size=(1 - val_ratio_adjusted), random_state=cfg.random_state)
   
    print(f"📂 Dataset: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
   
    class_names_df = pd.DataFrame({'index': range(len(class_names)), 'class_name': class_names})
    class_names_df.to_csv(f"{data_dir}/class_names.csv", index=False)
   
    all_results = []
   
    for model_name in cfg.models_to_compare:
        print(f"\n{'='*60}\nProcessing {model_name}")
        model_seed = cfg.random_state + cfg.models_to_compare.index(model_name)
        set_all_seeds(model_seed)
       
        train_dataset = Subset(datasets.ImageFolder(cfg.data_path, transform=train_tf), train_idx)
        val_dataset = Subset(datasets.ImageFolder(cfg.data_path, transform=val_tf), val_idx)
        test_dataset = Subset(datasets.ImageFolder(cfg.data_path, transform=val_tf), test_idx)
       
        class_counts = Counter([labels[i] for i in train_idx])
        sample_weights = [1.0 / class_counts[labels[i]] for i in train_idx]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights)) if cfg.use_weighted_sampler else None
       
        g = torch.Generator().manual_seed(model_seed)
       
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=sampler,
                                  shuffle=(sampler is None), num_workers=4, generator=g)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
       
        model_save_dir = os.path.join(data_dir, model_name)
        os.makedirs(model_save_dir, exist_ok=True)
       
        results = train_model(model_name, train_loader, val_loader, test_loader,
                              train_dataset, val_dataset, test_dataset, class_names, model_save_dir)
        all_results.append(results)
   
    pd.DataFrame(all_results).to_csv(f"{data_dir}/model_comparison_summary.csv", index=False)
   
    with open(f"{data_dir}/model_comparison_table.tex", 'w') as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Performance Comparison of Different CNN Models (Unified 224×224 Input)}\n")
        f.write("\\begin{tabular}{lcccc}\n\\hline\n")
        f.write("Model & Accuracy & Macro F1 & Weighted F1 & Params (M) \\\\\n\\hline\n")
        for r in all_results:
            params = cfg.model_params.get(r['model'], '-')
            f.write(f"{r['model']} & {r['test_acc']:.4f} & {r['f1_macro']:.4f} & {r['f1_weighted']:.4f} & {params} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\label{tab:model_comparison_unified}\n\\end{table}\n")
   
    print("\n" + "="*60)
    print("✅ ALL EXPERIMENTS COMPLETED (FULLY REPRODUCIBLE)!")
    print(f"📁 Results saved to: {data_dir}")
    print(f"📐 All models use unified input size: {cfg.img_size}×{cfg.img_size}")
   
    return all_results, data_dir


if __name__ == "__main__":
    results, data_dir = main()