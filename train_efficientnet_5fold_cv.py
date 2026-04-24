# =========================================
# EfficientNet-B3 - 5-fold CV (Fully Fixed Version)
# =========================================
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
from collections import Counter
import glob

# ====================== Configuration ======================
class Config:
    num_classes = 45
    epochs = 30
    batch_size = 16
    lr = 0.0003
    data_path = '/home/featurize/data/Ilex_data46/Ilex_data'
    seed = 42
    n_folds = 5
    device = 'cuda:0'

cfg = Config()
device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
print(f"🧠 Using device: {device}")

# ====================== Fix random seed ======================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(cfg.seed)

# ====================== Data augmentation ======================
img_size = 300

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(img_size + 20),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ====================== Custom Dataset ======================
class IlexDataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# ====================== Load all data ======================
print("\n📂 Loading dataset...")
full_dataset = datasets.ImageFolder(root=cfg.data_path, transform=None)

all_img_paths = [s[0] for s in full_dataset.samples]
all_labels = [s[1] for s in full_dataset.samples]
class_names = full_dataset.classes

print(f"Total samples: {len(all_img_paths)}")
print(f"Number of classes: {len(class_names)}")

# ====================== Check 1: Data loading ======================
print("\n" + "="*60)
print("🔍 Check 1: Data loading")
print("="*60)

label_counts = Counter(all_labels)
print(f"\nClass distribution (first 5 classes):")
for cls_idx in range(min(5, len(class_names))):
    count = label_counts[cls_idx]
    print(f"  Class {cls_idx} ({class_names[cls_idx]}): {count} images")

trainval_idx, test_idx, trainval_labels, test_labels = train_test_split(
    range(len(all_img_paths)), 
    all_labels,
    test_size=0.2,
    stratify=all_labels,
    random_state=cfg.seed
)

print(f"\nSplit result:")
print(f"  Train+Val: {len(trainval_idx)} ({len(trainval_idx)/len(all_labels)*100:.1f}%)")
print(f"  Test: {len(test_idx)} ({len(test_idx)/len(all_labels)*100:.1f}%)")

trainval_paths = [all_img_paths[i] for i in trainval_idx]
trainval_labels_list = [all_labels[i] for i in trainval_idx]
test_paths = [all_img_paths[i] for i in test_idx]
test_labels_final = [all_labels[i] for i in test_idx]

# ====================== Check 2: Train/Test leakage ======================
print("\n" + "="*60)
print("🔍 Check 2: Data leakage")
print("="*60)

trainval_set = set(trainval_paths)
test_set = set(test_paths)
overlap_train_test = trainval_set & test_set
print(f"⚠️ Train+Val and Test overlap: {len(overlap_train_test)} (should be 0)")

if len(overlap_train_test) > 0:
    print("❌ Critical error: test set overlaps with training set!")
    print("Overlap examples:", list(overlap_train_test)[:3])
    raise ValueError("Data leakage detected!")

all_paths_set = set(all_img_paths)
print(f"Total paths: {len(all_img_paths)} | Unique paths: {len(all_paths_set)}")
if len(all_img_paths) != len(all_paths_set):
    print("⚠️ Warning: duplicate files found")

# ====================== Check 3: Sample image content ======================
print("\n" + "="*60)
print("🔍 Check 3: Sample image content")
print("="*60)

sample_indices = random.sample(range(len(trainval_paths)), 5)
print("\nSampling Train+Val images:")
for i, idx in enumerate(sample_indices):
    img_path = trainval_paths[idx]
    img = Image.open(img_path)
    print(f"  Sample{i}: {os.path.basename(img_path)}, size={img.size}, class={class_names[trainval_labels_list[idx]]}")

raw_pixels = []
for idx in sample_indices[:3]:
    img = Image.open(trainval_paths[idx]).convert('RGB')
    raw_pixels.append(np.array(img))
raw_pixels = np.array(raw_pixels)
print(f"\nRaw pixel stats (3 samples):")
print(f"  Range: [{raw_pixels.min()}, {raw_pixels.max()}]")
print(f"  Mean: {raw_pixels.mean():.1f}")
print(f"  Std: {raw_pixels.std():.1f}")

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, idx in enumerate(sample_indices):
    img = Image.open(trainval_paths[idx])
    axes[i].imshow(img)
    axes[i].set_title(f"{class_names[trainval_labels_list[idx]]}", fontsize=8)
    axes[i].axis('off')
plt.tight_layout()
plt.savefig('/tmp/sample_train_images.png', dpi=150)
print(f"\nImage saved: /tmp/sample_train_images.png")

# ====================== Create test set ======================
test_dataset = IlexDataset(test_paths, test_labels_final, val_transform)
test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

print("="*60)

# ====================== 5-fold cross validation ======================
skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)

timestamp = time.strftime("%Y-%m-%d_%H-%M")
base_dir = f"./results/{timestamp}_Ilex45_5Fold"
os.makedirs(base_dir, exist_ok=True)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(trainval_paths, trainval_labels_list)):
    print(f"\n{'='*60}")
    print(f"🔄 Fold {fold+1}/{cfg.n_folds}")
    print(f"{'='*60}")
    
    fold_dir = os.path.join(base_dir, f"fold_{fold+1}")
    os.makedirs(fold_dir, exist_ok=True)
    
    fold_train_paths = [trainval_paths[i] for i in train_idx]
    fold_train_labels = [trainval_labels_list[i] for i in train_idx]
    fold_val_paths = [trainval_paths[i] for i in val_idx]
    fold_val_labels = [trainval_labels_list[i] for i in val_idx]
    
    # ====================== Check 4: Current fold data ======================
    print(f"\n🔍 Fold {fold+1} data check:")
    print(f"  Train: {len(fold_train_paths)} | Val: {len(fold_val_paths)}")
    
    train_set_paths = set(fold_train_paths)
    val_set_paths = set(fold_val_paths)
    overlap = train_set_paths & val_set_paths
    print(f"  Train/Val path overlap: {len(overlap)} (should be 0)")
    
    train_dataset = IlexDataset(fold_train_paths, fold_train_labels, train_transform)
    val_dataset = IlexDataset(fold_val_paths, fold_val_labels, val_transform)
    
    sample_img, sample_label = train_dataset[0]
    print(f"  Train sample: shape={sample_img.shape}, range=[{sample_img.min():.2f}, {sample_img.max():.2f}]")
    
    sample_val_img, sample_val_label = val_dataset[0]
    print(f"  Val sample: shape={sample_val_img.shape}, range=[{sample_val_img.min():.2f}, {sample_val_img.max():.2f}]")
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
    
    # ====================== Model ======================
    print("  🧱 Creating model...")
    
    existing = glob.glob('./results/*/fold_*/best_model.pth')
    if existing:
        print(f"    Found old models: {len(existing)} (ignored)")
    
    net = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    in_features = net.classifier[1].in_features
    
    net.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, cfg.num_classes)
    )
    net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    
    # ====================== Training ======================
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    best_model_path = os.path.join(fold_dir, 'best_model.pth')
    
    train_loss_all, val_loss_all = [], []
    train_acc_all, val_acc_all = [], []
    
    for epoch in range(cfg.epochs):
        net.train()
        train_loss, train_correct = 0.0, 0
        for images, labels in tqdm(train_loader, leave=False, desc=f"E{epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
        
        net.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_acc = train_correct / len(train_dataset)
        avg_val_acc = val_correct / len(val_dataset)
        
        train_loss_all.append(avg_train_loss)
        val_loss_all.append(avg_val_loss)
        train_acc_all.append(avg_train_acc)
        val_acc_all.append(avg_val_acc)
        
        if epoch == 0:
            print(f"\n  🔍 First epoch check:")
            print(f"    Train Loss: {avg_train_loss:.4f} (expected 2.0~4.0)")
            print(f"    Train Acc: {avg_train_acc:.4f} (expected 0.05~0.15)")
            print(f"    Val Acc: {avg_val_acc:.4f}")
            
            with torch.no_grad():
                sample_out = net(images[:1])
                probs = torch.softmax(sample_out, dim=1)
                top3 = probs.topk(3)
                print(f"    Output probs top3: {top3.values.cpu().numpy()[0]}")
                print(f"    Corresponding classes: {[class_names[i] for i in top3.indices.cpu().numpy()[0]]}")
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    E{epoch+1:2d} | Loss: {avg_train_loss:.3f}/{avg_val_loss:.3f} | Acc: {avg_train_acc:.3f}/{avg_val_acc:.3f}")
        
        scheduler.step()
        
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(net.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    ⏹️ Early stop @ E{epoch+1}")
                break
    
    print(f"    ✅ Best Val Acc: {best_val_acc:.4f}")
    
    df = pd.DataFrame({
        "Epoch": range(1, len(train_loss_all)+1),
        "Train_Loss": train_loss_all, "Val_Loss": val_loss_all,
        "Train_Acc": train_acc_all, "Val_Acc": val_acc_all
    })
    df.to_csv(os.path.join(fold_dir, 'metrics.csv'), index=False)
    
    # ====================== Testing ======================
    print(f"    🧪 Testing...")
    net.load_state_dict(torch.load(best_model_path))
    net.eval()
    
    test_correct = 0
    all_preds, all_labels_test = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            preds = outputs.argmax(1)
            test_correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels_test.extend(labels.cpu().numpy())
    
    test_acc = test_correct / len(test_dataset)
    fold_results.append(test_acc)
    
    print(f"\n    🔍 Fold {fold+1} test results:")
    print(f"      Test Acc: {test_acc:.4f}")
    
    pred_counts = Counter(all_preds)
    zero_pred_classes = [i for i in range(cfg.num_classes) if pred_counts[i] == 0]
    print(f"      Classes never predicted: {len(zero_pred_classes)}/45")
    
    if len(zero_pred_classes) > 0:
        print(f"      Unpredicted classes: {[class_names[i] for i in zero_pred_classes[:5]]}...")
    
    report = classification_report(all_labels_test, all_preds, 
                                   target_names=class_names, digits=4, zero_division=0)
    with open(os.path.join(fold_dir, 'report.txt'), 'w') as f:
        f.write(f"Fold {fold+1} Test Acc: {test_acc:.4f}\n\n{report}")
    
    print(f"    ✅ Fold {fold+1} completed")

# ====================== Summary ======================
print(f"\n{'='*60}")
print(f"📊 5-Fold Final Summary")
print(f"{'='*60}")
for i, acc in enumerate(fold_results, 1):
    print(f"  Fold {i}: {acc:.4f}")

mean_acc, std_acc = np.mean(fold_results), np.std(fold_results)
print(f"\n  Mean ± Std: {mean_acc:.4f} ± {std_acc:.4f}")

with open(os.path.join(base_dir, 'summary.txt'), 'w') as f:
    f.write(f"5-Fold CV Results (seed={cfg.seed})\n")
    f.write(f"Data path: {cfg.data_path}\n")
    f.write(f"Total samples: {len(all_img_paths)}\n\n")
    for i, acc in enumerate(fold_results, 1):
        f.write(f"Fold {i}: {acc:.4f}\n")
    f.write(f"\nMean ± Std: {mean_acc:.4f} ± {std_acc:.4f}\n")

print(f"\n✅ All done! Results saved to: {base_dir}")