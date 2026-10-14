import os
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor, RobertaTokenizer
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from src.models import HuBERTRobertaGatedClassifier
from src.dataset import MultimodalPodcastDataset, get_collate_fn
from src.losses import FocalLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTION_NAMES = ['Angry', 'Sad', 'Happy', 'Surprise', 'Fear', 'Disgust', 'Contempt', 'Neutral']
OUTPUT_DIR = "results_hubert_roberta_gating"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_training():
    audio_proc = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    text_tok = RobertaTokenizer.from_pretrained("roberta-base")
    collate_fn = get_collate_fn(audio_proc, text_tok)
    
    audio_dir = "/work/lab26/soe/MSP-Podcast_Challenge_IS2025/dataset/All_Audios"
    
    train_loader = DataLoader(
        MultimodalPodcastDataset("labels_with_text.csv", "Train", audio_dir=audio_dir), 
        batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        MultimodalPodcastDataset("labels_with_text.csv", "Test1", audio_dir=audio_dir), 
        batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    
    model = HuBERTRobertaGatedClassifier(num_emotions=len(EMOTION_NAMES)).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    
    alpha_weights = torch.tensor([1.0, 1.2, 1.0, 1.5, 4.0, 4.0, 1.5, 1.0]).to(DEVICE)
    criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)
    scaler = GradScaler('cuda')

    for epoch in range(30):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            optimizer.zero_grad()
            with autocast('cuda'):
                logits = model(batch["input_values"].to(DEVICE), batch["input_ids"].to(DEVICE), batch["text_mask"].to(DEVICE))
                loss = criterion(logits, batch["labels"].to(DEVICE))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                with autocast('cuda'):
                    logits = model(batch["input_values"].to(DEVICE), batch["input_ids"].to(DEVICE), batch["text_mask"].to(DEVICE))
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_labels.extend(batch["labels"].numpy())
        
        report = classification_report(all_labels, all_preds, target_names=EMOTION_NAMES, digits=4, zero_division=0)
        with open(f"{OUTPUT_DIR}/report_epoch_{epoch+1}.txt", "w") as f: f.write(report)
        
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTION_NAMES, yticklabels=EMOTION_NAMES)
        plt.title(f'HuBERT+RoBERTa Gating CM - Epoch {epoch+1}')
        plt.xlabel('Predicted Emotion')
        plt.ylabel('True Emotion')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/cm_epoch_{epoch+1}.png')
        plt.close()
        
        print(f"Epoch {epoch+1} complete. Report and CM saved.")

if __name__ == "__main__":
    run_training()