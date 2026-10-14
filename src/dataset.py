import os
import librosa
import pandas as pd
import torch
from torch.utils.data import Dataset

EMOTION_MAP = {'A': 0, 'S': 1, 'H': 2, 'U': 3, 'F': 4, 'D': 5, 'C': 6, 'N': 7}

class MultimodalPodcastDataset(Dataset):
    def __init__(self, csv_file, split_set="Train", limit=None, audio_dir=""):
        df = pd.read_csv(csv_file).rename(columns={'EmoClass': 'Emotion', 'Split_Set': 'Split'})
        self.df = df[df['Split'] == split_set].reset_index(drop=True)
        self.audio_dir = audio_dir
        if limit: self.df = self.df.head(limit)
        
    def __len__(self): 
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.audio_dir, row['FileName'])
        speech, _ = librosa.load(file_path, sr=16000, duration=11)
        return {"speech": speech, "text": str(row['Transcription']), "label": EMOTION_MAP.get(row['Emotion'], 0)}

def get_collate_fn(audio_proc, text_tok):
    def collate_fn(batch):
        audio_inputs = audio_proc([b["speech"] for b in batch], sampling_rate=16000, padding=True, return_tensors="pt")
        text_inputs = text_tok([b["text"] for b in batch], padding=True, truncation=True, max_length=128, return_tensors="pt")
        return {
            "input_values": audio_inputs.input_values, 
            "input_ids": text_inputs.input_ids, 
            "text_mask": text_inputs.attention_mask, 
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long)
        }
    return collate_fn
