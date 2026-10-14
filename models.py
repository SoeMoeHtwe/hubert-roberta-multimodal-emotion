import torch
import torch.nn as nn
from transformers import HubertModel, RobertaModel

class HuBERTRobertaGatedClassifier(nn.Module):
    def __init__(self, num_emotions):
        super().__init__()
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        
        self.audio_gate = nn.Linear(768, 768)
        self.text_gate = nn.Linear(768, 768)
        
        self.classifier = nn.Sequential(
            nn.Linear(768 + 768, 512), 
            nn.ReLU(), 
            nn.Dropout(0.3), 
            nn.Linear(512, num_emotions)
        )
    
    def forward(self, input_values, input_ids, text_mask):
        audio_feats = torch.mean(self.hubert(input_values).last_hidden_state, dim=1)
        text_feats = self.roberta(input_ids=input_ids, attention_mask=text_mask).last_hidden_state[:, 0, :]
        
        g_audio = torch.sigmoid(self.audio_gate(audio_feats))
        g_text = torch.sigmoid(self.text_gate(text_feats))
        
        controlled_audio = audio_feats * g_audio
        controlled_text = text_feats * g_text
        
        combined = torch.cat((controlled_audio, controlled_text), dim=1)
        return self.classifier(combined)