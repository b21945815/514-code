import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_router(path="./my_router_model"):
    tokenizer = DistilBertTokenizer.from_pretrained(path)
    model = DistilBertForSequenceClassification.from_pretrained(path)
    model.to(device)
    model.eval()
    return tokenizer, model

def predict_intent(text, loaded_tokenizer, loaded_model):
    """
    Analyze the text:
    0 -> General Chat
    1 -> Database Query
    """
    inputs = loaded_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    
    with torch.no_grad():
        logits = loaded_model(**inputs).logits
    
    # Probability
    probs = F.softmax(logits, dim=-1)
    score, predicted_id = torch.max(probs, dim=1)
    
    label_map = {0: "GENERAL CHAT", 1: "DATABASE QUERY"}
    return label_map[predicted_id.item()], score.item()