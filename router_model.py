import torch
import sys
import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("--- System Information ---")
print(f"Python Version: {sys.version.split()[0]}")
print(f"Used Device: {device}")

if device.type == 'cuda':
    print(f"Graphic Card: {torch.cuda.get_device_name(0)}")
    print("GPU is ready")
else:
    print("CPU is ready")

# DATA SET (SPECIFY WITH CHOSEN DATASET)
data = [
    # --- LABEL 0: GENERAL CHAT (50 Example) ---
    {"text": "Hello, how are you?", "label": 0},
    {"text": "Good morning!", "label": 0},
    {"text": "Tell me a joke.", "label": 0},
    {"text": "Who are you?", "label": 0},
    {"text": "Are you a robot?", "label": 0},
    {"text": "I'm feeling happy today.", "label": 0},
    {"text": "Thanks for your help.", "label": 0},
    {"text": "See you later.", "label": 0},
    {"text": "Can you help me with something else?", "label": 0},
    {"text": "Who created you?", "label": 0},
    {"text": "Nice to meet you.", "label": 0},
    {"text": "What's up?", "label": 0},
    {"text": "Have a great day.", "label": 0},
    {"text": "I am very tired.", "label": 0},
    {"text": "What is the meaning of life?", "label": 0},
    {"text": "Do you like pizza?", "label": 0},
    {"text": "What is the weather like?", "label": 0},
    {"text": "Can you sing a song?", "label": 0},
    {"text": "You are very smart.", "label": 0},
    {"text": "I don't understand.", "label": 0},
    {"text": "Please explain that again.", "label": 0},
    {"text": "I am bored.", "label": 0},
    {"text": "What is your favorite color?", "label": 0},
    {"text": "Good night.", "label": 0},
    {"text": "Talk to you later.", "label": 0},
    {"text": "That is funny.", "label": 0},
    {"text": "I need advice on life.", "label": 0},
    {"text": "Do you sleep?", "label": 0},
    {"text": "What time is it?", "label": 0},
    {"text": "Is it raining outside?", "label": 0},
    {"text": "My name is John.", "label": 0},
    {"text": "Do you have a family?", "label": 0},
    {"text": "I want to learn Python.", "label": 0},
    {"text": "This is amazing.", "label": 0},
    {"text": "You are helpful.", "label": 0},
    {"text": "What language do you speak?", "label": 0},
    {"text": "Can we be friends?", "label": 0},
    {"text": "I am just testing you.", "label": 0},
    {"text": "Write a poem for me.", "label": 0},
    {"text": "Do you know Siri?", "label": 0},
    {"text": "I had a bad day.", "label": 0},
    {"text": "Let's change the topic.", "label": 0},
    {"text": "Do you like sports?", "label": 0},
    {"text": "Where do you live?", "label": 0},
    {"text": "Goodbye.", "label": 0},
    {"text": "Hi there!", "label": 0},
    {"text": "Cool stuff.", "label": 0},
    {"text": "Explain quantum physics simply.", "label": 0},
    {"text": "I'm hungry.", "label": 0},
    {"text": "What date is it today?", "label": 0},

    # --- LABEL 1: DATABASE QUERY (50 Example) ---
    {"text": "Show me the sales data for 2024.", "label": 1},
    {"text": "List all users who signed up yesterday.", "label": 1},
    {"text": "What is the total revenue for Q3?", "label": 1},
    {"text": "Find the email of the customer named John Doe.", "label": 1},
    {"text": "How many products are currently in stock?", "label": 1},
    {"text": "Get the latest report from the finance table.", "label": 1},
    {"text": "Select top 10 performing employees.", "label": 1},
    {"text": "What was the average order value last month?", "label": 1},
    {"text": "Retrieve the transaction history for ID 554.", "label": 1},
    {"text": "Count the number of active subscriptions.", "label": 1},
    {"text": "Who bought the most items?", "label": 1},
    {"text": "Show me the inventory status.", "label": 1},
    {"text": "Filter results by city equals New York.", "label": 1},
    {"text": "Delete the user with ID 99.", "label": 1},
    {"text": "Update the address for client X.", "label": 1},
    {"text": "Insert a new record into the logs.", "label": 1},
    {"text": "Which product has the lowest stock?", "label": 1},
    {"text": "List employees hired before 2020.", "label": 1},
    {"text": "Calculate the total profit margin.", "label": 1},
    {"text": "Search for invoices created today.", "label": 1},
    {"text": "Display the list of all admins.", "label": 1},
    {"text": "How many tickets are open in Jira?", "label": 1},
    {"text": "Get me the phone number of the CEO.", "label": 1},
    {"text": "Sort the customers by spending amount.", "label": 1},
    {"text": "Check if item 123 is available.", "label": 1},
    {"text": "What is the status of order #4455?", "label": 1},
    {"text": "Give me a list of pending payments.", "label": 1},
    {"text": "Total count of visitors this week.", "label": 1},
    {"text": "Find users with age greater than 30.", "label": 1},
    {"text": "Show details for product category Electronics.", "label": 1},
    {"text": "Export the monthly sales report.", "label": 1},
    {"text": "Who is the manager of the IT department?", "label": 1},
    {"text": "When was the last login for user admin?", "label": 1},
    {"text": "Count all rows in the feedback table.", "label": 1},
    {"text": "What is the sum of all expenses?", "label": 1},
    {"text": "List all cancelled orders.", "label": 1},
    {"text": "Find the most recent transaction.", "label": 1},
    {"text": "Group sales by region.", "label": 1},
    {"text": "What is the maximum salary in the company?", "label": 1},
    {"text": "Show me the database schema.", "label": 1},
    {"text": "Are there any duplicate entries?", "label": 1},
    {"text": "Fetch the data for the last 7 days.", "label": 1},
    {"text": "Who has the highest bonus?", "label": 1},
    {"text": "List all suppliers in Germany.", "label": 1},
    {"text": "What is the churn rate for this month?", "label": 1},
    {"text": "Show me the growth percentage.", "label": 1},
    {"text": "Select distinct names from the list.", "label": 1},
    {"text": "How many returns did we process?", "label": 1},
    {"text": "Find the order linked to this tracking number.", "label": 1},
    {"text": "Get the profile picture URL for user 5.", "label": 1},
]
#TODO add the data from train test split
# DATA
df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)

# 2. MODEL VE TOKENIZER
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device) 

# 3. TRAINING (FINE-TUNING)
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=1,
    save_strategy="no" 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

print("\nTraining Start")
trainer.train()

# 4. SAVE
save_path = "./my_router_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"\nModel saved to '{save_path}'")

# Get the model from save
path = "./my_router_model"
loaded_tokenizer = DistilBertTokenizer.from_pretrained(path)
loaded_model = DistilBertForSequenceClassification.from_pretrained(path)
loaded_model.to(device)

def predict_intent(text):
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

# Test 
#TODO use real test data as well
test_sentences = [
    "Hey, what's up?",
    "Select * from users where age > 25",
    "Show me the inventory list",
    "I am really tired today",
    "How many items did we sell yesterday?",
    "Tell me a story about space"
]

print(f"{'Input':<40} | {'Prediction':<15} | {'Trust'}")
print("-" * 70)

for sentence in test_sentences:
    label, conf = predict_intent(sentence)
    print(f"{sentence:<40} | {label:<15} | %{conf*100:.1f}")