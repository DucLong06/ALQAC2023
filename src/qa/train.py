from transformers import (AutoTokenizer, AutoModel, AutoModelForMaskedLM, RobertaTokenizer,
                          RobertaForSequenceClassification, AdamW, LlamaTokenizer, LlamaForSequenceClassification, GPT2Tokenizer, GPT2LMHeadModel)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train_linear_classifier(train_text, train_labels, test_text):
    model_name = "xlm-roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    class LinearClassifier(nn.Module):
        def __init__(self, input_size, num_classes):
            super(LinearClassifier, self).__init__()
            self.linear = nn.Linear(input_size, num_classes)

        def forward(self, x):
            return self.linear(x)

    linear_classifier = LinearClassifier(input_size=1024, num_classes=2)

    train_encodings = tokenizer(
        train_text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        model_output = model(**train_encodings)

    text_representation = model_output.last_hidden_state[:, 0, :]

    train_labels = torch.tensor(train_labels)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(linear_classifier.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        logits = linear_classifier(text_representation)
        loss = loss_fn(logits, train_labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    test_encoding = tokenizer(test_text, return_tensors="pt",
                              padding=True, truncation=True)
    with torch.no_grad():
        test_output = model(**test_encoding)

    test_representation = test_output.last_hidden_state[:, 0, :]
    test_logits = linear_classifier(test_representation)
    probabilities = torch.softmax(test_logits, dim=1)

    for i, text in enumerate(test_text):
        if probabilities[i][1] > 0.5:
            print(f"True: {text}")
        else:
            print(f"False: {text}")


def train_sequence_classification(train_text, train_labels, test_text):
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
    train_encodings = tokenizer(
        train_text, return_tensors='pt', padding=True, truncation=True)
    train_labels = torch.tensor(train_labels)

    model = RobertaForSequenceClassification.from_pretrained(
        'xlm-roberta-large', num_labels=2)

    train_dataset = TensorDataset(
        train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss}")

    test_encodings = tokenizer(
        test_text, return_tensors='pt', padding=True, truncation=True)

    model.eval()
    with torch.no_grad():
        test_outputs = model(
            input_ids=test_encodings['input_ids'], attention_mask=test_encodings['attention_mask'])
        logits = test_outputs.logits
        probabilities = torch.softmax(logits, dim=1)

    for i, text in enumerate(test_text):
        if probabilities[i][1] > 0.5:
            print(f"True: {text}")
        else:
            print(f"False: {text}")


def train_llm(train_text, train_labels, test_text):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Combine all training text into a single string
    train_text = ' '.join(train_text)

    inputs = tokenizer.encode(train_text, return_tensors="pt")
    outputs = model(inputs, labels=inputs)
    loss = outputs.loss
    loss.backward()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    optimizer.step()
    test_input = tokenizer.encode(test_text[0], return_tensors="pt")
    generated_output = model.generate(
        test_input, max_length=100, num_beams=5, early_stopping=True)
    generated_text = tokenizer.decode(
        generated_output[0], skip_special_tokens=True)
    print(f"Generated Text: {generated_text}")


if __name__ == "__main__":
    train_text = ["Hà nội là thủ đô của Việt Nam phải không?",
                  "Thời gian tập sự của viên chức được quy định trong hợp đồng làm việc là đúng hay sai?",
                  "Thanh niên là người trên 18 tuổi?"]
    train_labels = [1, 0, 1]
    test_text = ["Vinh là thủ đô của Việt Nam?"]

    # train_linear_classifier(train_text, train_labels, test_text)
    train_sequence_classification(train_text, train_labels, test_text)
