# llm-basics
## Overview
Author: Paul Foran
Date: 5th Nov 2024
Purpose: Showcasing how LLMs are Generated, leaveragin basic python examples

I'll break down the basics of Large Language Models (LLMs) into steps and provide some example Python code to illustrate the principles. Keep in mind that this is a simplified explanation, as actual LLMs are much more complex and require significant computational resources.

## Arch Overview
``` mermaid
graph TD
    A([Start]) --> B
    B[Data Preprocessing]:::blue --> C
    C[Model Creation]:::green --> D
    D[Training]:::orange --> E
    E[Text Generation]:::purple --> F([End])

    B --> |Tokenize & Encode| B1([Prepare Data]):::blue
    C --> |Define Architecture| C1([Build Model]):::green
    D --> |Backpropagation| D1([Optimize]):::orange
    E --> |Use Model| E1([Generate]):::purple

    classDef blue fill:#2196F3,stroke:#1565C0,color:white;
    classDef green fill:#4CAF50,stroke:#2E7D32,color:white;
    classDef orange fill:#FF9800,stroke:#EF6C00,color:white;
    classDef purple fill:#9C27B0,stroke:#6A1B9A,color:white;
    classDef default fill:#ECEFF1,stroke:#90A4AE,color:#37474F;
```

## Step 1: Data Collection and Preprocessing
LLMs are trained on vast amounts of text data. The first step is to collect and preprocess this data.

``` python
import nltk
from nltk.tokenize import word_tokenize

# Sample text data
text_data = [
    "The quick brown fox jumps over the lazy dog.",
    "Language models are trained on large datasets.",
    "Python is a popular programming language."
]

# Tokenize the text
tokenized_data = [word_tokenize(sentence.lower()) for sentence in text_data]

print(tokenized_data)
```

## Step 2: Vocabulary Creation
Create a vocabulary of unique words in the dataset.
``` python
# Create vocabulary
vocab = set(word for sentence in tokenized_data for word in sentence)
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

print(word_to_idx)
```

## Step 3: Data Encoding
Convert the tokenized text into numerical sequences.
``` python
import numpy as np

# Encode the sentences
encoded_data = [[word_to_idx[word] for word in sentence] for sentence in tokenized_data]

print(encoded_data)
```

## Step 4: Model Architecture
Define the model architecture. Here's a simple example using a basic recurrent neural network (RNN).
``` python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        return self.fc(output)

# Model parameters
vocab_size = len(vocab)
embedding_dim = 50
hidden_dim = 100

model = SimpleRNN(vocab_size, embedding_dim, hidden_dim)
```

## Step 5: Training Data Preparation
Prepare the training data by creating input-output pairs.

``` python
def prepare_sequence(seq, word_to_idx):
    return torch.tensor([word_to_idx[w] for w in seq], dtype=torch.long)

# Prepare training data
X = [prepare_sequence(seq[:-1], word_to_idx) for seq in tokenized_data]
y = [prepare_sequence(seq[1:], word_to_idx) for seq in tokenized_data]
```

## Step 6: Training Loop
Train the model using the prepared data.

``` python
import torch.optim as optim

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

num_epochs = 100

for epoch in range(num_epochs):
    total_loss = 0
    for sentence, target in zip(X, y):
        model.zero_grad()
        output = model(sentence.unsqueeze(0))
        loss = loss_function(output.squeeze(0), target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss}")
```

## Step 7: Text Generation
Use the trained model to generate text.

``` python
def generate_text(model, start_sequence, max_length=20):
    model.eval()
    current_sequence = start_sequence
    generated_sequence = start_sequence.copy()

    with torch.no_grad():
        for _ in range(max_length):
            input_tensor = prepare_sequence(current_sequence, word_to_idx).unsqueeze(0)
            output = model(input_tensor)
            next_word_idx = output[0, -1, :].argmax().item()
            generated_sequence.append(idx_to_word[next_word_idx])
            current_sequence = current_sequence[1:] + [idx_to_word[next_word_idx]]

    return " ".join(generated_sequence)

# Generate text
start_sequence = ["the", "quick"]
generated_text = generate_text(model, start_sequence)
print(generated_text)
```

## Summary
This example provides a basic illustration of how LLMs work. Real-world LLMs like GPT-3 or BERT are much more complex, involving:

- Transformer architectures instead of simple RNNs
- Attention mechanisms
- Much larger datasets and vocabularies
- More sophisticated training techniques (e.g., unsupervised pretraining, fine-tuning)
- Advanced tokenization methods (e.g., subword tokenization)
- Parallel processing and distributed training
- 
The principles, however, remain similar: they learn to predict the next word (or token) in a sequence based on the context provided by the previous words.