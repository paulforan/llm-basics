# llm-basics
## Overview
Author: Paul Foran
Date: 5th Nov 2024
Purpose: Showcasing how LLMs are Generated, leaveragin basic python examples

I'll break down the basics of Large Language Models (LLMs) into steps and provide some example Python code to illustrate the principles. Keep in mind that this is a simplified explanation, as actual LLMs are much more complex and require significant computational resources.

## Arch Overview
``` mermaid
graph TD
    A[Start] --> B[Step 1: Data Collection and Preprocessing]
    B --> C[Step 2: Vocabulary Creation]
    C --> D[Step 3: Data Encoding]
    D --> E[Step 4: Model Architecture]
    E --> F[Step 5: Training Data Preparation]
    F --> G[Step 6: Training Loop]
    G --> H[Step 7: Text Generation]
    H --> I[End]

    B --> B1[Tokenize text data]
    C --> C1[Create word-to-index mapping]
    D --> D1[Convert tokens to numerical sequences]
    E --> E1[Define neural network structure]
    F --> F1[Create input-output pairs]
    G --> G1[Train model using backpropagation]
    H --> H1[Use trained model to generate text]
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