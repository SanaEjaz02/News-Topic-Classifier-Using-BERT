📘 Text Classification with BERT on AG News

This repository contains code, model, and resources for **fine-tuning BERT on the AG News dataset** using [Hugging Face Transformers](https://huggingface.co/transformers/).
The goal is to classify news articles into one of **4 categories**:

1. World 🌍
2. Sports 🏅
3. Business 💼
4. Science/Technology 🔬

🚀 Project Overview

We fine-tuned BERT (bert-base-uncased) on the AG News dataset for a text classification task.
This project demonstrates the end-to-end workflow of NLP model fine-tuning:

1. Load Datase → AG News dataset (train + test splits).
2. Preprocess Data → Tokenization using BERT tokenizer.
3. Prepare Dataloaders → Batch data with padding/collation.
4.Fine-Tune BERT → Train on a subset for efficiency.
5. Evaluate Model → Compute accuracy on test set.
6. Save Model → Export weights + tokenizer for later inference.

📦 Dependencies

Main Python libraries used:

[transformers](https://pypi.org/project/transformers/)
[datasets](https://pypi.org/project/datasets/)
[evaluate](https://pypi.org/project/evaluate/)
[torch](https://pytorch.org/)
[tqdm](https://pypi.org/project/tqdm/)

You can install them directly with:

```bash
pip install torch transformers datasets evaluate tqdm
```

🏋️ Training Procedure

We fine-tuned BERT with the following setup:

* Batch size: 16
* Max sequence length: 128 tokens
* Optimizer: AdamW
* Learning rate scheduler: Linear
* Epochs: 1 (light run for demo; can increase)
* Device: GPU (Tesla T4 in Google Colab)

Training code snippet:

```python
for batch in train_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    batch["labels"] = batch.pop("label")

    outputs = model(**batch)
    loss = outputs.loss

    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()


📊 Results

After fine-tuning on a small subset of AG News, we achieved:

* Accuracy: \~92% ✅

This demonstrates that BERT generalizes well even with limited training.


💾 Saving & Loading the Model

We saved the trained model and tokenizer:

```python
model.save_pretrained("./bert_agnews_model")
tokenizer.save_pretrained("./bert_agnews_model")
```

To load it later:

```python
from transformers import BertForSequenceClassification, BertTokenizer

model = BertForSequenceClassification.from_pretrained("./bert_agnews_model")
tokenizer = BertTokenizer.from_pretrained("./bert_agnews_model")
```

---

🔍 Inference

Example of running predictions on your own sentence:

```python
text = "NASA launches a new satellite to study climate change."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
predicted_class = outputs.logits.argmax(dim=-1).item()

labels = ["World", "Sports", "Business", "Sci/Tech"]
print("Predicted label:", labels[predicted_class])
```

Output:

```
Predicted label: Sci/Tech
```


📌 Next Steps

* Train for more epochs on the **full dataset**.
* Try different models (`distilbert-base-uncased`, `roberta-base`, etc.).
* Push the model to [🤗 Hugging Face Hub](https://huggingface.co/) for easy sharing.


