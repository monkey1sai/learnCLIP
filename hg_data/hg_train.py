from datasets import load_dataset
raw = load_dataset("imdb")

# 只取 5% 當作示範
train_small = load_dataset("imdb", split="train[:5%]")
test_small = load_dataset("imdb", split="test[:5%]")

# 合併成一個 DatasetDict
from datasets import DatasetDict
data = DatasetDict({"train": train_small, "test": test_small})


# 文字斷詞, "bert-base-uncased" 代表小寫英文 BERT 模型, 可換成其他模型如中文的 "bert-base-chinese", 繁體中文的 "hfl/chinese-bert-wwm-ext"
# 參考 https://huggingface.co/transformers/pretrained_models.html
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("bert-base-uncased")


def tokenize(batch):
    return tok(batch["text"], truncation=True, padding="max_length", max_length=256)


# 斷詞並轉成 PyTorch tensors
tokenized = data.map(tokenize, batched=True, remove_columns=["text"])
tokenized.set_format("torch")

# 建立分類模型, 2 類別 (正評/負評)
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 訓練參數
# 參考 https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments
args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="steps",
    logging_steps=100,
    num_train_epochs=1,
    save_steps=500,
    seed=42
)

# 評估指標
# 參考 https://huggingface.co/transformers/main_classes/trainer.html#trainer
import evaluate
acc = evaluate.load("accuracy")
# 計算準確率, 取機率最高的類別當作預測結果, p.label_ids 是正確答案, preds 是預測結果, p.predictions 是模型輸出的機率, 格式是 (num_samples, num_labels)
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return acc.compute(predictions=preds, references=p.label_ids)


# 開始訓練
trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["test"], compute_metrics=compute_metrics)
trainer.train()
trainer.evaluate() # 評估

# 儲存斷詞後的資料集, 之後可以直接用 load_from_disk() 載入
tokenized.save_to_disk("proc/imdb_tok")
trainer.predict(tokenized["test"])

# 轉成 parquet 格式, 可用 pandas.read_parquet() 載入
from datasets import load_from_disk
re_tok = load_from_disk("proc/imdb_tok")


# 此次訓練資料儲存成 parquet 檔案
re_tok["train"].to_parquet("imdb_train.parquet")
re_tok["test"].to_parquet("imdb_test.parquet")