from datasets import load_dataset
dataset = load_dataset("imdb")


train = load_dataset("imdb", split="train")
test = load_dataset("imdb", split="test")

# 從 train 裡再切出 10% 當作驗證集
train = train.train_test_split(test_size=0.1)

# 重新定義 train 和 valid
train, valid = train["train"], train["test"]
print(train)
print(valid)
