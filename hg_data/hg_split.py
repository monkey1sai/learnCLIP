from datasets import load_dataset
dataset = load_dataset("imdb")


train = load_dataset("imdb", split="train")
test = load_dataset("imdb", split="test")
# 從 train 裡再切出 dev, 0.1=10%
train = train.train_test_split(test_size=0.1)
test10_from_origin_train = train['test']
train90_from_origin_train = train['train']

