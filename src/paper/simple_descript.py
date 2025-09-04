import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

# 簡化影像描述的函數
def simplify_description(description):
    tokens = word_tokenize(description)
    pos_tags = nltk.pos_tag(tokens)
    simplified_tokens = []
    
    for word, pos in pos_tags:
        if pos.startswith('NN'):  # 名詞替換為上位詞
            synsets = wn.synsets(word, pos=wn.NOUN)
            if synsets:
                hypernym = synsets[0].hypernyms()
                if hypernym:
                    simplified_tokens.append(hypernym[0].lemmas()[0].name())
                else:
                    simplified_tokens.append(word)
            else:
                simplified_tokens.append(word)
        elif pos.startswith('VB'):  # 動詞簡化
            simplified_tokens.append(word)  # 可進一步處理動詞
        else:
            simplified_tokens.append(word)
    
    return ' '.join(simplified_tokens)

# 示例描述
descriptions = [
    "A man is running on the beach",
    "A person is jogging near the ocean"
]

# 簡化描述
simplified_descriptions = [simplify_description(desc) for desc in descriptions]

print(simplified_descriptions)
