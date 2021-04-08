from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pandas as pd

model_name = 'distilbert-base-nli-mean-tokens'
model = SentenceTransformer(model_name)
word_list = [
  "computer science",
  "mathematics",
  "artificial intelligence",
  "algorithm",
  "discrete mathematics",
  "combinatorics",
  "theoretical computer science",
  "computer network",
  "computer vision",
  "pattern recognition" 
]

kw_emb = model.encode(word_list)
word_similarity = cosine_similarity(kw_emb)
d = defaultdict(lambda:defaultdict(), {})
for idx1, word1 in enumerate(word_list):
    for idx2, word2 in enumerate(word_list):
        d[word1][word2] = word_similarity[idx1][idx2]
df = pd.DataFrame(d)
print(df)