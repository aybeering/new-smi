# Requires sentence_transformers>=2.7.0
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

sentences = ['That is a happy person', 'That is a very happy person']

model = SentenceTransformer('/Users/ayang/agent/new-smi/dir', trust_remote_code=True)
embeddings = model.encode(sentences)
print(cos_sim(embeddings[0], embeddings[1]))