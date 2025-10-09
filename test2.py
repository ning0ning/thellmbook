from sentence_transformers import SentenceTransformer
import gensim.downloader as api

#load model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

#Convert text to text embeddings
vector = model.encode("Best movie ever!")

print(vector[:10])