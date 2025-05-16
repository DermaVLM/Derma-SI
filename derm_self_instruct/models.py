from sentence_transformers import SentenceTransformer

from derm_self_instruct.config import MODEL_ID_SENTENCE


# Initialize the sentence embedding model
def get_embedding_model():
    return SentenceTransformer(MODEL_ID_SENTENCE)
