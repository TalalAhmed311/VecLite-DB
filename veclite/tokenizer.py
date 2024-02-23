from transformers import AutoTokenizer, AutoModel

class Tokenizer:

    def __init__(self):
        self._model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name)

    def create_embedding(self,text):
        inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self._model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embeddings.flatten().tolist()