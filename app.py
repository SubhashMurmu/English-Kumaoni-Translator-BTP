import gradio as gr
import torch
import faiss
import pickle
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from peft import PeftModel
from sentence_transformers import SentenceTransformer

INDEX_PATH = "faiss_cache.index"
LOOKUP_PATH = "translation_memory.pkl"
BASE_MODEL = "facebook/mbart-large-50-many-to-many-mmt"
LORA_WEIGHTS = "./lora_weights"
SIMILARITY_THRESHOLD = 0.95

index = faiss.read_index(INDEX_PATH)
with open(LOOKUP_PATH, "rb") as f:
    translation_memory = pickle.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

tokenizer = MBart50TokenizerFast.from_pretrained(BASE_MODEL)
base_model = MBartForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

try:
    model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS)
except Exception:
    model = base_model

def translate(text):
    text = text.strip()
    if not text:
        return ""
    
    query_vec = embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    distances, indices = index.search(query_vec, 1)
    
    if distances[0][0] >= SIMILARITY_THRESHOLD and indices[0][0] != -1:
        return f"[Cache Hit: {distances[0][0]:.3f}]\n" + translation_memory[indices[0][0]]
    
    tokenizer.src_lang = "en_XX"
    encoded = tokenizer(text, return_tensors="pt").to(model.device)
    generated = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id.get("hi_IN", tokenizer.eos_token_id),
        max_new_tokens=128
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

demo = gr.Interface(
    fn=translate,
    inputs=gr.Textbox(lines=3, placeholder="Enter English text..."),
    outputs=gr.Textbox(lines=3, label="Kumaoni Translation"),
    title="English to Kumaoni Translator"
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
