import os

import gradio as gr

from kmt.infer import TranslationMemory, Translator, load_hindi_only

MODEL_DIR = os.environ.get("KMT_MODEL", "runs/B/final")
PRESET = os.environ.get("KMT_PRESET", "mbart")
ALPHA = float(os.environ.get("KMT_ALPHA", "1.0"))   # tune on dev; 0 disables leak penalty

translator = Translator(MODEL_DIR, PRESET, load_hindi_only("synth/hindi_only.txt"))  # raises if wrong dir
memory = (TranslationMemory("faiss_cache.index", "translation_memory.pkl")
          if os.path.exists("faiss_cache.index") else None)


def translate(text, use_memory):
    text = (text or "").strip()
    if not text:
        return "", ""
    if use_memory and memory is not None:
        hit = memory.lookup(text)
        if hit:
            return hit[0], f"translation memory (similarity {hit[1]:.2f})"
    return translator.translate([text], alpha=ALPHA)[0], "neural model"


demo = gr.Interface(
    fn=translate,
    inputs=[gr.Textbox(lines=3, placeholder="Enter English text..."),
            gr.Checkbox(value=True, label="Use translation memory")],
    outputs=[gr.Textbox(lines=3, label="Kumaoni translation"), gr.Textbox(label="Source")],
    title="English to Kumaoni Translator",
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
