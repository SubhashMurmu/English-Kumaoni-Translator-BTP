"""Model presets and the Kumaoni language token (initialised from Hindi's)."""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PRESETS = {
    "mbart": dict(name="facebook/mbart-large-50-many-to-many-mmt",
                  src="en_XX", hi="hi_IN", kmn="kfy_IN"),   # kfy = ISO 639-3 code for Kumaoni
    "nllb": dict(name="facebook/nllb-200-distilled-600M",
                 src="eng_Latn", hi="hin_Deva", kmn="kfy_Deva"),
}


def load_with_kmn_token(path_or_name, preset):
    """Load model+tokenizer. If the Kumaoni token is missing, add it and copy Hindi's embedding.

    Why: the old code trained with the *Hindi* token as target, so the decoder could not tell
    Kumaoni from Hindi and real Hindi data could not be mixed in. A separate token that STARTS
    as Hindi gives the transfer benefit and lets the two languages diverge during training.
    (The row must be trainable, so use full fine-tuning, not frozen-embedding LoRA.)
    """
    p = PRESETS[preset]
    tok = AutoTokenizer.from_pretrained(path_or_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(path_or_name).float()
    if tok.convert_tokens_to_ids(p["kmn"]) in (None, tok.unk_token_id):
        tok.add_tokens([p["kmn"]], special_tokens=True)
        if len(tok) > model.get_input_embeddings().num_embeddings:
            model.resize_token_embeddings(len(tok))
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            emb[tok.convert_tokens_to_ids(p["kmn"])] = emb[tok.convert_tokens_to_ids(p["hi"])].clone()
    tok.src_lang = p["src"]
    ids = {k: tok.convert_tokens_to_ids(p[k]) for k in ("src", "hi", "kmn")}
    assert len(set(ids.values())) == 3 and tok.unk_token_id not in ids.values(), ids
    return tok, model, ids
