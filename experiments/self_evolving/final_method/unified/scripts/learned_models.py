"""Compact Time-LLM and MM-TSFlib ports for the common task format."""
import torch
from torch import nn
import torch.nn.functional as F


class Saved(nn.Module):
    def trainable_state(self):
        return {name: value.detach().cpu() for name, value in self.state_dict().items()
                if name in {n for n, p in self.named_parameters() if p.requires_grad}}

    def load_trainable_state(self, state):
        own = self.state_dict()
        own.update(state); self.load_state_dict(own)


class TimeLLM(Saved):
    def __init__(self, context=512, horizon=256, patch=16, stride=8, dim=64, heads=8):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = AutoModel.from_pretrained("openai-community/gpt2")
        for value in self.llm.parameters(): value.requires_grad = False
        self.context, self.horizon = context, horizon
        self.patch = nn.Conv1d(1, dim, patch, stride=stride)
        self.keys = nn.Linear(self.llm.config.n_embd, dim)
        self.reprogram = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.to_llm = nn.Linear(dim, self.llm.config.n_embd)
        patches = (context - patch) // stride + 1
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(patches * self.llm.config.n_embd, horizon))

    def train(self, mode=True):
        super().train(mode); self.llm.eval(); return self

    def forward(self, data):
        patches = self.patch(data["x"][:, None]).transpose(1, 2)
        words = self.llm.get_input_embeddings()(data["ids"])
        mapped = self.keys(words)
        patches = self.reprogram(patches, mapped, mapped, key_padding_mask=~data["attention"].bool())[0]
        hidden = self.llm(inputs_embeds=torch.cat([words, self.to_llm(patches)], 1),
                          attention_mask=torch.cat([data["attention"],
                                                    torch.ones_like(data["attention"][:, :patches.shape[1]])], 1)).last_hidden_state
        return self.head(hidden[:, -patches.shape[1]:])


class MMTSF(Saved):
    def __init__(self, context=512, horizon=256, dim=128, heads=8):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.text = AutoModel.from_pretrained("google-bert/bert-base-uncased")
        for value in self.text.parameters(): value.requires_grad = False
        self.context, self.horizon = context, horizon
        self.value = nn.Linear(2, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, 0.1, batch_first=True)
        self.numeric = nn.TransformerEncoder(layer, 3)
        self.text_key = nn.Linear(self.text.config.hidden_size, dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.head = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, horizon))

    def train(self, mode=True):
        super().train(mode); self.text.eval(); return self

    def forward(self, data):
        x = data["x"]
        trend = F.avg_pool1d(F.pad(x[:, None], (12, 12), mode="replicate"), 25, stride=1)[:, 0]
        numeric = self.numeric(self.value(torch.stack([x - trend, trend], -1)),
                               src_key_padding_mask=~data["mask"].bool())
        query = (numeric * data["mask"][..., None]).sum(1) / data["mask"].sum(1, keepdim=True).clamp_min(1)
        with torch.no_grad(): text = self.text(data["ids"], attention_mask=data["attention"]).last_hidden_state
        pooled = self.attention(query[:, None], self.text_key(text), self.text_key(text),
                                key_padding_mask=~data["attention"].bool())[0][:, 0]
        return self.head(torch.cat([query, pooled], -1))
