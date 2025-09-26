"""
KV Cache Comparison (GPT-2)

Measures:
- Per-token latency (no_cache vs with_cache)
- Cumulative time
- Optional GPU memory usage
- Estimated KV cache size (float16/float32)

Both paths generate token-by-token for fair timing.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import gc
from typing import Dict, List
import psutil
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt


class KVCacheComparison:
    def __init__(self, model_name: str = "gpt2", kv_dtype: str = "float32"):
        """
        kv_dtype: 'float16' or 'float32' (used only for KV size estimation)
        """
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"[Init] Device: {self.device}")
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device).eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        cfg = self.model.config
        self.num_layers = cfg.n_layer
        self.num_heads = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dtype_bytes = 2 if kv_dtype == "float16" else 4
        print(f"[Model] layers={self.num_layers} heads={self.num_heads} head_dim={self.head_dim} hidden={cfg.n_embd}")

    # ---------- helpers ----------
    def _memory(self) -> Dict[str, float]:
        p = psutil.Process()
        mem = {"cpu_rss_mb": p.memory_info().rss / 1024**2}
        if torch.cuda.is_available():
            mem["gpu_alloc_mb"] = torch.cuda.memory_allocated() / 1024**2
            mem["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
        else:
            mem["gpu_alloc_mb"] = 0.0
            mem["gpu_reserved_mb"] = 0.0
        return mem

    def _sample(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        if temperature != 1.0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)

    def estimate_kv_cache_mb(self, seq_len: int) -> float:
        # 2 (K,V) * layers * heads * seq_len * head_dim * bytes
        total = 2 * self.num_layers * self.num_heads * seq_len * self.head_dim * self.dtype_bytes
        return total / 1024**2

    # ---------- generation (no cache) ----------
    def generate_no_cache(self, prompt: str, max_new_tokens: int) -> Dict:
        inp = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        seq = inp.clone()
        times: List[float] = []
        mem_trace: List[Dict[str, float]] = []

        for t in range(max_new_tokens):
            t0 = time.time()
            with torch.no_grad():
                out = self.model(seq, use_cache=False)
                next_logits = out.logits[:, -1, :]
                next_tok = self._sample(next_logits)
                seq = torch.cat([seq, next_tok], dim=-1)
            dt = time.time() - t0
            times.append(dt)
            mem_trace.append(self._memory())
            print(f"[NoCache] step={t+1:02d}  {dt*1000:7.2f} ms  token='{self.tokenizer.decode(next_tok[0])}'")

        return {
            "mode": "no_cache",
            "tokens": seq,
            "text": self.tokenizer.decode(seq[0], skip_special_tokens=True),
            "step_times": times,
            "total_time": float(np.sum(times)),
            "mem_trace": mem_trace
        }

    # ---------- generation (with cache) ----------
    def generate_with_cache(self, prompt: str, max_new_tokens: int) -> Dict:
        inp = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        seq = inp.clone()
        past = None
        times: List[float] = []
        mem_trace: List[Dict[str, float]] = []

        for t in range(max_new_tokens):
            t0 = time.time()
            with torch.no_grad():
                if t == 0:
                    out = self.model(seq, use_cache=True)
                else:
                    out = self.model(seq[:, -1:], use_cache=True, past_key_values=past)
                past = out.past_key_values
                next_logits = out.logits[:, -1, :]
                next_tok = self._sample(next_logits)
                seq = torch.cat([seq, next_tok], dim=-1)
            dt = time.time() - t0
            times.append(dt)
            mem_trace.append(self._memory())
            print(f"[Cache]   step={t+1:02d}  {dt*1000:7.2f} ms  token='{self.tokenizer.decode(next_tok[0])}'")

        return {
            "mode": "with_cache",
            "tokens": seq,
            "text": self.tokenizer.decode(seq[0], skip_special_tokens=True),
            "step_times": times,
            "total_time": float(np.sum(times)),
            "mem_trace": mem_trace
        }

    # ---------- comparison ----------
    def run(self, prompt: str, max_new_tokens: int, plot: bool = True) -> Dict:
        print("=" * 68)
        print(f"Prompt: {prompt!r}")
        print(f"Generating {max_new_tokens} new tokens")
        print("=" * 68)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print("\n-- WITHOUT KV CACHE --")
        res_no = self.generate_no_cache(prompt, max_new_tokens)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print("\n-- WITH KV CACHE --")
        res_cache = self.generate_with_cache(prompt, max_new_tokens)

        speedup = res_no["total_time"] / res_cache["total_time"] if res_cache["total_time"] > 0 else float("inf")
        seq_len_total = res_cache["tokens"].shape[1]
        kv_est = self.estimate_kv_cache_mb(seq_len_total)

        print("\nSummary")
        print("-" * 68)
        print(f"Total time (no cache):   {res_no['total_time']:.3f} s")
        print(f"Total time (with cache): {res_cache['total_time']:.3f} s")
        print(f"Speedup factor:          {speedup:.2f}x")
        print(f"Estimated KV cache size: {kv_est:.2f} MB (sequence length {seq_len_total})")

        if plot:
            self._plot(res_no, res_cache, kv_est)

        return {
            "no_cache": res_no,
            "with_cache": res_cache,
            "speedup": speedup,
            "kv_cache_est_mb": kv_est
        }

    # ---------- plotting ----------
    def _plot(self, res_no: Dict, res_cache: Dict, kv_est_mb: float):
        steps = np.arange(1, len(res_no["step_times"]) + 1)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("KV Cache Performance Comparison", fontsize=15, fontweight="bold")

        # Per-token latency
        axes[0, 0].plot(steps, np.array(res_no["step_times"]) * 1000, "r-o", label="No Cache")
        axes[0, 0].plot(steps, np.array(res_cache["step_times"]) * 1000, "g-o", label="With Cache")
        axes[0, 0].set_title("Per-token Latency (ms)")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("ms")
        axes[0, 0].grid(alpha=0.3)
        axes[0, 0].legend()

        # Cumulative time
        axes[0, 1].plot(steps, np.cumsum(res_no["step_times"]), "r-", lw=2, label="No Cache")
        axes[0, 1].plot(steps, np.cumsum(res_cache["step_times"]), "g-", lw=2, label="With Cache")
        axes[0, 1].set_title("Cumulative Time (s)")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Seconds")
        axes[0, 1].grid(alpha=0.3)
        axes[0, 1].legend()

        # GPU memory (if available)
        if torch.cuda.is_available():
            mem_no = [m["gpu_alloc_mb"] for m in res_no["mem_trace"]]
            mem_ca = [m["gpu_alloc_mb"] for m in res_cache["mem_trace"]]
            axes[1, 0].plot(steps, mem_no, "r-", label="No Cache")
            axes[1, 0].plot(steps, mem_ca, "g-", label="With Cache")
            axes[1, 0].set_title("GPU Allocated Memory (MB)")
            axes[1, 0].set_xlabel("Step")
            axes[1, 0].set_ylabel("MB")
            axes[1, 0].grid(alpha=0.3)
            axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, "GPU not available", ha="center", va="center")
            axes[1, 0].set_axis_off()

        # Summary bars
        labels = ["Total Time (s)", "Avg Time (ms/token)"]
        no_vals = [res_no["total_time"], np.mean(res_no["step_times"]) * 1000]
        ca_vals = [res_cache["total_time"], np.mean(res_cache["step_times"]) * 1000]
        x = np.arange(len(labels))
        w = 0.35
        axes[1, 1].bar(x - w / 2, no_vals, w, color="red", alpha=0.7, label="No Cache")
        axes[1, 1].bar(x + w / 2, ca_vals, w, color="green", alpha=0.7, label="With Cache")
        for idx, v in enumerate(no_vals):
            axes[1, 1].text(x[idx] - w / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        for idx, v in enumerate(ca_vals):
            axes[1, 1].text(x[idx] + w / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(labels)
        axes[1, 1].set_title(f"Summary (KV Cache ≈ {kv_est_mb:.2f} MB)")
        axes[1, 1].grid(axis="y", alpha=0.3)
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

    # ---------- explanation ----------
    def explain(self):
        print(
            "KV Cache Concept:\n"
            "- No cache: each decoding step reprocesses entire sequence (O(N^2)).\n"
            "- With cache: reuse past K,V; process only the new token (O(N)).\n"
            "- Trade-off: extra memory ~ 2 * L * H * T * D * bytes."
        )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    demo = KVCacheComparison(model_name="gpt2", kv_dtype="float32")
    demo.explain()
    prompt_text = "The future of AI is"
    result = demo.run(prompt=prompt_text, max_new_tokens=20, plot=True)
    print("\n--- Generated (No Cache) ---")
    print(result["no_cache"]["text"])
    print("\n--- Generated (With Cache) ---")
    print(result["with_cache"]["text"])