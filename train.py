import math
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


# ---------- 1. 配置 ----------

CFG = {
    "context": 128,    # 模型一次最多看多少个字符
    "width": 128,      # 每个字符的向量维度
    "layers": 4,
    "heads": 4,
    "dropout": 0.1,
}

BATCH = 8
STEPS = 2000
LR = 3e-4
EVAL_EVERY = 200
EVAL_BATCHES = 10

torch.manual_seed(1234)

if not torch.backends.mps.is_available():
    raise RuntimeError("MPS 不可用，请先检查 ARM64 Python、PyTorch 和 macOS。")

DEVICE = torch.device("mps")


# ---------- 2. 因果多头自注意力 ----------

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        width = cfg["width"]
        self.heads = cfg["heads"]
        assert width % self.heads == 0

        self.head_dim = width // self.heads

        # 一次线性映射，同时得到 Q、K、V
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.dropout = nn.Dropout(cfg["dropout"])

        # True 表示允许看：当前位置及之前的位置
        mask = torch.tril(
            torch.ones(cfg["context"], cfg["context"], dtype=torch.bool)
        )
        self.register_buffer("mask", mask)

    def forward(self, x):
        batch, length, width = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # [B, T, C] -> [B, H, T, D]
        q = q.reshape(batch, length, self.heads, self.head_dim)
        k = k.reshape(batch, length, self.heads, self.head_dim)
        v = v.reshape(batch, length, self.heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 每个位置与其他位置的相关程度
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 屏蔽未来：训练时也不能偷看后面的答案
        scores = scores.masked_fill(
            ~self.mask[:length, :length], float("-inf")
        )

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # 根据注意力权重汇总信息
        out = weights @ v

        # 拼回多个头：[B, H, T, D] -> [B, T, C]
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(batch, length, width)

        return self.dropout(self.proj(out))


# ---------- 3. Transformer block ----------

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width = cfg["width"]

        self.norm1 = nn.LayerNorm(width)
        self.attn = Attention(cfg)
        self.norm2 = nn.LayerNorm(width)

        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
            nn.Dropout(cfg["dropout"]),
        )

    def forward(self, x):
        # Pre-LayerNorm + 残差连接
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------- 4. GPT ----------

class GPT(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.context = cfg["context"]
        width = cfg["width"]

        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(self.context, width)

        self.blocks = nn.Sequential(
            *[Block(cfg) for _ in range(cfg["layers"])]
        )
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size, bias=False)

        self.apply(self.initialize)

        # 输入字符向量和输出分类器共享权重
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, ids, targets=None):
        batch, length = ids.shape
        assert length <= self.context

        positions = torch.arange(length, device=ids.device)

        x = self.token(ids) + self.position(positions)
        x = self.blocks(x)
        logits = self.head(self.norm(x))   # [B, T, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )

        return logits, loss


# ---------- 5. 随机抽取训练片段 ----------

def get_batch(data, context, generator=None):
    starts = torch.randint(
        len(data) - context,
        (BATCH,),
        generator=generator,
    ).tolist()

    x = torch.stack([data[i:i + context] for i in starts])
    y = torch.stack([data[i + 1:i + context + 1] for i in starts])

    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def evaluate(model, train_data, val_data):
    model.eval()

    # 固定评估片段，便于比较不同训练阶段
    generator = torch.Generator().manual_seed(2026)
    result = {}

    for name, data in [("train", train_data), ("val", val_data)]:
        losses = []

        for _ in range(EVAL_BATCHES):
            x, y = get_batch(data, model.context, generator)
            _, loss = model(x, y)
            losses.append(loss.item())

        result[name] = sum(losses) / len(losses)

    model.train()
    return result


# ---------- 6. 自回归生成 ----------

@torch.no_grad()
def generate(model, ids, count=400, temperature=0.8):
    model.eval()

    for _ in range(count):
        # 超出上下文时，只保留最近的字符
        logits, _ = model(ids[:, -model.context:])

        # 最后一个位置负责预测下一个字符
        probs = F.softmax(logits[:, -1] / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)

    return ids


# ---------- 7. 训练 ----------

def train():
    text = Path("input.txt").read_text(encoding="utf-8")

    # 前 95% 训练，后 5% 验证
    cut = int(len(text) * 0.95)

    # 词表只从训练部分建立；0 留给未知字符
    chars = ["<UNK>"] + sorted(set(text[:cut]))
    stoi = {ch: i for i, ch in enumerate(chars)}

    data = torch.tensor(
        [stoi.get(ch, 0) for ch in text],
        dtype=torch.long,
    )
    train_data, val_data = data[:cut], data[cut:]
    del text

    if min(len(train_data), len(val_data)) <= CFG["context"]:
        raise ValueError("文本太短，训练集和验证集都必须长于上下文。")

    model = GPT(len(chars), CFG).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.01
    )

    params = sum(p.numel() for p in model.parameters())
    print(f"device={DEVICE}, vocab={len(chars)}, params={params:,}")
    print(f"随机均匀预测的参考 loss: {math.log(len(chars)):.3f}")

    best_val = float("inf")
    interval_start = time.perf_counter()

    for step in range(1, STEPS + 1):
        # 前 100 步 warmup，此后余弦下降到初始学习率的 10%
        warmup = min(100, max(1, STEPS // 10))

        if step <= warmup:
            lr = LR * step / warmup
        else:
            progress = (step - warmup) / (STEPS - warmup)
            lr = LR * (
                0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
            )

        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, CFG["context"])

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()

        # 限制梯度范数，减少不稳定更新
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % EVAL_EVERY == 0 or step == STEPS:
            torch.mps.synchronize()
            elapsed = time.perf_counter() - interval_start

            metrics = evaluate(model, train_data, val_data)

            print(
                f"step={step:5d} "
                f"train={metrics['train']:.3f} "
                f"val={metrics['val']:.3f} "
                f"interval_seconds={elapsed:.1f}"
            )

            if metrics["val"] < best_val:
                best_val = metrics["val"]

                # 保存 CPU 权重，加载时再搬到 GPU
                weights = {
                    k: v.detach().cpu()
                    for k, v in model.state_dict().items()
                }
                torch.save(
                    {
                        "model": weights,
                        "cfg": CFG,
                        "chars": chars,
                        "step": step,
                    },
                    "best.pt",
                )

            interval_start = time.perf_counter()


def sample():
    checkpoint = torch.load(
        "best.pt", map_location="cpu", weights_only=True
    )
    chars = checkpoint["chars"]
    stoi = {ch: i for i, ch in enumerate(chars)}

    model = GPT(len(chars), checkpoint["cfg"])
    model.load_state_dict(checkpoint["model"])
    model = model.to(DEVICE)

    prompt = sys.argv[2] if len(sys.argv) > 2 else "\n"
    if not prompt:
        prompt = "\n"

    ids = torch.tensor(
        [[stoi.get(ch, 0) for ch in prompt]],
        dtype=torch.long,
        device=DEVICE,
    )

    output = generate(model, ids)
    print("".join(chars[i] for i in output[0].cpu().tolist()))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--sample":
        sample()
    else:
        train()