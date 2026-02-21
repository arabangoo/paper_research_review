# LoRA: Low-Rank Adaptation of Large Language Models - 논문 리뷰

> **대규모 언어 모델의 효율적 파인튜닝: 학습 가능한 파라미터 수를 10,000배 줄이면서 성능은 동등하게**

[![arXiv](https://img.shields.io/badge/arXiv-2106.09685-b31b1b.svg)](https://arxiv.org/abs/2106.09685)
[![Publication Date](https://img.shields.io/badge/Published-June%202021-blue)]()
[![ICLR](https://img.shields.io/badge/ICLR-2022-green)]()

**저자**: Edward J. Hu, Yelong Shen, Phillip Wallis 외 (Microsoft)   
**발표**: ICLR 2022   
**arXiv**: https://arxiv.org/abs/2106.09685   

---

## 📋 목차

- [논문 소개 및 핵심 가치](#논문-소개-및-핵심-가치)
- [연구 배경 및 동기](#연구-배경-및-동기)
- [핵심 아이디어: Low-Rank 가설](#핵심-아이디어-low-rank-가설)
- [LoRA 방법론](#lora-방법론)
- [실험 결과](#실험-결과)
- [Ablation Study](#ablation-study)
- [실사용 예시](#실사용-예시)
- [한계점 및 미래 방향](#한계점-및-미래-방향)
- [참고 자료](#참고-자료)

---

## 🎯 논문 소개 및 핵심 가치

### Executive Summary

LoRA(Low-Rank Adaptation)는 Microsoft에서 발표한 **파라미터 효율적 파인튜닝(PEFT)** 기법으로, 사전 학습된 모델의 가중치를 동결한 채 각 Transformer 레이어에 학습 가능한 **저차원(low-rank) 행렬**을 삽입하는 방식입니다.

### 🏆 핵심 성과

| 지표 | LoRA | Full Fine-tuning |
|------|------|-----------------|
| **학습 가능한 파라미터** | GPT-3 대비 0.01% | 100% (175B) |
| **GPU 메모리** | ~3배 감소 | 기준 |
| **추론 지연 시간** | **0 (증가 없음)** | 기준 |
| **성능** | 동등 또는 우위 | 기준 |
| **저장 크기** | 체크포인트 10,000배 감소 | 기준 |

### 💡 왜 이 논문이 중요한가?

**Before LoRA:**
```
GPT-3(175B) 파인튜닝
├─ 필요 GPU: A100 80GB × 수십 장
├─ 저장 공간: 175B × float32 = 700GB
├─ 추론: 모델 교체 필요 (태스크당 별도 배포)
└─ 비용: 현실적으로 불가능한 수준
```

**After LoRA:**
```
GPT-3(175B) + LoRA
├─ 학습 파라미터: ~4.7M (0.003%)
├─ 저장 공간: ~35MB (태스크당)
├─ 추론: 원본 가중치 + LoRA 병합 → 추가 지연 없음
└─ 비용: 대폭 감소, 실용적 수준
```

---

## 🔙 연구 배경 및 동기

### 기존 Full Fine-Tuning의 문제

GPT-3(175B), Megatron-LM, T5 같은 초대형 언어 모델이 등장하면서, 이를 다운스트림 태스크에 적용하기 위한 파인튜닝의 비용이 급격히 증가했습니다.

**1. 메모리 문제**
```
Full Fine-tuning 메모리 구성:
├─ 파라미터: 175B × 4byte = 700GB
├─ Gradient: 700GB
├─ Optimizer State (Adam): 700GB × 2 = 1400GB
└─ 총합: ~2800GB (A100 80GB 35장 필요)
```

**2. 배포 문제**
- 태스크마다 별도의 175B 모델을 저장해야 함
- 서비스 교체 시 엄청난 I/O 및 메모리 비용

**3. 기존 대안들의 한계**

| 방법 | 아이디어 | 한계 |
|------|---------|------|
| **Adapter** | 레이어 사이에 작은 MLP 삽입 | 순차 처리 → 추론 지연 증가 |
| **Prefix Tuning** | 입력에 학습 가능한 토큰 추가 | 시퀀스 길이 감소, 학습 불안정 |
| **Prompt Tuning** | 소프트 프롬프트 학습 | 대형 모델에서만 효과적 |
| **BitFit** | Bias만 업데이트 | 성능 한계 |

---

## 💡 핵심 아이디어: Low-Rank 가설

### Intrinsic Dimensionality 연구에서 출발

Aghajanyan et al. (2020)의 연구는 사전 학습된 언어 모델이 실제로는 **매우 낮은 내재적 차원(intrinsic dimension)**에서 동작함을 보였습니다.

> **핵심 주장**: "사전 학습된 모델의 가중치 변화량(ΔW)은 실제로 낮은 랭크(rank)를 가진다."

### 직관적 이해

```
사전 학습된 거대 모델 W ∈ ℝ^(d×k):
├─ 이미 엄청난 양의 지식을 인코딩
├─ 새로운 태스크 적응에 필요한 변화(ΔW)는 상대적으로 단순
└─ ΔW가 저차원 구조를 가질 것이라는 가설

예시: d=4096, k=4096인 행렬 ΔW (약 16M 파라미터)
→ rank=4짜리 분해: B(4096×4) × A(4×4096) = 단 32K 파라미터
```

### Low-Rank 분해의 표현력

```
Full rank 행렬 ΔW:
[w11 w12 ... w1k]
[w21 w22 ... w2k]
...
[wd1 wd2 ... wdk]
(d×k = 16M 파라미터)

Low-rank 분해 ΔW = BA (rank r=4):
B = [b11 b12 b13 b14]    A = [a11 a12 ... a1k]
    [b21 b22 b23 b24]        [a21 a22 ... a2k]
    ...                      [a31 a32 ... a3k]
    [bd1 bd2 bd3 bd4]        [a41 a42 ... a4k]
(d×r + r×k = 32K 파라미터, 500배 압축!)
```

---

## 🏗️ LoRA 방법론

### 1. 수식 및 구조

**기존 Forward Pass:**
```
h = W₀x
```

**LoRA 적용 후:**
```
h = W₀x + ΔWx = W₀x + BAx
```

여기서:
- `W₀ ∈ ℝ^(d×k)`: 동결된 사전 학습 가중치
- `B ∈ ℝ^(d×r)`: 학습 가능한 행렬 (0으로 초기화)
- `A ∈ ℝ^(r×k)`: 학습 가능한 행렬 (가우시안으로 초기화)
- `r << min(d, k)`: 랭크 하이퍼파라미터

**스케일링 팩터:**
```
h = W₀x + (α/r) · BAx
```
- `α`: 스케일링 상수 (보통 r과 동일하거나 2배)
- 학습률 조정 없이 랭크 r을 변경하더라도 업데이트 규모가 일관되게 유지

### 2. 초기화 전략

```python
# LoRA 행렬 초기화
# A: 가우시안 (랜덤 초기화)
A = torch.randn(r, k) * 0.01

# B: 0으로 초기화
B = torch.zeros(d, r)

# 학습 시작 시 ΔW = BA = 0
# → 처음에는 원본 모델과 동일하게 동작
# → 학습이 진행되면서 ΔW가 형성됨
```

**왜 이렇게 초기화하는가?**
- `B=0`으로 시작 → `ΔW=0` → 학습 초기에 원본 모델 그대로 보존
- 안정적인 파인튜닝 시작점 확보

### 3. 적용 위치

논문에서는 Transformer의 **Attention 가중치 행렬**에 적용합니다.

```
Transformer Self-Attention:
├─ W_q (Query 행렬)  ← LoRA 적용
├─ W_k (Key 행렬)    ← 실험에서는 선택적
├─ W_v (Value 행렬)  ← LoRA 적용
└─ W_o (출력 행렬)   ← 실험에서는 선택적

Feed-Forward Network:
├─ W_1              ← 선택적 (보통 생략)
└─ W_2              ← 선택적 (보통 생략)
```

**논문의 핵심 발견:** 동일한 파라미터 예산 내에서는 W_q와 W_v에만 LoRA를 적용하되 랭크를 높이는 것보다, 더 많은 레이어에 낮은 랭크로 적용하는 것이 더 효과적

### 4. 추론 시 가중치 병합

LoRA의 가장 큰 장점: **추론 지연 없음**

```python
# 학습 후 가중치 병합
W_merged = W₀ + B @ A  # (α/r 스케일링 포함)

# 이후 추론은 일반 모델과 동일
output = W_merged @ x
```

```
Adapter 방식 추론:
x → W₀ → Adapter(x) → 출력  (순차 처리, 지연 있음)

LoRA 추론 (병합 후):
x → W₀ + ΔW → 출력  (단일 행렬 곱, 지연 없음)
```

### 5. PyTorch 구현

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """LoRA를 적용한 Linear 레이어"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 원본 가중치 (동결)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features),
            requires_grad=False  # 동결
        )

        # LoRA 행렬 (학습 가능)
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features)  # A: 가우시안 초기화
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank)  # B: 0으로 초기화
        )

        self.dropout = nn.Dropout(p=dropout)

        # 초기화
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B는 이미 0으로 초기화

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 원본 경로
        base_output = nn.functional.linear(x, self.weight)

        # LoRA 경로
        lora_output = (
            self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        ) * self.scaling

        return base_output + lora_output

    def merge_weights(self) -> None:
        """추론을 위해 LoRA 가중치를 원본에 병합"""
        self.weight.data += (self.lora_B @ self.lora_A) * self.scaling


class LoRAAttention(nn.Module):
    """Multi-Head Attention에 LoRA 적용"""

    def __init__(self, embed_dim, num_heads, rank=4, alpha=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Query, Key, Value, Output 행렬에 LoRA 적용
        self.q_proj = LoRALinear(embed_dim, embed_dim, rank=rank, alpha=alpha)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)  # LoRA 미적용
        self.v_proj = LoRALinear(embed_dim, embed_dim, rank=rank, alpha=alpha)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = (Q @ K.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)
```

---

## 📊 실험 결과

### 1. RoBERTa & DeBERTa (GLUE 벤치마크)

**RoBERTa-base 결과:**

| 방법 | 학습 파라미터 | MNLI | SST-2 | MRPC | CoLA | QNLI | 평균 |
|------|-------------|------|-------|------|------|------|------|
| Full FT | 125M | 87.6 | 94.8 | 90.2 | 63.6 | 92.8 | 85.8 |
| BitFit | 0.1M | 84.7 | 93.7 | 90.8 | 62.0 | 91.8 | 84.6 |
| Adapter | 0.3M | 87.1 | 94.5 | 88.4 | 60.8 | 93.0 | 84.8 |
| Prefix | 0.1M | 84.0 | 94.5 | 88.4 | 57.4 | 92.9 | 83.4 |
| **LoRA** | **0.3M** | **87.5** | **95.1** | **90.5** | **63.4** | **93.3** | **85.9** |

**DeBERTa-XXL 결과:**

| 방법 | 학습 파라미터 | MNLI | SST-2 | MRPC | CoLA | QNLI | 평균 |
|------|-------------|------|-------|------|------|------|------|
| Full FT | 1.5B | 91.7 | 97.2 | 92.0 | 72.0 | 96.0 | 89.8 |
| **LoRA** | **4.7M** | **91.9** | **96.9** | **92.6** | **72.4** | **96.3** | **90.0** |

Full Fine-tuning 대비 0.3% 파라미터만 사용하면서 **동등 이상의 성능** 달성!

### 2. GPT-2 (자연어 생성)

**E2E NLG 벤치마크:**

| 방법 | 학습 파라미터 | BLEU | NIST | MET | ROUGE-L | CIDEr |
|------|-------------|------|------|-----|---------|-------|
| GPT-2 Medium FT | 345M | 68.2 | 8.62 | 46.2 | 71.0 | 2.47 |
| GPT-2 Large FT | 774M | 68.5 | 8.78 | 46.0 | 69.9 | 2.45 |
| Adapter (M) | 0.37M | 66.3 | 8.41 | 45.0 | 69.8 | 2.40 |
| Prefix (M) | 0.35M | 69.7 | 8.81 | 46.1 | 71.4 | 2.49 |
| **LoRA (M)** | **0.35M** | **70.4** | **8.85** | **46.8** | **71.8** | **2.53** |

LoRA가 **더 큰 Full Fine-tuning 모델보다도 우수**한 성능!

### 3. GPT-3 175B (핵심 실험)

GPT-3의 경우 Full Fine-tuning 자체가 사실상 불가능한 스케일.

**WikiSQL (텍스트→SQL):**

| 방법 | 학습 파라미터 | 정확도 |
|------|-------------|--------|
| GPT-3 Zero-shot | 0 | 70.1% |
| GPT-3 Few-shot (prompt) | 0 | 78.4% |
| Full Fine-tuning (참고) | 175B | 79.2% |
| **LoRA (r=4)** | **4.7M** | **73.4%** |
| **LoRA (r=4, 더 많은 레이어)** | **37.7M** | **79.9%** |

**MultiNLI:**

| 방법 | 학습 파라미터 | 정확도 |
|------|-------------|--------|
| GPT-3 Zero-shot | 0 | 40.6% |
| Full Fine-tuning | 175B | 89.5% |
| **LoRA** | **4.7M** | **91.7%** |

Full Fine-tuning보다 **더 높은 성능** 달성 (0.003% 파라미터로)!

### 4. 학습 효율 비교

```
GPT-3 175B 파인튜닝 비교:

Full Fine-tuning:
├─ 학습 파라미터: 175,255,168,000 (175B)
├─ GPU 메모리: ~1.2TB (A100 15장)
└─ 저장 공간 (태스크당): 700GB

LoRA (r=4):
├─ 학습 파라미터: 4,718,592 (4.7M)
├─ GPU 메모리: ~350GB (A100 5장)
└─ 저장 공간 (태스크당): ~35MB

절감 효율:
├─ 파라미터: 37,000배 감소
├─ 메모리: ~3배 감소
└─ 저장 공간: 20,000배 감소
```

---

## 🧪 Ablation Study

### 1. 랭크(r)의 영향

**핵심 발견: 랭크가 높다고 무조건 좋지 않다**

```
GPT-3에서 랭크 r에 따른 성능 (WikiSQL):

r=1:  73.1%  ████████████████████████
r=2:  73.2%  ████████████████████████
r=4:  73.4%  █████████████████████████
r=8:  73.6%  █████████████████████████
r=64: 73.7%  █████████████████████████
r=256: 73.5% █████████████████████████

→ r=4~8에서 이미 충분한 성능
→ r를 높여도 개선 미미 (수렴)
```

**왜 낮은 랭크로도 충분한가?**
```
가설: ΔW의 실제 정보는 매우 낮은 차원에 집중됨

검증 (SVD 분석):
LoRA (r=64)로 학습한 ΔW를 SVD로 분해하면:
├─ 상위 1개 singular value: 전체 분산의 40%
├─ 상위 4개 singular value: 전체 분산의 75%
└─ 상위 8개 singular value: 전체 분산의 90%

결론: 실제 유용한 정보는 rank ~8 이내에 집중
```

### 2. 어떤 가중치 행렬에 적용할 것인가?

**Transformer 4가지 행렬에 대한 실험 (GPT-3, 총 파라미터 18M 고정):**

| 적용 행렬 | r | 성능 (WikiSQL) |
|---------|---|--------------|
| W_q only | 8 | 70.4% |
| W_v only | 8 | 73.0% |
| W_q, W_v | 4 | **73.4%** |
| W_q, W_k, W_v, W_o | 2 | 73.7% |
| W_q, W_k, W_v, W_o, FFN | 1 | 73.5% |

**핵심 인사이트:**
- 동일 파라미터 예산에서 **더 많은 행렬에 낮은 랭크로 적용**하는 것이 더 효과적
- W_q와 W_v를 함께 적용하는 것이 좋은 기본값

### 3. Adapter와의 공정한 비교

```
추론 지연 비교 (GPT-2 Medium):

방법          | 파라미터 | 추론 지연
-------------|---------|----------
Full FT      | 345M    | 기준 (1.0x)
Adapter      | 0.37M   | 1.06x  ← 6% 느림
Adapter (병렬)| 0.37M   | 1.03x  ← 3% 느림
LoRA         | 0.35M   | 1.00x  ← 추가 지연 없음

GPU 배치 크기가 작을 때 Adapter 지연이 더 두드러짐
(실제 서비스 환경에서 문제)
```

---

## 💻 실사용 예시

### Hugging Face PEFT 라이브러리 활용

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch

# 기본 모델 로드
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# LoRA 설정
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                      # 랭크 (보통 4~16)
    lora_alpha=16,            # 스케일링 (보통 r의 2배)
    lora_dropout=0.05,        # 드롭아웃
    bias="none",              # bias 처리
    target_modules=[          # LoRA 적용 모듈
        "q_proj",
        "v_proj",
        # "k_proj",           # 선택적
        # "o_proj",           # 선택적
    ]
)

# LoRA 적용
model = get_peft_model(model, lora_config)

# 학습 가능한 파라미터 확인
model.print_trainable_parameters()
# 출력: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622%
```

### 파인튜닝 학습 루프

```python
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

# 데이터셋 로드 (예: Alpaca)
dataset = load_dataset("tatsu-lab/alpaca")

def format_prompt(example):
    if example["input"]:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    else:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return {"text": prompt}

dataset = dataset.map(format_prompt)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./lora-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,           # LoRA는 상대적으로 큰 LR 사용 가능
    fp16=True,                    # 메모리 절약
    logging_steps=100,
    save_steps=500,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)

# 학습 시작
trainer.train()

# LoRA 가중치 저장 (매우 작은 파일!)
model.save_pretrained("./lora-weights")
# 저장 크기: ~35MB (vs 전체 모델 14GB)
```

### 추론 (Inference)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 방법 1: LoRA 가중치 별도 로드 (메모리 공유)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./lora-weights")

# 방법 2: 가중치 병합 (추론 속도 최적화)
model = model.merge_and_unload()  # LoRA를 기본 모델에 병합
# → 이후 추론은 순수 기본 모델과 동일한 속도

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

def generate_response(instruction: str, max_new_tokens: int = 256) -> str:
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:\n")[-1].strip()

# 사용 예시
response = generate_response("파이썬으로 피보나치 수열을 출력하는 함수를 작성해줘")
print(response)
```

### 다중 태스크를 위한 LoRA 전환

```python
# LoRA의 또 다른 장점: 태스크별 가중치 전환이 매우 빠름

class MultiTaskLoRAServer:
    def __init__(self, base_model_name: str):
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.current_lora = None

    def switch_task(self, task_name: str, lora_path: str):
        """태스크 전환: LoRA 가중치만 교체 (35MB 로딩)"""
        if self.current_lora:
            self.current_lora.unload()

        self.current_model = PeftModel.from_pretrained(
            self.base_model,
            lora_path,
            adapter_name=task_name
        )
        self.current_lora = task_name
        print(f"태스크 '{task_name}'으로 전환 완료")

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.current_model.device)
        with torch.no_grad():
            outputs = self.current_model.generate(**inputs, max_new_tokens=256)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# 사용 예시
server = MultiTaskLoRAServer("meta-llama/Llama-2-7b-hf")

# 코드 생성 태스크
server.switch_task("coding", "./lora-coding-weights")
code = server.generate("Sort a list in Python")

# 번역 태스크
server.switch_task("translation", "./lora-translation-weights")
translation = server.generate("Translate: Hello World")
```

---

## ⚠️ 한계점 및 미래 방향

### 1. 어떤 레이어에 어떤 랭크를 적용할지 결정 어려움

```
현재 방식: 모든 레이어에 동일한 r 적용
문제: 레이어마다 중요도가 다를 수 있음

예시:
├─ 초기 레이어: 일반적인 언어 패턴 담당
├─ 중간 레이어: 문맥 이해 담당
└─ 후기 레이어: 태스크 특화 표현 담당

→ 레이어별 중요도에 따라 r을 다르게 설정하면?
  (예: AdaLoRA, 2022)
```

### 2. 배치 추론 시 다중 태스크 처리 어려움

```
서비스 시나리오: 동시에 여러 태스크 요청 처리

문제:
├─ 배치 내 요청이 서로 다른 태스크를 위한 것일 때
├─ 각 요청마다 다른 LoRA 가중치 적용이 필요
└─ 현재 구현에서 효율적 처리 어려움

해결책:
└─ LoRAX (2023): 런타임에서 배치별 LoRA 동적 적용
```

### 3. 첫 번째 토큰 생성까지의 지연 (TTFT)

```
추론 시 가중치 병합 없이 사용할 경우:
├─ Forward pass: W₀x + BAx (두 번의 행렬 곱)
└─ 병합 없이 동적 추론 시 약간의 오버헤드

해결: merge_and_unload()로 사전 병합
단점: 태스크 전환 시마다 재병합 필요
```

### 4. 후속 발전 연구

| 방법 | 기여 | 연도 |
|------|------|------|
| **AdaLoRA** | 중요도에 따라 랭크 동적 조정 | 2022 |
| **QLoRA** | 4-bit 양자화 + LoRA로 65B 모델을 단일 GPU에서 | 2023 |
| **LoRA+** | A와 B에 다른 학습률 적용 | 2024 |
| **DoRA** | 크기와 방향 분리 학습 | 2024 |
| **rsLoRA** | 스케일링 팩터 개선 | 2023 |
| **LoftQ** | 초기화 방식 개선 | 2023 |

---

## 🎓 핵심 교훈

1. **Low-rank 가설이 실제로 유효**: 적응에 필요한 변화는 저차원 공간에 집중됨
2. **추론 지연 없음이 핵심 차별점**: Adapter 대비 가장 큰 실용적 강점
3. **적은 파라미터로도 충분**: r=4~8로 대부분의 태스크에서 Full FT와 동등한 성능
4. **많은 행렬, 낮은 랭크가 효과적**: 동일 파라미터 예산에서 분산 적용이 유리
5. **초기화 전략이 중요**: B=0 초기화로 안정적인 파인튜닝 시작점 확보

---

## 📖 참고 자료

- **원 논문**: https://arxiv.org/abs/2106.09685
- **공식 코드**: https://github.com/microsoft/LoRA
- **Hugging Face PEFT**: https://github.com/huggingface/peft
- **QLoRA 논문**: https://arxiv.org/abs/2305.14314
- **AdaLoRA 논문**: https://arxiv.org/abs/2303.10512

---

**LoRA는 LLM 파인튜닝의 현실적 장벽을 낮추어 AI 민주화에 크게 기여한 핵심 기법으로, 현재 거의 모든 오픈소스 LLM 파인튜닝의 사실상 표준(de facto standard)이 되었습니다.**
