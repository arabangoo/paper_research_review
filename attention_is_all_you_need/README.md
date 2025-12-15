# Attention Is All You Need - 논문 리뷰

## 📌 논문 소개

**제목**: Attention Is All You Need  
**저자**: Ashish Vaswani, Noam Shazeer, Niki Parmar 외 (Google Brain/Research)  
**발표**: NIPS 2017  
**arXiv**: https://arxiv.org/abs/1706.03762

## 🎯 핵심 가치

이 논문은 자연어 처리 분야의 패러다임을 완전히 바꾼 혁신적인 연구입니다.    
RNN과 CNN 없이 오직 **Attention 메커니즘**만으로 구성된 **Transformer 아키텍처**를 제안하여 많은 성과를 이루었습니다.       

- 기존 Seq2Seq 모델들의 순차 처리 병목 현상 해결
- 병렬 처리를 통한 학습 속도 대폭 향상
- WMT 2014 영어-독일어 번역에서 BLEU 28.4 달성 (당시 SOTA)
- BERT, GPT 등 현대 LLM의 기반이 되는 아키텍처 제시

## 🔙 연구 배경 및 동기

### 기존 모델의 한계

**1. RNN/LSTM의 근본적인 문제**
- **순차 처리 제약**: 시간 t의 출력을 계산하려면 t-1의 hidden state가 필요 → 병렬화 불가
- **Long-term Dependency 문제**: 긴 시퀀스에서 초기 정보가 손실되는 vanishing gradient 문제
- **학습 시간**: 긴 문장일수록 학습 시간이 선형적으로 증가
- **메모리 병목**: 각 timestep의 hidden state를 저장해야 하므로 메모리 비효율

**2. CNN 기반 Seq2Seq의 한계**
- **ByteNet, ConvS2S** 같은 모델들이 병렬화를 시도했지만:
  - 긴 거리 dependency 학습에 여전히 제약 (O(log_k(n)) 경로 길이)
  - Receptive field 확장을 위해 많은 레이어 필요
  - 계산 복잡도: O(k·n·d²) - 커널 크기 k에 비례

**3. Attention Mechanism의 부상**
- **Bahdanau et al. (2015)**: Seq2Seq에 Attention 도입
- **문제점**: RNN/LSTM과 함께 사용 → 여전히 순차 처리 필요
- **핵심 질문**: "Attention만으로 충분하지 않을까?" → 이 논문의 출발점

### 이 논문이 해결하고자 한 핵심 과제

1. **병렬화**: 순차 처리 없이 모든 위치를 동시에 계산
2. **효율성**: 적은 계산량으로 긴 dependency 학습
3. **성능**: 기존 SOTA 모델 능가
4. **일반화**: 다양한 NLP 태스크에 적용 가능한 범용 아키텍처

## 🏗️ 모델 아키텍처

### 1. 전체 구조
Transformer는 Encoder-Decoder 구조를 따르지만, RNN/LSTM을 사용하지 않습니다.

```
Inputs → Input Embedding + Positional Encoding
        ↓
    [Encoder Stack (N=6)]
        ↓
    [Decoder Stack (N=6)]
        ↓
    Linear + Softmax → Output Probabilities
```

### 2. Encoder
- **N=6개**의 동일한 레이어를 쌓은 구조
- 각 레이어는 2개의 Sub-layer로 구성:
  - Multi-Head Self-Attention
  - Position-wise Feed-Forward Network
- 각 Sub-layer 후에 **Residual Connection + Layer Normalization** 적용
- 출력 차원: **d_model = 512**

### 3. Decoder
- **N=6개**의 동일한 레이어를 쌓은 구조
- 각 레이어는 3개의 Sub-layer로 구성:
  - **Masked** Multi-Head Self-Attention (미래 토큰 참조 방지)
  - Multi-Head Encoder-Decoder Attention
  - Position-wise Feed-Forward Network
- 동일하게 **Residual Connection + Layer Normalization** 적용## 🔍 핵심 메커니즘 해설

### 1. Scaled Dot-Product Attention

**수식**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**동작 원리**:
1. Query(Q)와 Key(K)의 내적으로 유사도 계산
2. √d_k로 스케일링 (gradient 안정화)
3. Softmax로 확률 분포 변환
4. Value(V)에 가중치를 곱해 최종 출력

**왜 스케일링이 필요한가?**
- d_k가 클 때 내적 값이 너무 커져 softmax가 극단적으로 작은 gradient를 가지게 됨
- 이를 방지하기 위해 √d_k로 나눠줌

### 2. Multi-Head Attention

**개념**:
- 단일 Attention 대신 **h=8개**의 병렬 Attention Head 사용
- 각 Head는 서로 다른 representation subspace를 학습
- 최종적으로 Concatenate 후 Linear 변환

**수식**:
```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**장점**:
- 다양한 관점에서 문맥 정보 포착
- 병렬 처리로 계산 효율성 유지 (총 계산량은 single-head와 유사)

### 3. Attention의 3가지 활용

**(1) Encoder Self-Attention**
- Q, K, V 모두 이전 Encoder 레이어 출력 → 입력 문장 내 모든 단어 간 관계 학습
- 예시: "I love her"
  - "I"가 Query일 때 → "I", "love", "her" 모두와 관계 계산

**(2) Decoder Self-Attention (Masked)**
- Q, K, V 모두 이전 Decoder 레이어 출력 → 단, 미래 위치 참조 방지 (Masking)
- **Masking 이유**: Auto-regressive 속성 유지 (미래 정보 누설 방지)

**(3) Encoder-Decoder Attention**
- Q: Decoder 출력, K, V: Encoder 최종 출력 → 입력 문장과 출력 문장 간 관계 학습

### 4. Positional Encoding

**문제**: Transformer는 순차 처리가 없어 위치 정보 부재  
**해결**: 사인/코사인 함수로 위치 정보 인코딩

**수식**:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**특징**:
- 각 위치마다 고유한 벡터 생성
- 상대적 위치 관계 학습 가능
- 학습 데이터보다 긴 시퀀스에도 대응 가능

### 5. Position-wise Feed-Forward Network

**구조**:
```python
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```
- 2개의 Linear Layer + ReLU activation
- 입출력 차원: **d_model = 512**
- 중간 차원: **d_ff = 2048**
- 각 위치마다 독립적으로 적용

### 6. Residual Connection과 Layer Normalization

**왜 필요한가?**
- Transformer는 6개의 레이어를 쌓은 깊은 네트워크
- 깊은 네트워크의 고질적 문제: gradient vanishing/exploding

**Residual Connection (Skip Connection)**
```python
# 각 Sub-layer의 출력
output = LayerNorm(x + Sublayer(x))
```

**동작 원리**:
1. Sublayer(x): Attention 또는 FFN 계산
2. x + Sublayer(x): 입력을 직접 더함 (residual)
3. LayerNorm: 정규화

**효과**:
- **Gradient Flow 개선**: Backprop 시 gradient가 residual path를 통해 직접 전달
- **학습 안정화**: 초기 레이어의 학습이 용이
- **Identity Mapping**: 필요시 Sublayer가 0을 학습하여 입력을 그대로 통과 가능

**Layer Normalization**
```python
# 각 샘플, 각 위치마다 독립적으로 정규화
mean = x.mean(dim=-1, keepdim=True)
std = x.std(dim=-1, keepdim=True)
LayerNorm(x) = γ * (x - mean) / (std + ε) + β
```

**Batch Norm vs Layer Norm**:
| 구분 | Batch Normalization | Layer Normalization |
|------|---------------------|---------------------|
| 정규화 축 | Batch 차원 | Feature 차원 |
| 시퀀스 처리 | 길이가 다르면 문제 | 각 샘플 독립적 처리 |
| 추론 시 | Running statistics 필요 | 추가 통계 불필요 |
| Transformer | ❌ 부적합 | ✅ 적합 |

**실제 구현**:
```python
class SublayerConnection(nn.Module):
    """Residual connection + Layer Normalization"""
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # Post-LN: sublayer 후 normalization
        return x + self.dropout(sublayer(self.norm(x)))
```

**Pre-LN vs Post-LN**:
- **논문 (Post-LN)**: LayerNorm(x + Sublayer(x))
- **현대 구현 (Pre-LN)**: x + Sublayer(LayerNorm(x))
  - Pre-LN이 학습 안정성 면에서 더 우수 (GPT-2/3, BERT 등에서 채택)## ⚡ 주요 장점

### 1. 계산 복잡도 비교

| Layer Type | Complexity per Layer | Sequential Operations | Max Path Length |
|------------|---------------------|----------------------|-----------------|
| Self-Attention | O(n²·d) | O(1) | O(1) |
| Recurrent | O(n·d²) | O(n) | O(n) |
| Convolutional | O(k·n·d²) | O(1) | O(log_k(n)) |

**Self-Attention의 우위**:
- **병렬 처리**: 순차 연산 O(1)
- **Long-range dependency**: 최대 경로 길이 O(1)
- **계산 효율**: 대부분의 경우 n < d이므로 RNN보다 빠름

**상세 복잡도 분석**

**시간 복잡도**:
```
Self-Attention 한 층의 계산량:
1. Q, K, V 계산: 3 × (n × d × d) = O(n·d²)
2. Attention Score: (n × d) × (d × n) = O(n²·d)
3. Attention × V: (n × n) × (n × d) = O(n²·d)
→ 총: O(n²·d + n·d²)
→ n < d일 때: O(n·d²) (RNN과 동일)
→ n > d일 때: O(n²·d) (병목)
```

**실제 예시 (d_model=512)**:
```python
# 짧은 문장 (n=50)
Self-Attention: 50² × 512 = 1.28M ops
RNN: 50 × 512² = 13.1M ops
→ Self-Attention이 10배 빠름

# 긴 문서 (n=2048)
Self-Attention: 2048² × 512 = 2.15B ops
RNN: 2048 × 512² = 537M ops
→ RNN이 4배 빠름 (하지만 순차 처리 필요)
```

**공간 복잡도 (메모리)**:
```
1. Attention Matrix: O(n² × h)
   - n=512, h=8: 512² × 8 = 2.1M 요소
   - n=2048, h=8: 2048² × 8 = 33.6M 요소 (16배 증가!)

2. Key-Value Cache (추론 시):
   - 각 레이어마다: O(n × d × 2)
   - 6개 레이어: O(6 × n × 1024) = O(n × 6K)
```

**처리량(Throughput) vs 지연시간(Latency) 트레이드오프**:
- **학습**: Transformer 압도적 우위 (병렬화)
- **추론 (짧은 시퀀스)**: Transformer 우세
- **추론 (긴 시퀀스)**: RNN이 메모리 효율적
- **스트리밍**: RNN 유리 (토큰별 순차 생성)

### 2. 병렬화 가능
- **RNN**: 이전 hidden state 필요 → 순차 처리 필수
- **Transformer**: 모든 위치 동시 계산 → GPU 활용 극대화

### 3. Long-range Dependency 학습
- **RNN**: 긴 문장에서 정보 손실 (vanishing gradient)
- **Transformer**: 모든 위치 간 직접 연결 (O(1) path)

## 📊 실험 결과

### 1. 기계 번역 성능 (WMT 2014)

| Model | EN-DE BLEU | EN-FR BLEU | Training Cost |
|-------|------------|------------|---------------|
| GNMT + RL | 24.6 | 39.92 | 2.3 × 10¹⁹ FLOPs |
| ConvS2S | 25.16 | 40.46 | 1.5 × 10²⁰ FLOPs |
| Transformer (base) | 27.3 | 38.1 | 3.3 × 10¹⁸ FLOPs |
| Transformer (big) | **28.4** | **41.8** | 2.3 × 10¹⁹ FLOPs |

**학습 환경**:
- Base model: 8 × P100 GPU, 12시간
- Big model: 8 × P100 GPU, 3.5일

### 2. 하이퍼파라미터

```python
# Base Model
N = 6                # Encoder/Decoder layers
d_model = 512        # Hidden dimension
d_ff = 2048          # FFN inner dimension
h = 8                # Attention heads
d_k = d_v = 64       # Key/Value dimension (d_model/h)
P_drop = 0.1         # Dropout rate

# Big Model
d_model = 1024
d_ff = 4096
h = 16
P_drop = 0.3
```

### 3. 학습 기법 및 정규화

**Optimizer: Adam**
```python
# 논문에서 사용한 설정
beta1 = 0.9
beta2 = 0.98
epsilon = 1e-9
```

**Learning Rate Scheduler (핵심!)**
```python
lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
```

**Warmup 전략**:
- **warmup_steps = 4000**
- 초기 4000 step 동안 learning rate를 선형으로 증가
- 이후 step 수의 제곱근에 반비례하여 감소

**왜 Warmup이 필요한가?**
1. **파라미터 초기화 문제**: 초기에 파라미터가 불안정한 상태
2. **큰 learning rate 위험**: 초기 큰 LR은 발산 위험
3. **점진적 학습**: 작은 LR로 시작해 안정화 후 본격 학습

**시각화**:
```
Learning Rate
│
│     /╲
│    /  ╲___
│   /       ╲___
│  /            ╲___
│ /                 ╲___
│/________________________
  warmup    decay phase
  (4000)
```

**구현 예시**:
```python
class NoamOptimizer:
    def __init__(self, d_model, warmup_steps, optimizer):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.step()
```

**Regularization**:
1. **Dropout (P_drop = 0.1)**:
   - 각 Sub-layer 출력에 적용
   - Attention weights에도 적용
   - Embeddings + Positional Encoding에도 적용

2. **Label Smoothing (ε = 0.1)**:
   ```python
   # Hard label: [0, 0, 1, 0, 0]
   # Smoothed label: [0.025, 0.025, 0.9, 0.025, 0.025]
   ```
   - **효과**: Overfitting 방지, 모델이 너무 확신하지 않도록
   - **BLEU 향상**: 정확도는 약간 떨어지지만 일반화 성능 향상## 💻 실사용 예시

### 1. PyTorch 기본 구현

```python
import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
        
    def forward(self, Q, K, V, mask=None):
        # Q, K, V: [batch_size, n_heads, seq_len, d_k]
        
        # Attention Score 계산
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Masking (옵션)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Value와 곱하기
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear layers for Q, K, V
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear projection and split into heads
        Q = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        x, attn_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear layer
        output = self.W_O(x)
        
        return output, attn_weights
```### 2. Positional Encoding 구현

```python
import numpy as np

def get_positional_encoding(seq_len, d_model):
    """
    Args:
        seq_len: 시퀀스 길이
        d_model: 모델 차원
    Returns:
        positional_encoding: [seq_len, d_model]
    """
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    
    return torch.FloatTensor(pos_encoding)

# 사용 예시
seq_len = 100
d_model = 512
pos_enc = get_positional_encoding(seq_len, d_model)

# Input embedding에 더하기
input_embeddings = torch.randn(1, seq_len, d_model)  # [batch, seq_len, d_model]
output = input_embeddings + pos_enc.unsqueeze(0)
```

### 3. Hugging Face Transformers 활용

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 모델 로드 (예: T5 - Transformer 기반)
model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 번역 예시
def translate(text, model, tokenizer):
    # Tokenization
    inputs = tokenizer(
        f"translate English to German: {text}",
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    
    # Generate
    outputs = model.generate(
        inputs.input_ids,
        max_length=512,
        num_beams=4,
        early_stopping=True
    )
    
    # Decode
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# 실행
text = "Hello, how are you?"
result = translate(text, model, tokenizer)
print(result)
```## 🎯 실무 적용 시 고려사항

### 1. 메모리 최적화

```python
# Gradient Checkpointing (메모리 절약)
model.gradient_checkpointing_enable()

# Mixed Precision Training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 2. 긴 시퀀스 처리

```python
# Sliding Window Attention (긴 문서 처리)
from transformers import LongformerModel

model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
# 최대 4096 토큰까지 처리 가능
```

### 3. 추론 최적화

```python
# Model Quantization (추론 속도 향상)
from transformers import AutoModelForSeq2SeqLM
import torch

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# ONNX Export (프로덕션 배포)
from transformers.onnx import export

export(
    preprocessor=tokenizer,
    model=model,
    config=model.config,
    opset=13,
    output=Path("model.onnx")
)
```

## 🧪 Ablation Study (소거 실험)

논문에서는 각 컴포넌트의 중요성을 검증하기 위해 체계적인 실험을 수행했습니다.

### 1. Attention Head 개수 변화

| Heads (h) | d_k | BLEU (EN-DE) | PPL (EN-DE) | Params |
|-----------|-----|--------------|-------------|--------|
| 1 | 512 | 25.8 | 5.29 | 65M |
| 4 | 128 | 27.2 | 4.91 | 65M |
| **8** | **64** | **27.3** | **4.88** | **65M** |
| 16 | 32 | 27.3 | 4.91 | 65M |
| 32 | 16 | 26.5 | 5.01 | 65M |

**결론**:
- h=8이 최적 (너무 많거나 적으면 성능 저하)
- 단일 Head보다 Multi-Head가 확실히 우수 (25.8 vs 27.3)
- d_k가 너무 작으면 (h=32, d_k=16) 표현력 부족

### 2. Key/Value 차원 (d_k) 변화

| d_k | d_model | BLEU | 분석 |
|-----|---------|------|------|
| 64 | 512 | **27.3** | 최적 균형 |
| 128 | 512 | 27.2 | 약간 과다 |
| 32 | 512 | 26.4 | 표현력 부족 |

**해석**: d_k=64가 충분한 표현력과 계산 효율의 균형점

### 3. Model Size 비교

| 구분 | N | d_model | d_ff | h | Params | BLEU | Train Time |
|------|---|---------|------|---|--------|------|------------|
| Base | 6 | 512 | 2048 | 8 | 65M | 27.3 | 12h |
| Big | 6 | 1024 | 4096 | 16 | 213M | **28.4** | 3.5일 |
| Small | 6 | 256 | 1024 | 4 | 16M | 24.9 | 6h |

### 4. Positional Encoding 방식 비교

| 방식 | BLEU | 설명 |
|------|------|------|
| Sinusoidal (논문) | **27.3** | sin/cos 함수 사용 |
| Learned | 27.2 | 학습 가능한 embedding |

**놀라운 발견**:
- 학습된 positional encoding과 성능 차이 거의 없음
- Sinusoidal의 장점: 학습 시퀀스보다 긴 입력에도 일반화 가능

### 5. Dropout 비율 영향

| P_drop | BLEU (Base) | BLEU (Big) |
|--------|-------------|------------|
| 0.0 | 26.8 | 27.6 |
| 0.1 | **27.3** | 28.1 |
| 0.2 | 27.1 | **28.4** |
| 0.3 | 26.9 | 28.3 |

**패턴**: 큰 모델일수록 더 높은 dropout 필요 (overfitting 방지)

### 6. Attention Type 비교

| Attention 종류 | EN-DE BLEU | 설명 |
|----------------|------------|------|
| Multi-Head (논문) | **27.3** | 8개 Head 병렬 |
| Single-Head | 25.8 | 1개 Head만 |
| Multi-Head (no residual) | 24.2 | Residual 제거 시 |
| Multi-Head (no LayerNorm) | Diverge | 학습 실패 |

**핵심 발견**:
- **Multi-Head 필수**: +1.5 BLEU 향상
- **Residual Connection 필수**: 없으면 -3.1 BLEU
- **Layer Normalization 필수**: 없으면 학습 자체가 불안정

### 7. FFN 중간 차원 (d_ff) 영향

| d_ff | d_model | BLEU | 분석 |
|------|---------|------|------|
| 1024 | 512 | 26.1 | 용량 부족 |
| **2048** | **512** | **27.3** | 최적 (4배) |
| 4096 | 512 | 27.4 | 약간 향상 (계산 비용 2배) |

**경험적 법칙**: d_ff = 4 × d_model이 효율적

## 🔬 Attention 시각화

### Attention Weights 확인

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs, output_attentions=True)
    
    # 첫 번째 레이어, 첫 번째 헤드의 Attention
    attention = outputs.attentions[0][0, 0].detach().numpy()
    
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis"
    )
    plt.title("Attention Weights")
    plt.show()

# 사용
text = "The cat sat on the mat"
visualize_attention(model, tokenizer, text)
```

## 📚 주요 개념 정리

### Q, K, V의 직관적 이해

**딕셔너리 비유**:
```python
# 일반 딕셔너리
dictionary = {
    "cat": "고양이",
    "dog": "강아지"
}
result = dictionary["cat"]  # 정확히 일치해야 값 반환

# Attention Mechanism
# Query: "cat" (찾고자 하는 것)
# Keys: ["cat", "dog", "animal", ...]
# Values: ["고양이", "강아지", "동물", ...]
# → "cat"과 각 Key의 유사도를 계산하여 Value들의 가중합 반환
```

### Self-Attention vs Cross-Attention

| 구분 | Self-Attention | Cross-Attention |
|------|----------------|-----------------|
| Q 출처 | 같은 시퀀스 | Decoder |
| K, V 출처 | 같은 시퀀스 | Encoder |
| 목적 | 문장 내 단어 간 관계 | 입력-출력 간 관계 |
| 예시 | Encoder Self-Attention | Encoder-Decoder Attention |## ⚠️ 논문의 한계점

### 1. 계산 복잡도 문제
- **O(n²) 복잡도**: 시퀀스 길이가 길어질수록 메모리/계산량 폭증
  ```
  seq_len = 512  → Attention Matrix: 512×512 = 262K
  seq_len = 2048 → Attention Matrix: 2048×2048 = 4.2M (16배 증가)
  ```
- **긴 문서 처리 어려움**: 논문 실험은 대부분 100단어 이하 문장
- **실시간 처리 한계**: RNN보다 추론 latency가 높을 수 있음

### 2. Positional Encoding의 한계
- **절대 위치 정보**: 상대적 위치 관계를 직접 학습하지 못함
- **최대 길이 제한**: 학습 시퀀스보다 훨씬 긴 입력은 성능 저하
- **후속 연구에서 개선**:
  - Relative Positional Encoding (Shaw et al., 2018)
  - Rotary Position Embedding (RoPE, Su et al., 2021)

### 3. Inductive Bias 부족
- **RNN**: 순차성 (sequential bias)
- **CNN**: 지역성 (locality bias)
- **Transformer**: 구조적 가정 없음 → **데이터가 많이 필요**
  - 작은 데이터셋에서는 RNN/CNN보다 성능이 낮을 수 있음

### 4. 해석 가능성 문제
- Attention weights가 항상 의미 있는 것은 아님
- Multi-Head의 각 Head가 정확히 무엇을 학습하는지 불명확
- "Attention is not Explanation" (Jain & Wallace, 2019) 논쟁

### 5. 실무적 제약
- **메모리 요구량**: Base model 학습에도 8×P100 GPU 필요
- **학습 시간**: Big model은 3.5일 (비용 높음)
- **에너지 소비**: 환경적 영향 (탄소 발자국)

## 🚀 발전 방향 및 후속 연구

### Transformer 이후 등장한 주요 모델

**1. Encoder-only 모델**
- **BERT (2018)**:
  - 양방향 학습으로 문맥 이해 향상
  - Masked Language Modeling (MLM)
  - 11개 NLP 태스크에서 SOTA 달성
- **RoBERTa, ALBERT, ELECTRA**: BERT 개선 변형들

**2. Decoder-only 모델 (현대 LLM의 주류)**
- **GPT (2018)**: 자기회귀 생성
- **GPT-2 (2019)**: 1.5B 파라미터, Zero-shot 학습
- **GPT-3 (2020)**: 175B 파라미터, Few-shot 학습
- **GPT-4 (2023)**: 멀티모달, 추론 능력 향상
- **LLaMA, Mistral, Claude**: 오픈소스/상용 LLM

**3. Encoder-Decoder 모델**
- **T5 (2019)**: 모든 태스크를 Text-to-Text로 통합
- **BART (2020)**: Denoising autoencoder
- **mT5, mBART**: 다국어 변형

**4. 효율성 개선 (O(n²) 문제 해결)**
- **Longformer (2020)**: Sliding Window Attention → O(n)
- **BigBird (2020)**: Sparse Attention → O(n)
- **Linformer (2020)**: Low-rank Approximation → O(n)
- **Flash Attention (2022)**: 메모리 최적화 알고리즘

**5. Vision/Multimodal 확장**
- **Vision Transformer (ViT, 2020)**: 이미지 분류
- **CLIP (2021)**: 이미지-텍스트 학습
- **Flamingo, GPT-4V**: 멀티모달 LLM

**6. 아키텍처 개선**
- **Sparse Transformers**: 계산량 감소
- **Mixture of Experts (MoE)**: 조건부 계산
- **State Space Models (Mamba, 2023)**: Transformer 대안

### 핵심 트렌드
1. **스케일 업**: 수조 개 파라미터 (GPT-4, PaLM)
2. **효율화**: Attention 복잡도 줄이기
3. **멀티모달**: 텍스트 넘어 이미지/오디오/비디오
4. **In-context Learning**: 파인튜닝 없이 Few-shot으로 학습

## 🎓 핵심 교훈

1. **병렬화가 핵심**: RNN의 순차 처리 제약을 제거
2. **Attention이 충분하다**: 복잡한 구조 없이도 SOTA 달성
3. **Position 정보 필수**: 순서 정보를 명시적으로 주입
4. **Multi-Head의 힘**: 다양한 관점에서 문맥 파악
5. **Scaling의 중요성**: 모델 크기 확장으로 성능 향상

## 📖 참고 자료

- **원 논문**: https://arxiv.org/abs/1706.03762
- **공식 코드**: https://github.com/tensorflow/tensor2tensor
- **The Illustrated Transformer**: https://jalammar.github.io/illustrated-transformer/
- **Annotated Transformer**: http://nlp.seas.harvard.edu/annotated-transformer/

---

**이 논문은 현대 LLM의 근간이 되는 아키텍처를 제시했으며, 실무에서 반드시 이해해야 할 필수 지식입니다.**