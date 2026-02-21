# The Rise and Potential of Large Language Model Based Agents: A Survey

> **AGI를 향한 여정: LLM을 두뇌로 하는 자율 에이전트의 현재와 미래에 대한 종합 서베이**

[![arXiv](https://img.shields.io/badge/arXiv-2309.07864-b31b1b.svg)](https://arxiv.org/abs/2309.07864)
[![Publication Date](https://img.shields.io/badge/Published-September%202023-blue)]()
[![Pages](https://img.shields.io/badge/Pages-86-orange)]()
[![GitHub](https://img.shields.io/badge/GitHub-Paper--List-181717.svg)](https://github.com/WooooDyy/LLM-Agent-Paper-List)

**저자**: Zhiheng Xi, Wenxiang Chen, Xin Guo 외 (복단대학교 등)
**발표**: 2023년 9월 / SCIS (Science China Information Sciences) 커버 논문
**arXiv**: https://arxiv.org/abs/2309.07864

---

## 📋 목차

- [논문 소개 및 핵심 가치](#논문-소개-및-핵심-가치)
- [에이전트란 무엇인가](#에이전트란-무엇인가)
- [두뇌 모듈 (Brain)](#두뇌-모듈-brain)
- [지각 모듈 (Perception)](#지각-모듈-perception)
- [행동 모듈 (Action)](#행동-모듈-action)
- [단일 에이전트 응용](#단일-에이전트-응용)
- [멀티 에이전트 시스템](#멀티-에이전트-시스템)
- [인간-에이전트 협력](#인간-에이전트-협력)
- [에이전트 사회](#에이전트-사회)
- [벤치마크 및 평가](#벤치마크-및-평가)
- [핵심 도전 과제](#핵심-도전-과제)
- [실사용 예시](#실사용-예시)
- [참고 자료](#참고-자료)

---

## 🎯 논문 소개 및 핵심 가치

### Executive Summary

이 논문은 LLM(Large Language Model)을 핵심 두뇌로 하는 **자율 에이전트(Autonomous Agent)**의 역사, 구조, 응용, 도전 과제를 총망라한 **86페이지의 종합 서베이**입니다.

저자들은 LLM이 AGI(Artificial General Intelligence)의 잠재적 불꽃이 될 수 있다고 보며, 에이전트 개념의 철학적 기원부터 최신 연구까지 체계적으로 정리했습니다.

### 🏆 논문의 핵심 기여

```
1. 통합 프레임워크 제시
   └─ Brain + Perception + Action 3요소 아키텍처

2. 포괄적 응용 범위
   ├─ 단일 에이전트 (Single-Agent)
   ├─ 멀티 에이전트 (Multi-Agent)
   └─ 인간-에이전트 협력

3. 에이전트 사회 연구
   └─ 창발적 사회 행동, 성격, 가치 전파

4. 향후 연구 방향
   └─ 미해결 문제 및 도전 과제 정리
```

### 📊 이 서베이가 다루는 범위

```
에이전트 진화 타임라인:

1950년대        1980년대        2010년대        2022년~
철학적 개념  →  규칙 기반 AI  →  딥러닝 에이전트  →  LLM 기반 에이전트
(Turing Test)   (Expert System)  (DQN, AlphaGo)    (GPT-4, LLaMA)

핵심 전환점: ChatGPT 등장 이후 LLM이 에이전트의 "두뇌" 역할 가능성 증명
```

---

## 🤖 에이전트란 무엇인가

### 철학적 정의부터 AI까지

```
[철학적 기원]
에이전트(Agent) = "행동하는 존재" (라틴어 agere: to do)
├─ 자율성 (Autonomy): 스스로 결정
├─ 반응성 (Reactivity): 환경에 반응
├─ 능동성 (Proactiveness): 목표 추구
└─ 사회성 (Social Ability): 다른 에이전트와 상호작용
```

### LLM이 에이전트의 두뇌가 될 수 있는 이유

기존 AI 에이전트의 한계를 LLM이 어떻게 극복하는지:

| 특성 | 기존 AI 에이전트 | LLM 기반 에이전트 |
|------|----------------|-----------------|
| **지식** | 좁은 도메인 규칙 | 방대한 세계 지식 |
| **추론** | 수식/논리 규칙 | 자연어 추론 (CoT 등) |
| **언어** | 제한적 NLU | 유창한 이해·생성 |
| **적응성** | 재학습 필요 | In-context Learning |
| **멀티모달** | 단일 입력 | 텍스트·이미지·오디오 |
| **일반화** | 태스크 특화 | 범용 문제 해결 |

### 논문의 통합 프레임워크

```
[지각 (Perception)]
  ├─ 텍스트 입력
  ├─ 이미지 입력
  ├─ 오디오 입력
  └─ 기타 센서 데이터
       ↓
[두뇌 (Brain)] - LLM Core (GPT-4, LLaMA 등)
  ├─ 지식 (Knowledge)
  ├─ 메모리 (Memory)
  ├─ 추론 (Reasoning)
  └─ 계획 (Planning)
       ↓
[행동 (Action)]
  ├─ 도구 사용
  ├─ 외부 실행
  └─ 환경 조작
```

---

## 🧠 두뇌 모듈 (Brain)

에이전트의 핵심으로, LLM이 수행하는 4가지 주요 기능입니다.

### 1. 지식 (Knowledge)

LLM에 내재화된 지식과 그 한계:

```
LLM의 지식 유형:
├─ 언어적 지식 (Linguistic Knowledge)
│   └─ 문법, 어휘, 담화 구조 등
│
├─ 상식 지식 (Commonsense Knowledge)
│   └─ 물리 법칙, 사회 규범, 일상 경험 등
│
├─ 전문 도메인 지식 (Domain Knowledge)
│   └─ 의학, 법률, 코딩, 수학 등
│
└─ 실행 가능 지식 (Actionable Knowledge)
    └─ 어떻게 행동할지 아는 절차적 지식
```

**지식의 한계와 대응:**

```
문제: 환각(Hallucination)
├─ 존재하지 않는 사실 생성
├─ 지식 커트오프 이후 정보 부재
└─ 전문 도메인의 깊이 한계

해결책:
├─ RAG (Retrieval-Augmented Generation): 외부 DB 검색
├─ Tool Use: 검색엔진, 계산기 등 활용
└─ 지식 편집 (Knowledge Editing): 특정 사실 수정
```

### 2. 메모리 (Memory)

인간의 기억 체계를 모방한 에이전트 메모리 구조:

```
감각 메모리 (Sensory)   →   단기 메모리 (Short-term)   →   장기 메모리 (Long-term)
현재 입력                   컨텍스트 윈도우                  외부 스토리지
(이미지, 텍스트)             (수천~수만 토큰)                 (벡터 DB 등)
```

**메모리 유형 상세:**

| 메모리 유형 | 구현 방식 | 용량 | 지속성 |
|------------|---------|------|-------|
| **감각 메모리** | 현재 입력 (원시 데이터) | 수 KB | 일시적 |
| **단기 메모리** | 컨텍스트 윈도우 | 수만 토큰 | 세션 내 |
| **장기 메모리 (외현)** | 벡터 DB, 외부 스토리지 | 무제한 | 영구 |
| **장기 메모리 (내재)** | 모델 파라미터 | 고정 | 영구 |

**메모리 확장 전략:**

```python
# 메모리 전략 예시

# 1. 요약 기반 압축 (Summarization)
def compress_memory(conversation_history):
    """긴 대화를 요약하여 컨텍스트 절약"""
    summary = llm.summarize(conversation_history[-100:])
    return summary + conversation_history[-10:]  # 요약 + 최근 대화 유지

# 2. 벡터 DB 기반 검색 메모리 (RAG)
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

vector_store = Chroma(embedding_function=OpenAIEmbeddings())

def retrieve_relevant_memory(query: str, k: int = 5):
    """관련 과거 경험을 벡터 유사도로 검색"""
    return vector_store.similarity_search(query, k=k)

# 3. 메모리 구조화 (Structured Memory)
memory = {
    "episodic": [],      # 구체적 경험 기록
    "semantic": {},      # 추상화된 지식
    "procedural": [],    # 행동 패턴/스킬
}
```

### 3. 추론 (Reasoning)

LLM 에이전트의 다양한 추론 전략:

**Chain-of-Thought (CoT):**
```
일반 프롬프트:
"3+5×2는?" → "16" (틀림)

CoT 프롬프트:
"3+5×2는? 단계별로 생각하자."
→ "곱셈 먼저: 5×2=10
    그 다음 덧셈: 3+10=13
    답: 13" (정확)
```

**Tree of Thoughts (ToT):**
```
단선적 추론 (CoT):
문제 → A → B → C → 답

트리 추론 (ToT):
         문제
        /  |  \
       A   B   C
      / \   \  |
     A1 A2   B2 C1
         |
        A21 ← 최적 경로 선택 (MCTS 등)
```

**ReAct (Reasoning + Acting):**
```python
# ReAct 패턴 예시
def react_agent(question):
    while True:
        # Thought: 추론
        thought = llm.think(f"질문: {question}\n현재까지: {history}")

        # Action: 행동 결정
        action = llm.decide_action(thought)

        if action.type == "search":
            # Observation: 결과 관찰
            observation = search_engine(action.query)
            history += f"검색결과: {observation}\n"

        elif action.type == "finish":
            return action.answer
```

**Reflexion (자기 반성):**
```
1회차: 코드 작성 → 테스트 실패
   ↓ (반성)
"오류 원인: 경계 조건 처리 누락"
   ↓
2회차: 수정 코드 → 테스트 통과
```

### 4. 계획 (Planning)

복잡한 목표를 실행 가능한 단계로 분해:

**계획 전략 비교:**

```
1. 태스크 분해 (Task Decomposition)
   목표: "논문 리뷰 작성"
   → Step 1: 논문 PDF 다운로드
   → Step 2: 핵심 섹션 요약
   → Step 3: 관련 선행 연구 조사
   → Step 4: 비판적 분석 작성
   → Step 5: 최종 리뷰 정리

2. 역방향 계획 (Backward Planning)
   목표에서 역으로 단계 추론:
   최종 목표 ← 중간 목표 ← 현재 상태

3. 계획-실행-반성 사이클
   Plan → Act → Observe → Reflect → Re-Plan
   (실패를 학습하고 계획 수정)
```

---

## 👁️ 지각 모듈 (Perception)

에이전트가 환경을 이해하는 방법:

### 텍스트 지각

```
텍스트 입력 처리:
├─ 자연어 명령 이해
├─ 코드 파싱
├─ 구조화 데이터 (JSON, CSV, XML)
└─ 대화 맥락 추적
```

### 멀티모달 지각

```
시각 입력:
├─ ViT (Vision Transformer): 이미지 → 패치 임베딩
├─ CLIP: 이미지-텍스트 정렬
├─ BLIP-2, LLaVA: 이미지를 LLM 입력으로 변환
└─ GPT-4V: 네이티브 멀티모달 이해

오디오 입력:
├─ Whisper: 음성 → 텍스트
└─ AudioLM: 오디오 직접 이해

기타:
├─ 3D 공간 (로봇 에이전트)
├─ 제스처/포즈 인식
└─ 생체 신호 (의료 에이전트)
```

**멀티모달 에이전트 아키텍처:**

```python
# 멀티모달 입력 처리 예시 (LLaVA 스타일)
class MultimodalAgent:
    def __init__(self):
        self.vision_encoder = CLIPVisualEncoder()   # 이미지 → 임베딩
        self.language_model = LLaMA()               # 언어 모델
        self.projection = LinearProjection()         # 임베딩 정렬

    def perceive(self, text_input=None, image_input=None):
        inputs = []

        if text_input:
            text_emb = self.language_model.embed(text_input)
            inputs.append(text_emb)

        if image_input:
            img_emb = self.vision_encoder(image_input)
            img_emb = self.projection(img_emb)  # LLM 공간으로 정렬
            inputs.append(img_emb)

        # 통합 표현으로 추론
        return self.language_model.forward(torch.cat(inputs))
```

---

## ⚡ 행동 모듈 (Action)

에이전트가 세상에 영향을 미치는 방법:

### 1. 도구 사용 (Tool Use)

```
에이전트가 활용할 수 있는 도구 유형:

정보 검색:
├─ 검색 엔진 (Google, Bing)
├─ 위키피디아 API
└─ 학술 DB (Arxiv, PubMed)

계산/실행:
├─ 코드 인터프리터 (Python 실행)
├─ 계산기/수학 엔진
└─ SQL 데이터베이스 쿼리

외부 서비스:
├─ 이메일/캘린더 API
├─ 파일 시스템 조작
└─ 웹 브라우저 자동화

전문 도구:
├─ 분자 시뮬레이터 (화학 에이전트)
├─ 코드 테스트 프레임워크
└─ 3D 렌더링 엔진
```

**Toolformer/Function Calling 패턴:**

```python
# OpenAI Function Calling 예시
tools = [
    {
        "name": "search_web",
        "description": "웹에서 정보 검색",
        "parameters": {
            "query": {"type": "string", "description": "검색 쿼리"}
        }
    },
    {
        "name": "execute_code",
        "description": "Python 코드 실행",
        "parameters": {
            "code": {"type": "string", "description": "실행할 코드"}
        }
    },
    {
        "name": "read_file",
        "description": "파일 내용 읽기",
        "parameters": {
            "path": {"type": "string", "description": "파일 경로"}
        }
    }
]

# 에이전트가 도구 선택 및 호출
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "2024년 노벨 물리학상 수상자는?"}],
    tools=tools,
    tool_choice="auto"  # 에이전트가 필요한 도구 자율 선택
)
```

### 2. 구현 행동 (Embodied Action)

```
가상 환경:
├─ Minecraft: 오픈 월드 탐험/건설
│   └─ 예: VOYAGER - 자율 스킬 발견 및 습득
├─ 시뮬레이션 마을: 사회적 시뮬레이션
└─ 웹 브라우저: UI 자동화

실제 로봇:
├─ 로봇 팔: 물체 조작
├─ 이동 로봇: 공간 탐색
└─ 드론: 항공 임무

제어 방식:
├─ SayCan: LLM 계획 + 로봇 실행 가능성 검증
├─ 계층적 제어: 고수준 계획 → 저수준 모터 제어
└─ 자연어 명령: "오른쪽 사과를 집어줘"
```

---

## 🎯 단일 에이전트 응용

### 1. 태스크 지향 (Task-Oriented)

**웹 자동화:**
```
사용자: "항공권 예약해줘, 서울→뉴욕, 다음 주 금요일"

에이전트 실행 흐름:
Step 1: 브라우저 열기
Step 2: 항공사 사이트 접속
Step 3: 날짜/경로 입력
Step 4: 최적 항공편 선택 (가격/시간 기준)
Step 5: 결제 정보 입력
Step 6: 예약 확인 및 사용자 통보
```

**소프트웨어 개발:**

| 도구 | 기능 | 특징 |
|------|------|------|
| **AutoGPT** | 자율 작업 실행 | 장기 목표 추구 |
| **GPT-Engineer** | 코드베이스 생성 | 스펙 → 전체 프로젝트 |
| **Devin** | 소프트웨어 엔지니어 | 이슈 해결, PR 작성 |
| **SWE-agent** | GitHub 이슈 해결 | 실제 저장소 조작 |

### 2. 혁신 지향 (Innovation-Oriented)

**과학적 발견:**

```python
# ChemCrow 스타일 화학 에이전트
class ChemistryAgent:
    """
    화학 연구 보조 에이전트
    - 분자 특성 예측
    - 합성 경로 설계
    - 실험 계획 수립
    """

    def __init__(self):
        self.tools = {
            "mol_simulator": MolecularDynamicsSimulator(),
            "reaction_db": ChemicalReactionDatabase(),
            "property_predictor": MLPropertyPredictor(),
            "literature_search": PubChemSearch()
        }

    def design_synthesis(self, target_molecule: str) -> dict:
        # Step 1: 문헌 검색
        known_routes = self.tools["literature_search"].search(target_molecule)

        # Step 2: LLM으로 새로운 합성 경로 제안
        proposed_route = self.llm.reason(
            f"목표 분자: {target_molecule}\n"
            f"알려진 경로: {known_routes}\n"
            "새로운 효율적 합성 경로를 제안하라."
        )

        # Step 3: 시뮬레이션 검증
        feasibility = self.tools["mol_simulator"].validate(proposed_route)

        return {"route": proposed_route, "feasibility": feasibility}
```

**수학 증명:**
```
예: AlphaCode, Lean 언어를 이용한 형식 수학 증명
- LLM이 증명 전략 수립
- 정리 증명기(Lean, Coq)로 형식적 검증
- 반례 발견 시 전략 수정
```

### 3. 수명 주기 지향 (Lifecycle-Oriented)

```
자율 에이전트의 장기적 자기 발전:

BabyAGI / VOYAGER 패턴:

목표 설정
  ↓
태스크 큐 생성 및 우선순위 결정
  ↓
태스크 실행 → 결과 저장
  ↓
새 태스크 생성 (결과 기반) + 스킬 라이브러리 업데이트
  ↓ (반복)
[목표 달성]
```

---

## 👥 멀티 에이전트 시스템

여러 에이전트가 협력하여 단일 에이전트의 한계를 극복하는 방식.

### 왜 멀티 에이전트인가?

```
단일 에이전트 한계:
├─ 컨텍스트 길이 제한 (초장문 작업 불가)
├─ 병렬 처리 불가 (순차 실행)
├─ 단일 관점 편향 (echo chamber)
└─ 전문성의 부재 (모든 것에 평균적)

멀티 에이전트 장점:
├─ 병렬 작업 처리 (속도 향상)
├─ 역할 분담 (전문화)
├─ 상호 검증 (오류 감소)
└─ 집단 지성 (더 나은 결정)
```

### 1. 협력적 멀티 에이전트

**MetaGPT: 소프트웨어 회사 시뮬레이션**

```
역할 구성 (표준 운영 절차 SOP 기반):

제품 관리자  →  아키텍트  →  프로젝트 매니저
    ↓               ↓               ↓
[요구사항]      [설계 문서]       [작업 분배]
                                     ↓
QA 엔지니어  →  엔지니어 B  →  엔지니어 A
    ↓               ↓               ↓
[테스트]        [코드 리뷰]      [코드 작성]
```

**CAMEL: 역할극 기반 협력**

```python
# CAMEL 프레임워크 예시
class CAMELFramework:
    """
    두 에이전트가 역할을 맡아 협력하는 프레임워크
    - AI Assistant: 태스크 수행
    - AI User: 지시 및 피드백
    """

    def __init__(self, task: str):
        self.task = task
        self.assistant = Agent(role="AI 어시스턴트", llm=GPT4())
        self.user_agent = Agent(role="AI 유저", llm=GPT4())
        self.history = []

    def run(self, max_turns: int = 20):
        # 초기 명령 생성
        instruction = self.user_agent.generate_instruction(self.task)

        for turn in range(max_turns):
            # 어시스턴트 응답
            response = self.assistant.respond(instruction)
            self.history.append(("Assistant", response))

            # 종료 조건 확인
            if "<TASK_DONE>" in response:
                break

            # 유저 에이전트 피드백
            instruction = self.user_agent.give_feedback(response)
            self.history.append(("User", instruction))

        return self.history
```

**AgentVerse: 동적 팀 구성**

```
AgentVerse 특징:
├─ 동적 에이전트 모집 (태스크에 맞는 전문가 선택)
├─ 협력 실행 (병렬/순차 작업)
├─ 민주적 의사결정
└─ 팀 반성 및 최적화
```

### 2. 대립적 멀티 에이전트

**토론 프레임워크 (Debate):**

```
사회자 (Moderator)
  ├─→ 에이전트 A (찬성 측)
  └─→ 에이전트 B (반대 측)
            ↓
       논쟁 / 반박 / 검토
            ↓
       최종 합의 도출

효과:
├─ 편향 감소 (단일 모델의 echo chamber 방지)
├─ 추론 품질 향상 (상호 검증)
└─ 불확실성 교정 (너무 확신하는 답변 방지)
```

### 3. 에이전트 간 통신

```
통신 방식 분류:

직접 통신 (Point-to-Point):
Agent A ──→ Agent B
(특정 에이전트에게 직접 메시지)

브로드캐스트 (Broadcast):
Agent A ──→ [Agent B, Agent C, Agent D, ...]
(모든 에이전트에게 공유)

공유 메모리 (Shared Memory):
Agent A, B, C ──→ [공유 메모리 공간] ←── 읽기
(블랙보드 패턴)

메시지 형식:
├─ 자연어: "다음 단계는 DB를 쿼리하세요"
├─ 구조화 포맷: JSON, XML
└─ 코드: 직접 함수 호출
```

---

## 🤝 인간-에이전트 협력

에이전트가 인간과 상호작용하는 다양한 패러다임.

### 1. 지시자-실행자 패러다임

```
교육 분야:
Human Teacher ──→ [교육 계획 설계]
                       ↓
           AI Agent ──→ [맞춤형 강의 제공]
                           ↓
           Human Student ←── [지식 습득]
                           ↓
           AI Agent ──→ [성취도 평가 및 피드백]
                           ↓ (반복)

의료 분야:
Human Doctor ──→ [진단 요청]
                      ↓
          AI Agent ──→ [의료 이미지 분석, 문헌 검색]
                            ↓
          Human Doctor ←── [추천 진단 결과]
                            ↓ (최종 결정은 의사)
```

### 2. 동등한 파트너 패러다임

```
공감적 협력:
├─ 인간의 감정 상태 인식
├─ 대화 스타일 적응
└─ 적절한 공감 표현

협상 (Negotiation):
├─ 에이전트가 자신의 관점 제시
├─ 인간의 반론 수용 및 반박
└─ 상호 이익의 합의점 탐색

혼합 주도권 (Mixed-Initiative):
Human: 방향 결정 + 창의적 결정
Agent: 세부 실행 + 정보 제공
```

### 3. RLHF와 피드백 통합

```python
# 인간 피드백을 통한 에이전트 개선
class HumanFeedbackLoop:
    def __init__(self, agent):
        self.agent = agent
        self.feedback_history = []

    def get_feedback(self, agent_output: str) -> dict:
        """인간에게 피드백 요청"""
        print(f"에이전트 출력: {agent_output}")
        rating = int(input("평점 (1-5): "))
        comment = input("코멘트: ")
        return {"rating": rating, "comment": comment, "output": agent_output}

    def train_on_feedback(self):
        """피드백 기반 에이전트 개선"""
        positive_examples = [
            f["output"] for f in self.feedback_history if f["rating"] >= 4
        ]
        negative_examples = [
            f["output"] for f in self.feedback_history if f["rating"] <= 2
        ]

        # RLHF / DPO 등으로 모델 개선
        self.agent.fine_tune(positive_examples, negative_examples)
```

---

## 🌐 에이전트 사회

다수의 에이전트가 상호작용하여 사회적 현상을 만들어내는 연구.

### Stanford Town: 생성적 에이전트 시뮬레이션

가장 주목받는 연구 중 하나: **25명의 AI 에이전트**가 작은 마을에서 생활하는 시뮬레이션.

```
마을 구성:
├─ 에이전트 수: 25명
├─ 직업: 교사, 의사, 카페 주인, 예술가 등
├─ 환경: 집, 카페, 학교, 공원 등이 있는 마을
└─ 시간: 하루 생활 시뮬레이션

핵심 메커니즘 (에이전트 인지 루프):

현재 상태 관찰
  ↓
기억 검색 (관련 과거 경험)
  ↓
반성 (Reflection): 고수준 통찰 생성
  ↓
계획 수립 (오늘 무엇을 할지)
  ↓
행동 실행 및 상호작용
  ↓
기억 저장 → (다음 루프)

창발적 사회 현상:
├─ 자발적 관계 형성 (친구, 연인)
├─ 정보 전파 (소문 퍼지기)
├─ 이벤트 조직 (발렌타인 파티 자발적 기획)
└─ 협력과 갈등
```

### 에이전트 사회의 구성 요소

```
사회적 행동:
├─ 협력 (Cooperation): 공동 목표 달성
├─ 경쟁 (Competition): 자원/지위 경쟁
├─ 협상 (Negotiation): 이해관계 조율
└─ 규범 형성 (Norm Formation): 암묵적 규칙 생성

개성과 성격:
├─ Big Five 성격 모델 적용
│   (개방성, 성실성, 외향성, 우호성, 신경증)
├─ 에이전트별 독자적 가치관
└─ 일관된 행동 패턴 유지

사회적 역학:
├─ 정보 확산: SNS 모의, 가짜 뉴스 전파 시뮬레이션
├─ 의견 형성: 집단 극화, 합의 형성
├─ 도덕적 의사결정: 윤리 딜레마 상황
└─ 권력 구조: 리더십 자연 발현
```

### 에이전트 사회의 환경 유형

| 환경 | 예시 | 특징 |
|------|------|------|
| **텍스트 기반** | MUD, 텍스트 RPG | 언어로만 상호작용 |
| **샌드박스** | Minecraft, 시뮬레이션 마을 | 반구조적 세계 |
| **물리적 환경** | 실제 로봇 세계 | Sim-to-Real 도전 |
| **하이브리드** | 현실+가상 혼합 | 최근 연구 방향 |

---

## 📊 벤치마크 및 평가

LLM 에이전트를 어떻게 평가할 것인가?

### 주요 벤치마크

```
태스크 완료 평가:
├─ WebArena: 실제 웹 환경에서의 작업 수행
├─ SWE-bench: GitHub 이슈 해결
├─ AgentBench: 다양한 환경에서 에이전트 평가
└─ MINT: 멀티턴 상호작용 평가

추론 능력:
├─ HotpotQA: 다단계 추론 QA
├─ ALFWorld: 가상 환경 탐색
└─ FEVER: 사실 검증

소셜 능력:
├─ 협상 게임 성과
├─ 설득 성공률
└─ 공감 표현 평가
```

### 평가 차원

```
다차원 평가 프레임워크:

1. 유용성 (Utility)
   └─ 목표 달성률, 작업 완료도

2. 효율성 (Efficiency)
   └─ 단계 수, 토큰 비용, 시간

3. 사회성 (Sociability)
   └─ 대화 자연스러움, 적절한 협력

4. 가치 정렬 (Value Alignment)
   └─ 윤리적 행동, 안전성

5. 강건성 (Robustness)
   └─ 오류 복구, 적대적 입력 대응
```

---

## ⚠️ 핵심 도전 과제

### 1. 환각과 신뢰성

```
문제:
├─ 사실이 아닌 정보 생성 (Hallucination)
├─ 도구 호출 결과를 무시하고 자체 생성
└─ 확신에 찬 오답

현재 접근법:
├─ RAG: 실시간 외부 지식 검색
├─ Self-Consistency: 여러 번 답변 후 다수결
├─ 도구 의존 설계: 계산은 코드로, 검색은 검색기로
└─ 팩트 검증 에이전트 추가
```

### 2. 장기 계획과 일관성

```
문제:
├─ 긴 에이전트 루프에서 목표 표류 (Goal Drift)
├─ 초기 계획과 실행 불일치
└─ 오류 누적 (Error Propagation)

해결 방향:
├─ 주기적 목표 재확인 (Goal Grounding)
├─ 서브골 분해 및 추적
└─ 외부 계획 관리자 도입
```

### 3. 안전성과 자율성의 균형

```
딜레마:
높은 자율성 ←────────────→ 높은 안전성
   장점: 강력한 문제 해결     장점: 예측 가능, 안전
   단점: 예기치 않은 행동     단점: 유연성 제한

위험 시나리오:
├─ 악의적 명령 실행 (Prompt Injection)
├─ 개인정보 유출
├─ 의도치 않은 부작용 (파일 삭제 등)
└─ 창발적 위험 행동

안전장치:
├─ Human-in-the-loop: 중요 결정 시 인간 확인
├─ 권한 제한: 최소 권한 원칙
├─ 행동 로깅: 모든 행동 기록
└─ 레드라인 설정: 절대 금지 행동 목록
```

### 4. 멀티 에이전트 조율

```
문제:
├─ 에이전트 간 충돌 (목표, 자원)
├─ 통신 비용 증가
├─ 무임승차 (Free-rider) 문제
└─ 계산 복잡도 (에이전트 수 증가 시)

해결 방향:
├─ 명확한 역할 정의 및 SOP
├─ 중앙 조율자 (Orchestrator) 도입
└─ 분산 합의 프로토콜
```

### 5. 평가의 어려움

```
벤치마크 한계:
├─ 실제 환경과의 괴리 (실험실 vs 현실)
├─ 다차원 성능의 단일 지표 압축 불가
├─ 사회적/윤리적 능력 정량화 어려움
└─ 새로운 태스크에 대한 일반화 측정 미흡

필요한 것:
├─ 실제 사용자와의 상호작용 평가
├─ 장기적 (수개월) 에이전트 성능 추적
└─ 도메인별 전문 벤치마크
```

---

## 💻 실사용 예시

### 실용적인 LLM 에이전트 구현

```python
"""
종합적인 LLM 에이전트 구현 예시
Brain-Perception-Action 프레임워크 기반
"""
import openai
import json
from typing import List, Dict, Any
from datetime import datetime


class LLMAgent:
    """Brain-Perception-Action 프레임워크 기반 에이전트"""

    def __init__(self, name: str, role: str, llm_model: str = "gpt-4"):
        self.name = name
        self.role = role
        self.llm = openai.OpenAI()
        self.model = llm_model

        # 메모리 시스템
        self.short_term_memory = []      # 현재 대화 컨텍스트
        self.long_term_memory = []       # 벡터 DB (단순화)
        self.working_memory = {}         # 현재 태스크 상태

        # 도구 등록
        self.tools = self._register_tools()

    def _register_tools(self) -> List[Dict]:
        """사용 가능한 도구 등록"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "웹에서 최신 정보를 검색합니다",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "검색 쿼리"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "description": "Python 코드를 실행하고 결과를 반환합니다",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "실행할 Python 코드"
                            }
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "store_memory",
                    "description": "중요한 정보를 장기 메모리에 저장합니다",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "category": {"type": "string"}
                        },
                        "required": ["content", "category"]
                    }
                }
            }
        ]

    # === 지각 (Perception) ===
    def perceive(self, user_input: str, context: Dict = None) -> str:
        """입력 처리 및 컨텍스트 통합"""
        perception = f"[입력] {user_input}"
        if context:
            perception += f"\n[컨텍스트] {json.dumps(context, ensure_ascii=False)}"

        # 관련 장기 메모리 검색 (단순 키워드 매칭)
        relevant_memories = [
            m for m in self.long_term_memory
            if any(word in m["content"] for word in user_input.split())
        ]
        if relevant_memories:
            perception += f"\n[관련 기억] {relevant_memories[:3]}"

        return perception

    # === 두뇌 (Brain) - 추론 및 계획 ===
    def think_and_plan(self, perceived_input: str) -> str:
        """ReAct 스타일 추론 및 계획"""
        system_prompt = f"""당신은 {self.role} 역할을 하는 AI 에이전트입니다.
현재 시간: {datetime.now().strftime("%Y-%m-%d %H:%M")}

다음 단계로 추론하세요:
1. Thought: 상황을 분석하고 필요한 것을 파악
2. Action: 실행할 행동 결정 (도구 사용 또는 직접 답변)
3. Observation: 결과 관찰
4. Reflection: 진행 상황 평가
"""
        self.short_term_memory.append({
            "role": "user",
            "content": perceived_input
        })

        messages = [
            {"role": "system", "content": system_prompt}
        ] + self.short_term_memory[-10:]  # 최근 10개 메시지만 유지

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )

        return response

    # === 행동 (Action) ===
    def act(self, llm_response) -> str:
        """LLM 결정에 따른 행동 실행"""
        message = llm_response.choices[0].message

        # 도구 호출이 없는 경우 직접 답변
        if not message.tool_calls:
            return message.content

        # 도구 호출 실행
        results = []
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            result = self._execute_tool(tool_name, tool_args)
            results.append(f"[{tool_name} 결과]: {result}")

        # 도구 결과를 바탕으로 최종 응답 생성
        self.short_term_memory.append({
            "role": "assistant",
            "content": f"도구 실행: {results}"
        })

        final_response = self.llm.chat.completions.create(
            model=self.model,
            messages=self.short_term_memory + [
                {"role": "user", "content": f"도구 실행 결과: {results}\n최종 답변을 작성하세요."}
            ]
        )

        return final_response.choices[0].message.content

    def _execute_tool(self, tool_name: str, args: Dict) -> str:
        """도구 실행 (실제 구현에서는 각 도구 연결)"""
        if tool_name == "search_web":
            return f"[검색 결과] '{args['query']}'에 대한 검색 결과..."  # 실제: API 호출
        elif tool_name == "execute_python":
            return f"[실행 결과] 코드 실행 완료"  # 실제: sandbox 실행
        elif tool_name == "store_memory":
            self.long_term_memory.append({
                "content": args["content"],
                "category": args["category"],
                "timestamp": datetime.now().isoformat()
            })
            return "메모리 저장 완료"
        return "알 수 없는 도구"

    # === 메인 루프 ===
    def run(self, user_input: str, context: Dict = None) -> str:
        """에이전트 실행 메인 루프"""
        # 1. 지각
        perceived = self.perceive(user_input, context)

        # 2. 추론 및 계획
        plan = self.think_and_plan(perceived)

        # 3. 행동 실행
        result = self.act(plan)

        # 4. 메모리 업데이트
        self.short_term_memory.append({
            "role": "assistant",
            "content": result
        })

        return result


# === 멀티 에이전트 시스템 ===
class MultiAgentSystem:
    """멀티 에이전트 협력 시스템"""

    def __init__(self):
        self.agents = {
            "researcher": LLMAgent("연구원", "정보 조사 및 분석 전문가"),
            "coder": LLMAgent("개발자", "코드 작성 및 디버깅 전문가"),
            "reviewer": LLMAgent("검토자", "품질 검토 및 피드백 전문가"),
        }
        self.shared_memory = {}

    def run_pipeline(self, task: str) -> str:
        """파이프라인 방식 협력"""
        print(f"[태스크 시작] {task}\n")

        # Step 1: 연구원 - 정보 수집
        research_result = self.agents["researcher"].run(
            f"다음 태스크를 위한 정보를 조사하세요: {task}"
        )
        print(f"[연구원] {research_result[:200]}...\n")
        self.shared_memory["research"] = research_result

        # Step 2: 개발자 - 구현
        code_result = self.agents["coder"].run(
            f"다음 조사 결과를 바탕으로 구현하세요:\n{research_result}\n태스크: {task}"
        )
        print(f"[개발자] {code_result[:200]}...\n")
        self.shared_memory["code"] = code_result

        # Step 3: 검토자 - 품질 검토
        review_result = self.agents["reviewer"].run(
            f"다음 결과물을 검토하고 개선사항을 제안하세요:\n{code_result}"
        )
        print(f"[검토자] {review_result[:200]}...\n")

        return review_result


# 사용 예시
if __name__ == "__main__":
    # 단일 에이전트
    agent = LLMAgent(
        name="어시스턴트",
        role="데이터 분석 및 코딩 전문 AI 에이전트"
    )

    response = agent.run(
        "파이썬으로 피보나치 수열의 첫 20개를 계산하고, 그 중 소수를 찾아줘"
    )
    print(response)

    # 멀티 에이전트
    mas = MultiAgentSystem()
    result = mas.run_pipeline("머신러닝으로 주식 가격 예측 모델 구현")
    print(result)
```

### LangGraph를 이용한 상태 기반 에이전트

```python
"""
LangGraph로 구현하는 ReAct 에이전트
노드(Node)와 엣지(Edge)로 에이전트 흐름 제어
"""
from langgraph.graph import StateGraph, END
from typing import TypedDict, List


class AgentState(TypedDict):
    messages: List[dict]
    thought: str
    action: str
    observation: str
    final_answer: str
    iteration: int


def think_node(state: AgentState) -> AgentState:
    """추론 단계"""
    # LLM으로 현재 상황 분석
    thought = llm.invoke(
        f"지금까지 관찰: {state['observation']}\n"
        f"다음 행동을 결정하기 위해 추론하세요."
    )
    return {**state, "thought": thought, "iteration": state["iteration"] + 1}


def act_node(state: AgentState) -> AgentState:
    """행동 결정 단계"""
    action = llm.invoke(
        f"추론: {state['thought']}\n"
        f"실행할 도구와 입력을 JSON으로 반환하세요."
    )
    return {**state, "action": action}


def observe_node(state: AgentState) -> AgentState:
    """관찰 단계"""
    action_dict = json.loads(state["action"])
    if action_dict["tool"] == "search":
        observation = search_tool(action_dict["input"])
    elif action_dict["tool"] == "calculate":
        observation = calc_tool(action_dict["input"])
    elif action_dict["tool"] == "finish":
        return {**state, "final_answer": action_dict["answer"]}
    return {**state, "observation": observation}


def should_continue(state: AgentState) -> str:
    """계속 반복할지 종료할지 결정"""
    if state.get("final_answer") or state["iteration"] >= 10:
        return "end"
    return "continue"


# 그래프 구성
workflow = StateGraph(AgentState)
workflow.add_node("think", think_node)
workflow.add_node("act", act_node)
workflow.add_node("observe", observe_node)

workflow.set_entry_point("think")
workflow.add_edge("think", "act")
workflow.add_edge("act", "observe")
workflow.add_conditional_edges(
    "observe",
    should_continue,
    {"continue": "think", "end": END}
)

agent = workflow.compile()

# 실행
result = agent.invoke({
    "messages": [{"role": "user", "content": "오늘 서울 날씨는?"}],
    "thought": "",
    "action": "",
    "observation": "",
    "final_answer": "",
    "iteration": 0
})
print(result["final_answer"])
```

---

## 🔮 향후 연구 방향

논문이 제시하는 미래 연구 과제들:

```
1. 장기 기억과 연속 학습
   ├─ 수개월~수년 단위의 에이전트 경험 보존
   └─ 새로운 지식 학습 시 기존 지식 망각 방지 (Catastrophic Forgetting)

2. 효율적인 멀티 에이전트 조율
   ├─ 수천 개 에이전트의 확장 가능한 조율
   └─ 최적 역할 분배 자동화

3. 실세계 적용 (Sim-to-Real)
   ├─ 시뮬레이션 능력을 현실 로봇에 이전
   └─ 현실의 불확실성과 노이즈 처리

4. 가치 정렬과 윤리
   ├─ 다양한 문화/가치관을 가진 에이전트 설계
   └─ 에이전트 사회에서 윤리 규범 자동 형성

5. 에이전트 경제학
   ├─ 에이전트 간 자원 할당 최적화
   └─ 인센티브 설계 (게임 이론 적용)

6. 인간-에이전트 신뢰 구축
   ├─ 에이전트 행동의 설명 가능성
   └─ 점진적 자율성 확대 메커니즘
```

---

## 🎓 핵심 교훈

1. **LLM은 에이전트의 뇌**: 지식, 추론, 계획 능력을 하나의 모델로 통합
2. **도구 사용이 능력을 확장**: LLM의 내재적 한계를 외부 도구로 보완
3. **메모리 설계가 핵심**: 단기-장기 메모리 체계가 에이전트의 일관성을 결정
4. **멀티 에이전트는 집단 지성**: 단일 에이전트의 한계를 역할 분담과 상호 검증으로 극복
5. **안전성은 처음부터**: 에이전트 설계 단계부터 안전장치 내재화 필요
6. **평가가 연구를 이끈다**: 좋은 벤치마크 없이는 진보를 측정할 수 없음

---

## 📖 참고 자료

- **원 논문**: https://arxiv.org/abs/2309.07864
- **논문 목록 (GitHub)**: https://github.com/WooooDyy/LLM-Agent-Paper-List
- **관련 서베이 (자율 에이전트)**: https://arxiv.org/abs/2308.11432
- **LangChain**: https://python.langchain.com/docs/modules/agents/
- **LangGraph**: https://langchain-ai.github.io/langgraph/
- **AutoGPT**: https://github.com/Significant-Gravitas/AutoGPT
- **MetaGPT**: https://github.com/geekan/MetaGPT
- **AgentBench**: https://github.com/THUDM/AgentBench

---

**이 서베이는 LLM 기반 에이전트 연구의 현황을 가장 체계적으로 정리한 레퍼런스 논문으로, 에이전트 관련 연구를 시작하는 모든 연구자와 엔지니어에게 필독 자료입니다.**
