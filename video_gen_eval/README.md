# Educational Video Generation CLI

교육용 AI 비디오를 생성하기 위한 CLI 도구입니다.

## 개요

이 도구는 다양한 AI 비디오 생성 모델을 통해 교육용 비디오를 쉽게 생성할 수 있습니다.

### 지원 모델

| 모델 | 모드 | GPU | API | 설명 |
|------|------|-----|-----|------|
| **Sora 2** | API | - | fal.ai | OpenAI 비디오 생성 모델 |
| **Veo 3** | API | - | fal.ai | Google DeepMind 최신 모델 |
| **Wan 2.2** | 로컬 | 필요 | - | Alibaba 오픈소스 (1.3B/5B/14B) |

---

## 설치

### Step 1: 저장소 클론

```bash
git clone <ANONYMOUS_REPO_URL>
cd EduVideoBench-anonymous/video_gen_eval
```

### Step 2: 의존성 설치

```bash
pip install -r requirements.txt
```

### Step 3: 환경변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# API 키 설정
nano .env  # 또는 vi, code 등
```

**.env 설정:**
```env
# fal.ai API 키 (Sora 2, Veo 3 사용)
# https://fal.ai/dashboard/keys 에서 발급
FAL_KEY=your_fal_api_key_here

# GPU 모델 활성화 (Wan 2.2 사용 시)
ENABLE_GPU_MODELS=false
```

---

## 사용법

### 기본 명령어

```bash
# 도움말 보기
python main.py --help

# 사용 가능한 모델 확인
python main.py status
```

### 단일 비디오 생성

```bash
# 기본 사용
python main.py generate --prompt "피타고라스 정리를 설명하는 애니메이션" --model sora2

# 옵션 지정
python main.py generate \
  --prompt "이차방정식 풀이 과정" \
  --model veo3 \
  --duration 20 \
  --resolution 1080p \
  --output ./my_videos
```

### 교육용 비디오 생성

문제를 입력하면 자동으로 교육용 프롬프트를 생성합니다.

```bash
python main.py edu \
  --problem "직각삼각형에서 빗변의 길이를 구하시오. 두 변의 길이는 3cm, 4cm" \
  --subject math \
  --grade "middle school" \
  --model sora2
```

### 배치 생성

JSON 파일로 여러 비디오를 일괄 생성합니다.

```bash
# examples/batch_input.json 파일 사용
python main.py batch --input examples/batch_input.json --model veo3

# 출력 디렉토리 지정
python main.py batch --input problems.json --output ./outputs --model sora2
```

**배치 입력 파일 형식 (JSON):**
```json
[
  {
    "title": "피타고라스 정리",
    "problem_text": "직각삼각형에서 한 변이 3cm, 다른 변이 4cm일 때 빗변의 길이는?",
    "subject": "math",
    "grade_level": "middle school",
    "duration": 30
  },
  {
    "title": "광합성 설명",
    "prompt": "식물의 광합성 과정을 보여주는 애니메이션",
    "duration": 20
  }
]
```

---

## 설정

### config.yaml

`config.yaml` 파일로 기본값을 설정할 수 있습니다:

```yaml
# 출력 디렉토리
output_dir: "./outputs"

# 기본 모델
default_model: "sora2"

# 비디오 기본 설정
video:
  duration: 10
  resolution: "1080p"
  aspect_ratio: "16:9"

# 모델별 설정
models:
  sora2:
    timeout: 600
  veo3:
    timeout: 600
    generate_audio: true
  wan22:
    model_size: "1.3B"
```

---

## 출력 구조

스크립트 실행 시 타임스탬프 폴더가 생성되고, 그 안에 비디오와 메타데이터가 저장됩니다.

```
outputs/
└── 20260106_143052/              # 타임스탬프 폴더
    ├── videos/                    # 비디오 파일 폴더
    │   ├── 001_sora2_pythagorean.mp4
    │   ├── 002_sora2_quadratic.mp4
    │   └── 003_sora2_photosynthesis.mp4
    └── metadata.tsv               # 메타정보 파일
```

### metadata.tsv 파일 형식

| 컬럼 | 설명 |
|------|------|
| index | 비디오 번호 |
| filename | 파일명 |
| model | 사용된 모델 |
| prompt | 생성 프롬프트 |
| title | 비디오 제목/라벨 |
| subject | 과목 |
| grade_level | 학년 수준 |
| duration | 비디오 길이 (초) |
| resolution | 해상도 |
| aspect_ratio | 화면 비율 |
| success | 생성 성공 여부 |
| generation_time_sec | 생성 소요 시간 |
| video_url | 원본 URL |
| error | 에러 메시지 |

---

## 프로젝트 구조

```
video_gen_eval/
├── main.py                 # CLI 진입점
├── config.yaml             # 설정 파일
├── requirements.txt        # 의존성
├── .env.example            # 환경변수 템플릿
├── video_generators/       # 비디오 생성 모델
│   ├── base.py             # 기본 클래스
│   ├── factory.py          # 팩토리 패턴
│   ├── sora/               # Sora 2 (fal.ai)
│   ├── veo/                # Veo 3 (fal.ai)
│   └── wan/                # Wan 2.2 (로컬 GPU)
├── examples/               # 예제 파일
│   └── batch_input.json    # 배치 입력 예제
└── outputs/                # 생성된 비디오 출력
    └── YYYYMMDD_HHMMSS/    # 타임스탬프 폴더
        ├── videos/         # 비디오 파일
        └── metadata.tsv    # 메타데이터
```

---

## CLI 명령어 레퍼런스

### generate

단일 비디오 생성

```
python main.py generate [OPTIONS]

Options:
  -p, --prompt TEXT        비디오 생성 프롬프트 (필수)
  -m, --model TEXT         모델 선택: sora2, veo3, wan22 (기본: sora2)
  -d, --duration INT       비디오 길이 (초, 기본: 10)
  -r, --resolution TEXT    해상도: 480p, 720p, 1080p (기본: 1080p)
  -a, --aspect-ratio TEXT  화면 비율 (기본: 16:9)
  -i, --image PATH         이미지-투-비디오용 입력 이미지
  -l, --label TEXT         비디오 파일명 라벨
  -o, --output PATH        출력 기본 디렉토리
```

### edu

교육용 비디오 생성

```
python main.py edu [OPTIONS]

Options:
  -p, --problem TEXT       문제 텍스트 (필수)
  -s, --subject TEXT       과목 (기본: math)
  -g, --grade TEXT         학년 수준 (기본: middle school)
  -m, --model TEXT         모델 선택 (기본: sora2)
  -d, --duration INT       비디오 길이 (초, 기본: 30)
  -r, --resolution TEXT    해상도 (기본: 1080p)
  -l, --label TEXT         비디오 파일명 라벨
  -o, --output PATH        출력 기본 디렉토리
```

### batch

배치 비디오 생성

```
python main.py batch [OPTIONS]

Options:
  -i, --input PATH         입력 JSON 파일 (필수)
  -m, --model TEXT         모델 선택 (기본: sora2)
  -d, --duration INT       기본 비디오 길이 (초, 기본: 10)
  -r, --resolution TEXT    기본 해상도 (기본: 1080p)
  -o, --output PATH        출력 기본 디렉토리
```

### status

사용 가능한 생성기 확인

```
python main.py status
```

---

## 참고 자료

- [fal.ai](https://fal.ai) - Unified Video Generation API
- [Sora 2](https://fal.ai/models/fal-ai/sora-2) - OpenAI Sora 2
- [Veo 3](https://fal.ai/models/fal-ai/veo3) - Google Veo 3
- [Wan 2.2](https://github.com/Wan-Video/Wan2.2) - Alibaba Wan 2.2

## 라이선스

Apache-2.0 License
