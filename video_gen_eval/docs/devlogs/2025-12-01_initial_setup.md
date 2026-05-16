# 개발 기록: 초기 플랫폼 구조 설정

**날짜**: 2025-12-01
**작성자**: Claude
**버전**: 1.0.0

---

## 개요

교육용 AI 생성 비디오 평가 플랫폼의 초기 구조를 설계하고 구현했습니다.

## 주요 결정 사항

### 1. 평가 프레임워크 구조

연구 계획에 따라 3개 상위 차원, 14개 하위 차원으로 평가 체계를 구성:

- **A. 교육 문제 해결 역량** (4개): A-1 ~ A-4
- **B. 교수설계·비디오 설계 품질** (5개): B-1 ~ B-5
- **C. 학습자 경험 및 효과** (5개): C-1 ~ C-5

특히 C-1, C-2는 VLM 가상 학생의 응답을 전문가가 채점하는 방식으로 설계.

### 2. 비디오 생성 모델 지원

세 가지 모델을 지원하도록 설계:

| 모델 | 버전 | 접근 방식 | GPU 필요 |
|------|------|-----------|----------|
| Sora | 2.0 | OpenAI API | ❌ |
| Veo | 3.1 | Google Gemini API | ❌ |
| Wan | 2.2 | 로컬 추론 | ✅ |

**Wan 2.2 GPU 요구사항**:
- 1.3B: 8GB VRAM
- 5B: 16GB VRAM
- 14B: 24GB+ VRAM

GPU 미탑재 환경에서는 `ENABLE_GPU_MODELS=false`로 비활성화.

### 3. 아키텍처 결정

**Docker Compose 기반 마이크로서비스**:
- `nginx`: 리버스 프록시
- `frontend`: React/Vue SPA
- `backend`: FastAPI 서버
- `db`: PostgreSQL
- `redis`: 세션/캐시
- `minio`: 비디오 파일 저장소
- `vlm-server`: vLLM 기반 가상 학생 (GPU 프로파일)

**vLLM 서버**:
- GPU 프로파일로 분리 (`--profile gpu`)
- OpenAI-compatible API 제공
- 기본 모델: Qwen2-VL-7B-Instruct

### 4. 설정 관리

```
configs/
├── .env.template    # API 키, DB 설정 템플릿
└── .env            # 실제 설정 (gitignore)
```

환경변수로 API 키 관리:
- `OPENAI_API_KEY`: Sora 2
- `GOOGLE_API_KEY`: Veo 3.1
- `HUGGINGFACE_TOKEN`: Wan 2.2 모델 다운로드

## 구현 내용

### 비디오 생성 모듈 (`video_generators/`)

```python
# Factory 패턴으로 모델 생성
from video_generators import create_generator

generator = create_generator("sora2")  # or "veo31", "wan22"
result = await generator.generate_from_problem(
    problem_text="3x + 5 = 14를 풀어라",
    subject="math",
    grade_level="중학교 2학년"
)
```

### Backend API 구조

```
/api/v1/
├── auth/          # 인증
├── videos/        # 비디오 관리
├── evaluations/   # 평가 CRUD
│   ├── content/   # A 차원
│   ├── design/    # B 차원
│   └── learner/   # C 차원
├── vlm/           # VLM 가상 학생
├── rubrics/       # 루브릭 관리
├── reports/       # 리포트 생성
└── admin/         # 관리자 기능
```

### 데이터베이스 스키마

주요 테이블:
- `users`: 평가자/관리자 계정
- `videos`: 비디오 메타데이터
- `rubrics`: 평가 기준 정의
- `vlm_responses`: VLM 응답 (text_only, text_video)
- `evaluations`: 평가 결과
- `task_assignments`: 과제 배정

## 향후 작업

### 우선순위 높음
1. [ ] Backend 서비스 레이어 구현
2. [ ] DB 모델 및 마이그레이션
3. [ ] Frontend 기본 구조 및 라우팅
4. [ ] 인증 시스템 완성

### 우선순위 중간
1. [ ] 평가 UI 컴포넌트
2. [ ] VLM 서비스 연동
3. [ ] 리포트 생성 로직

### 우선순위 낮음
1. [ ] 실시간 진행률 업데이트 (WebSocket)
2. [ ] 다국어 지원
3. [ ] 접근성 개선

## 참고 자료

- [Thinking with Video Paper](https://arxiv.org/abs/2511.04570)
- [Veo 3.1 API Docs](https://developers.googleblog.com/en/introducing-veo-3-1-and-new-creative-capabilities-in-the-gemini-api/)
- [Wan 2.2 GitHub](https://github.com/Wan-Video/Wan2.2)

---

**다음 개발 기록**: Frontend UI 구조 및 평가 화면 구현
