# Diffusers API 테스트 가이드

## 서버 실행

```bash
cd cv.lagzang.com/app/diffusers
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API 엔드포인트

### 1. Health Check

```bash
curl http://localhost:8000/health
```

### 2. 이미지 생성

```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cute robot barista, cinematic lighting",
    "width": 768,
    "height": 768,
    "steps": 4,
    "seed": 42
  }'
```

#### 요청 파라미터

- `prompt` (필수): 이미지 생성 프롬프트
- `negative_prompt` (선택): 제외할 요소
- `width` (선택): 이미지 너비 (기본값: 768)
- `height` (선택): 이미지 높이 (기본값: 768)
- `steps` (선택): 추론 스텝 수 (기본값: 4)
- `guidance_scale` (선택): 가이던스 스케일 (기본값: 0.0)
- `seed` (선택): 랜덤 시드

#### 응답 예시

```json
{
  "id": "abc123...",
  "image_url": "/outputs/images/abc123....png",
  "meta_url": "/outputs/metadata/abc123....json",
  "meta": {
    "id": "abc123...",
    "created_at": "2024-01-01T00:00:00Z",
    "model_id": "stabilityai/sdxl-turbo",
    "prompt": "a cute robot barista, cinematic lighting",
    "width": 768,
    "height": 768,
    "steps": 4,
    "guidance_scale": 0.0,
    "seed": 42,
    "device": "cuda",
    "image_file": "abc123....png",
    "meta_file": "abc123....json"
  }
}
```

### 3. 생성된 이미지 확인

생성된 이미지는 다음 URL로 접근할 수 있습니다:

```
http://localhost:8000/outputs/images/{image_file}
```

예시:
```
http://localhost:8000/outputs/images/abc123....png
```

### 4. 메타데이터 확인

생성된 메타데이터는 다음 URL로 접근할 수 있습니다:

```
http://localhost:8000/outputs/metadata/{meta_file}
```

예시:
```
http://localhost:8000/outputs/metadata/abc123....json
```

## 추가 예시

### 최소 요청 (프롬프트만)

```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful sunset over the ocean"
  }'
```

### 모든 파라미터 포함

```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a futuristic city at night, neon lights, cyberpunk style",
    "negative_prompt": "blurry, low quality, distorted",
    "width": 768,
    "height": 768,
    "steps": 6,
    "guidance_scale": 1.0,
    "seed": 12345
  }'
```

