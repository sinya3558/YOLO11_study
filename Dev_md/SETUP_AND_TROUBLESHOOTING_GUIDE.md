# 🛠️ YOLO11 Multi-Layer Detection 환경 설정 및 문제 해결 가이드

이 문서는 YOLO11 Multi-Layer Detection System의 완전한 실행 환경 구성과 일반적인 오류 해결 방법을 제공합니다.

---

## 📋 목차
1. [시스템 요구사항](#-시스템-요구사항)
2. [설치 가이드](#-설치-가이드)
3. [RGBA 컬러 인식 설정](#-rgba-컬러-인식-설정)
4. [환경 검증](#-환경-검증)
5. [일반적인 오류 해결](#-일반적인-오류-해결)
6. [성능 최적화](#-성능-최적화)
7. [플랫폼별 가이드](#-플랫폼별-가이드)

---

## 💻 시스템 요구사항

### 최소 요구사항
| 구성요소 | 최소 사양 | 권장 사양 |
|---------|---------|----------|
| **OS** | Windows 10, Ubuntu 20.04, macOS 11 | Windows 11, Ubuntu 22.04, macOS 12+ |
| **Python** | 3.8 | 3.10 - 3.11 |
| **RAM** | 8GB | 16GB 이상 |
| **GPU** | 없음 (CPU 모드) | NVIDIA GTX 1060 이상 |
| **GPU 메모리** | - | 6GB 이상 |
| **저장 공간** | 10GB | 20GB 이상 |
| **CUDA** | - | 11.7 이상 |
| **cuDNN** | - | 8.5 이상 |

### Python 버전 확인
```bash
python --version
# 또는
python3 --version
```

---

## 🚀 설치 가이드

### 1단계: Python 가상환경 생성

#### Windows
```bash
# venv 생성
python -m venv yolo_env

# 활성화
yolo_env\Scripts\activate

# 활성화 확인 (프롬프트에 (yolo_env) 표시)
```

#### Linux/macOS
```bash
# venv 생성
python3 -m venv yolo_env

# 활성화
source yolo_env/bin/activate

# 활성화 확인
which python
```

### 2단계: 기본 패키지 설치

```bash
# pip 업그레이드
python -m pip install --upgrade pip

# 기본 패키지 설치
pip install -r requirements.txt
```

### 3단계: GPU 지원 설정 (선택사항)

#### NVIDIA GPU가 있는 경우
```bash
# CUDA 확인
nvidia-smi

# PyTorch GPU 버전 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 설치 확인
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 4단계: YOLO 모델 다운로드

```python
# 모델 자동 다운로드 스크립트
from ultralytics import YOLO

print("모델 다운로드 중...")
models = ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11n-seg.pt']
for model_name in models:
    model = YOLO(model_name)
    print(f"✅ {model_name} 다운로드 완료")
```

---

## 🎨 RGBA 컬러 인식 설정

### RGBA 컬러 처리 구성

YOLO11 시스템에서 RGBA (Red, Green, Blue, Alpha) 이미지를 처리하려면 추가 설정이 필요합니다.

#### 1. 이미지 변환 함수
```python
import cv2
import numpy as np
from PIL import Image

def process_rgba_image(image_path):
    """
    RGBA 이미지를 YOLO가 처리할 수 있는 RGB로 변환
    투명도(Alpha) 채널을 처리하여 배경 합성
    """
    # PIL로 RGBA 이미지 로드
    img = Image.open(image_path)
    
    if img.mode == 'RGBA':
        # 흰색 배경 생성
        background = Image.new('RGB', img.size, (255, 255, 255))
        
        # Alpha 채널을 이용한 합성
        background.paste(img, mask=img.split()[3])  # 3은 alpha 채널
        
        # numpy 배열로 변환
        img_array = np.array(background)
        
        # OpenCV 형식으로 변환 (RGB → BGR)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_bgr
    else:
        # RGBA가 아닌 경우 일반 처리
        return cv2.imread(image_path)
```

#### 2. 컬러 기반 객체 구분
```python
def analyze_object_colors(image, detections):
    """
    검출된 객체의 주요 색상 분석
    """
    color_info = []
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        
        # 객체 영역 추출
        roi = image[int(y1):int(y2), int(x1):int(x2)]
        
        # 평균 색상 계산 (BGR)
        avg_color = cv2.mean(roi)[:3]
        
        # RGB로 변환
        avg_color_rgb = (avg_color[2], avg_color[1], avg_color[0])
        
        # HSV로 변환하여 색상 이름 결정
        hsv = cv2.cvtColor(
            np.uint8([[avg_color]]), 
            cv2.COLOR_BGR2HSV
        )[0][0]
        
        color_name = get_color_name(hsv)
        
        color_info.append({
            'bbox': det['bbox'],
            'rgb': avg_color_rgb,
            'hsv': hsv.tolist(),
            'color_name': color_name
        })
    
    return color_info

def get_color_name(hsv):
    """
    HSV 값을 기반으로 색상 이름 반환
    """
    h, s, v = hsv
    
    # 색상 범위 정의
    if s < 30:
        return "White/Gray/Black"
    elif h < 10 or h > 170:
        return "Red"
    elif 10 <= h < 25:
        return "Orange"
    elif 25 <= h < 35:
        return "Yellow"
    elif 35 <= h < 85:
        return "Green"
    elif 85 <= h < 135:
        return "Blue"
    elif 135 <= h <= 170:
        return "Purple"
    else:
        return "Unknown"
```

#### 3. 멀티 레이어 검출기에 통합
```python
# multi_layer_detector.py 수정
class MultiLayerObjectDetector:
    def detect_with_color(self, image_path, analyze_colors=True):
        """
        컬러 분석이 포함된 다중 레이어 검출
        """
        # RGBA 이미지 처리
        image = process_rgba_image(image_path)
        
        # 기존 검출 수행
        results = self.detect_multi_layer(image_path)
        
        # 컬러 분석 추가
        if analyze_colors and results['final_detections']:
            color_info = analyze_object_colors(
                image, 
                results['final_detections']
            )
            results['color_analysis'] = color_info
        
        return results
```

---

## ✅ 환경 검증

### 전체 환경 테스트 스크립트

```python
# test_environment.py
import sys
import subprocess
import importlib.util

def check_environment():
    """환경 설정 검증"""
    
    print("=" * 60)
    print("🔍 YOLO11 Multi-Layer Detection 환경 검증")
    print("=" * 60)
    
    # 1. Python 버전
    print(f"\n1. Python 버전: {sys.version}")
    if sys.version_info < (3, 8):
        print("   ❌ Python 3.8 이상이 필요합니다!")
    else:
        print("   ✅ Python 버전 OK")
    
    # 2. 필수 패키지
    required_packages = [
        'cv2',
        'numpy',
        'ultralytics',
        'torch',
        'PIL',
        'matplotlib',
        'tkinter',
        'pandas',
        'sklearn',
        'yaml'
    ]
    
    print("\n2. 필수 패키지 확인:")
    missing_packages = []
    for package in required_packages:
        if package == 'cv2':
            package_name = 'cv2'
            import_name = 'cv2'
        elif package == 'PIL':
            package_name = 'Pillow'
            import_name = 'PIL'
        elif package == 'sklearn':
            package_name = 'scikit-learn'
            import_name = 'sklearn'
        elif package == 'yaml':
            package_name = 'PyYAML'
            import_name = 'yaml'
        else:
            package_name = package
            import_name = package
        
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            print(f"   ❌ {package_name} 없음")
            missing_packages.append(package_name)
        else:
            print(f"   ✅ {package_name} OK")
    
    # 3. GPU 확인
    print("\n3. GPU 설정:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ CUDA 사용 가능")
            print(f"   - GPU: {torch.cuda.get_device_name(0)}")
            print(f"   - CUDA 버전: {torch.version.cuda}")
            print(f"   - GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("   ⚠️ GPU 없음 (CPU 모드로 실행)")
    except:
        print("   ❌ PyTorch GPU 확인 실패")
    
    # 4. 모델 파일 확인
    print("\n4. YOLO 모델 파일:")
    import os
    models = ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11n-seg.pt']
    for model in models:
        if os.path.exists(model):
            size = os.path.getsize(model) / 1024**2
            print(f"   ✅ {model} ({size:.1f} MB)")
        else:
            print(f"   ❌ {model} 없음")
    
    # 5. 결과 요약
    print("\n" + "=" * 60)
    if missing_packages:
        print("⚠️ 설치 필요한 패키지:")
        print(f"pip install {' '.join(missing_packages)}")
    else:
        print("✅ 모든 환경 설정이 완료되었습니다!")
    print("=" * 60)

if __name__ == "__main__":
    check_environment()
```

실행:
```bash
python test_environment.py
```

---

## 🐛 일반적인 오류 해결

### 1. ImportError: No module named 'ultralytics'

**원인**: YOLO11 패키지가 설치되지 않음

**해결**:
```bash
pip install ultralytics --upgrade
```

### 2. RuntimeError: CUDA out of memory

**원인**: GPU 메모리 부족

**해결**:
```python
# 배치 크기 감소
detector = MultiLayerObjectDetector()
# Layer 1, 3만 사용
results = detector.detect_multi_layer(
    image_path="image.jpg",
    use_layers=[True, False, True, False]
)

# 또는 CPU 사용
detector = MultiLayerObjectDetector(device='cpu')
```

### 3. cv2.error: OpenCV assertion failed

**원인**: 이미지 경로 오류 또는 잘못된 형식

**해결**:
```python
import os

# 경로 확인
if not os.path.exists(image_path):
    print(f"파일을 찾을 수 없습니다: {image_path}")
    
# 절대 경로 사용
abs_path = os.path.abspath(image_path)

# 이미지 형식 확인
valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
    print("지원하지 않는 이미지 형식입니다")
```

### 4. ModuleNotFoundError: No module named 'tkinter'

**원인**: tkinter가 설치되지 않음 (GUI 모드)

**해결**:

#### Ubuntu/Debian
```bash
sudo apt-get install python3-tk
```

#### macOS
```bash
brew install python-tk
```

#### Windows
tkinter는 기본 포함되어 있음. Python 재설치 필요할 수 있음

### 5. HTTPError downloading YOLO models

**원인**: 네트워크 문제 또는 방화벽

**해결**:
```python
# 수동 다운로드
import urllib.request

models = {
    'yolo11n.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt',
    'yolo11s.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt',
    'yolo11m.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt',
    'yolo11n-seg.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt'
}

for name, url in models.items():
    print(f"Downloading {name}...")
    urllib.request.urlretrieve(url, name)
    print(f"✅ {name} downloaded")
```

### 6. ValueError: not enough values to unpack

**원인**: 검출 결과가 없거나 예상과 다른 형식

**해결**:
```python
# 안전한 결과 처리
results = detector.detect_multi_layer(image_path)

if results and results['final_detections']:
    for det in results['final_detections']:
        # 안전한 언패킹
        bbox = det.get('bbox', [0, 0, 0, 0])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
        else:
            print(f"잘못된 bbox 형식: {bbox}")
else:
    print("검출된 객체가 없습니다")
```

### 7. PermissionError: [Errno 13]

**원인**: 파일 접근 권한 없음

**해결**:
```bash
# Linux/macOS
chmod 755 multi_layer_app.py
chmod 644 *.jpg

# Windows (관리자 권한으로 실행)
# 파일 속성 > 보안 > 권한 수정
```

---

## ⚡ 성능 최적화

### 1. GPU 최적화

```python
# Mixed Precision Training
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# GPU 메모리 정리
torch.cuda.empty_cache()
```

### 2. 이미지 전처리 최적화

```python
# 이미지 크기 조정
def optimize_image_size(image_path, max_size=1280):
    """큰 이미지를 적절한 크기로 조정"""
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    if width > max_size or height > max_size:
        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height))
    
    return img
```

### 3. 배치 처리

```python
# 여러 이미지 동시 처리
def batch_detection(image_paths, detector):
    """배치 처리로 속도 향상"""
    results = []
    
    # 이미지 로드
    images = [cv2.imread(path) for path in image_paths]
    
    # 배치 처리
    batch_results = detector.model(images, batch=True)
    
    return batch_results
```

---

## 🖥️ 플랫폼별 가이드

### Windows 10/11

#### 설치 순서
1. Python 3.10 설치 (python.org)
2. Visual Studio Build Tools 설치
3. CUDA Toolkit 설치 (NVIDIA GPU가 있는 경우)
4. 가상환경 생성 및 패키지 설치

#### 일반적인 문제
- **긴 경로 문제**: 레지스트리에서 LongPathsEnabled 활성화
- **권한 문제**: PowerShell을 관리자 권한으로 실행

### Ubuntu 20.04/22.04

#### 설치 스크립트
```bash
#!/bin/bash
# Ubuntu 전체 설치 스크립트

# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# Python 및 pip 설치
sudo apt install python3.10 python3.10-venv python3-pip -y

# 개발 도구
sudo apt install build-essential cmake -y

# OpenCV 의존성
sudo apt install libopencv-dev python3-opencv -y

# tkinter 설치
sudo apt install python3-tk -y

# 가상환경 생성
python3.10 -m venv yolo_env
source yolo_env/bin/activate

# 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt
```

### macOS (M1/M2 Silicon)

#### 특별 고려사항
```bash
# Homebrew로 Python 설치
brew install python@3.10

# Metal Performance Shaders (MPS) 지원
# PyTorch는 자동으로 MPS 사용
python -c "import torch; print(torch.backends.mps.is_available())"
```

---

## 📊 성능 모니터링

### 시스템 리소스 모니터링

```python
import psutil
import GPUtil

def monitor_resources():
    """시스템 리소스 사용량 모니터링"""
    
    # CPU 사용률
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU 사용률: {cpu_percent}%")
    
    # 메모리 사용량
    memory = psutil.virtual_memory()
    print(f"RAM 사용: {memory.percent}% ({memory.used/1024**3:.1f}/{memory.total/1024**3:.1f} GB)")
    
    # GPU 사용량 (NVIDIA)
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU {gpu.name}: {gpu.load*100:.1f}% | 메모리: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
    except:
        print("GPU 정보를 가져올 수 없습니다")
```

---

## 🔐 보안 고려사항

### 안전한 이미지 처리

```python
import os
import hashlib

def validate_image(image_path):
    """이미지 파일 검증"""
    
    # 파일 존재 확인
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {image_path}")
    
    # 파일 크기 확인 (100MB 제한)
    file_size = os.path.getsize(image_path)
    if file_size > 100 * 1024 * 1024:
        raise ValueError(f"파일이 너무 큽니다: {file_size/1024**2:.1f} MB")
    
    # 파일 확장자 확인
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in valid_extensions:
        raise ValueError(f"지원하지 않는 형식: {ext}")
    
    # 파일 헤더 확인 (magic number)
    with open(image_path, 'rb') as f:
        header = f.read(8)
    
    # 이미지 형식별 시그니처
    signatures = {
        b'\xff\xd8\xff': 'JPEG',
        b'\x89PNG\r\n\x1a\n': 'PNG',
        b'BM': 'BMP'
    }
    
    valid = False
    for sig, format in signatures.items():
        if header.startswith(sig):
            valid = True
            break
    
    if not valid:
        raise ValueError("유효하지 않은 이미지 파일입니다")
    
    return True
```

---

## 📝 체크리스트

### 프로그램 실행 전 확인사항

- [ ] Python 3.8 이상 설치
- [ ] 가상환경 활성화
- [ ] requirements.txt 패키지 설치
- [ ] YOLO 모델 파일 다운로드
- [ ] GPU 드라이버 설치 (선택)
- [ ] CUDA/cuDNN 설치 (선택)
- [ ] 테스트 이미지 준비
- [ ] 환경 검증 스크립트 실행

### 첫 실행 명령어

```bash
# 1. 환경 확인
python test_environment.py

# 2. 간단한 테스트
python multi_layer_detector.py -i sample.jpg -v

# 3. GUI 실행
python multi_layer_app.py --gui

# 4. 종합 테스트
python test_multi_layer.py --comprehensive
```

---

## 📞 추가 지원

### 도움말 리소스

1. **GitHub Issues**: https://github.com/aebonlee/YOLO11_study/issues
2. **Ultralytics Docs**: https://docs.ultralytics.com/
3. **PyTorch Forums**: https://discuss.pytorch.org/
4. **Stack Overflow**: Tag with `yolov11`, `ultralytics`

### 디버깅 모드 실행

```python
# 상세 로그 출력
import logging
logging.basicConfig(level=logging.DEBUG)

# YOLO 상세 출력
model = YOLO('yolo11n.pt')
results = model(image, verbose=True)
```

---

**Last Updated**: 2025년 11월 21일  
**Version**: 1.0  
**Author**: aebonlee  
**Project**: YOLO11 Multi-Layer Detection System