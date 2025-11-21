"""
YOLO11 Multi-Layer Detection 환경 검증 스크립트
시스템 환경을 확인하고 문제점을 진단합니다.
"""

import sys
import os
import subprocess
import importlib.util
import warnings
warnings.filterwarnings('ignore')


def check_environment():
    """환경 설정 검증"""
    
    print("=" * 60)
    print("🔍 YOLO11 Multi-Layer Detection 환경 검증")
    print("=" * 60)
    
    errors = []
    warnings_list = []
    
    # 1. Python 버전
    print(f"\n1. Python 버전 확인")
    print(f"   현재 버전: {sys.version}")
    if sys.version_info < (3, 8):
        errors.append("Python 3.8 이상이 필요합니다!")
        print("   ❌ Python 버전 불충족")
    else:
        print("   ✅ Python 버전 OK")
    
    # 2. 필수 패키지
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'ultralytics': 'ultralytics',
        'torch': 'torch',
        'PIL': 'Pillow',
        'matplotlib': 'matplotlib',
        'tkinter': 'tkinter (시스템 패키지)',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm'
    }
    
    print("\n2. 필수 패키지 확인:")
    missing_packages = []
    
    for import_name, package_name in required_packages.items():
        try:
            if import_name == 'tkinter':
                import tkinter
                print(f"   ✅ {package_name} OK")
            else:
                spec = importlib.util.find_spec(import_name)
                if spec is None:
                    print(f"   ❌ {package_name} 없음")
                    missing_packages.append(package_name)
                else:
                    # 버전 확인
                    module = __import__(import_name)
                    version = getattr(module, '__version__', 'unknown')
                    print(f"   ✅ {package_name} OK (버전: {version})")
        except Exception as e:
            print(f"   ❌ {package_name} 확인 실패: {e}")
            missing_packages.append(package_name)
    
    if missing_packages:
        errors.append(f"누락된 패키지: {', '.join(missing_packages)}")
    
    # 3. GPU 확인
    print("\n3. GPU 설정:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ CUDA 사용 가능")
            print(f"   - GPU: {torch.cuda.get_device_name(0)}")
            print(f"   - CUDA 버전: {torch.version.cuda}")
            print(f"   - PyTorch 버전: {torch.__version__}")
            print(f"   - GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # GPU 메모리 체크
            if torch.cuda.get_device_properties(0).total_memory < 6 * 1024**3:
                warnings_list.append("GPU 메모리가 6GB 미만입니다. 일부 레이어만 사용하세요.")
        else:
            print("   ⚠️ GPU 없음 (CPU 모드로 실행)")
            warnings_list.append("GPU가 없습니다. 처리 속도가 느릴 수 있습니다.")
    except ImportError:
        errors.append("PyTorch가 설치되지 않았습니다")
        print("   ❌ PyTorch 없음")
    except Exception as e:
        print(f"   ❌ GPU 확인 실패: {e}")
    
    # 4. 모델 파일 확인
    print("\n4. YOLO 모델 파일:")
    models_info = {
        'yolo11n.pt': 5.4,  # MB
        'yolo11s.pt': 18.4,
        'yolo11m.pt': 38.8,
        'yolo11n-seg.pt': 6.5
    }
    
    missing_models = []
    for model, expected_size in models_info.items():
        if os.path.exists(model):
            actual_size = os.path.getsize(model) / 1024**2
            if abs(actual_size - expected_size) > expected_size * 0.1:  # 10% 오차 허용
                print(f"   ⚠️ {model} 크기 이상 ({actual_size:.1f} MB, 예상: {expected_size} MB)")
                warnings_list.append(f"{model} 파일 크기가 예상과 다릅니다")
            else:
                print(f"   ✅ {model} ({actual_size:.1f} MB)")
        else:
            print(f"   ❌ {model} 없음")
            missing_models.append(model)
    
    if missing_models:
        warnings_list.append(f"모델 파일 누락: {', '.join(missing_models)}")
    
    # 5. 프로젝트 파일 확인
    print("\n5. 프로젝트 핵심 파일:")
    core_files = [
        'multi_layer_detector.py',
        'multi_layer_app.py',
        'test_multi_layer.py',
        'multi_layer_tutorial.ipynb'
    ]
    
    for file in core_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} 없음")
            errors.append(f"핵심 파일 누락: {file}")
    
    # 6. 메모리 확인
    print("\n6. 시스템 메모리:")
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"   총 메모리: {memory.total / 1024**3:.1f} GB")
        print(f"   사용 가능: {memory.available / 1024**3:.1f} GB")
        print(f"   사용률: {memory.percent}%")
        
        if memory.available < 4 * 1024**3:
            warnings_list.append("사용 가능한 메모리가 4GB 미만입니다")
    except ImportError:
        print("   ⚠️ psutil 없음 - 메모리 확인 불가")
    
    # 7. OpenCV 테스트
    print("\n7. OpenCV 기능 테스트:")
    try:
        import cv2
        import numpy as np
        
        # 더미 이미지 생성 및 처리
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        print(f"   ✅ OpenCV 기본 기능 정상")
        print(f"   - OpenCV 버전: {cv2.__version__}")
    except Exception as e:
        errors.append(f"OpenCV 오류: {e}")
        print(f"   ❌ OpenCV 테스트 실패")
    
    # 8. 디스크 공간
    print("\n8. 디스크 공간:")
    try:
        import shutil
        usage = shutil.disk_usage(".")
        free_gb = usage.free / 1024**3
        print(f"   사용 가능 공간: {free_gb:.1f} GB")
        
        if free_gb < 5:
            warnings_list.append("디스크 공간이 5GB 미만입니다")
    except Exception as e:
        print(f"   ⚠️ 디스크 공간 확인 실패: {e}")
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📋 검증 결과 요약")
    print("=" * 60)
    
    if errors:
        print("\n❌ 오류 (반드시 해결 필요):")
        for error in errors:
            print(f"   • {error}")
    
    if warnings_list:
        print("\n⚠️ 경고 (권장사항):")
        for warning in warnings_list:
            print(f"   • {warning}")
    
    if not errors and not warnings_list:
        print("\n✅ 완벽! 모든 환경이 정상적으로 설정되었습니다.")
        print("   다음 명령으로 시작하세요:")
        print("   python multi_layer_app.py --gui")
    elif not errors:
        print("\n✅ 실행 가능! 경고사항을 참고하여 최적화할 수 있습니다.")
    else:
        print("\n❌ 실행 불가! 위의 오류를 먼저 해결해주세요.")
        
        # 해결 방법 제안
        print("\n💡 해결 방법:")
        if missing_packages:
            print(f"   pip install {' '.join(set(missing_packages) - {'tkinter (시스템 패키지)'})}")
        if missing_models:
            print("   python -c \"from ultralytics import YOLO; YOLO('yolo11n.pt')\"")
    
    return len(errors) == 0


def download_missing_models():
    """누락된 모델 다운로드"""
    print("\n📥 누락된 모델 다운로드 시작...")
    
    try:
        from ultralytics import YOLO
        
        models = ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11n-seg.pt']
        for model_name in models:
            if not os.path.exists(model_name):
                print(f"   다운로드 중: {model_name}")
                model = YOLO(model_name)
                print(f"   ✅ {model_name} 다운로드 완료")
            else:
                print(f"   ✓ {model_name} 이미 존재")
    except Exception as e:
        print(f"   ❌ 모델 다운로드 실패: {e}")
        return False
    
    return True


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO11 환경 검증 도구')
    parser.add_argument('--download-models', action='store_true',
                       help='누락된 모델 자동 다운로드')
    parser.add_argument('--fix', action='store_true',
                       help='발견된 문제 자동 수정 시도')
    args = parser.parse_args()
    
    # 환경 검증
    success = check_environment()
    
    # 모델 다운로드 옵션
    if args.download_models or args.fix:
        download_missing_models()
        print("\n" + "=" * 60)
        print("재검증 중...")
        success = check_environment()
    
    # 자동 수정 옵션
    if args.fix and not success:
        print("\n🔧 자동 수정 시도...")
        os.system("pip install -r requirements.txt")
        print("\n재검증 중...")
        success = check_environment()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())