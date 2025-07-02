#!/usr/bin/env python3

import yaml
from pathlib import Path

def test_reliability_threshold_config():
    """reliability_threshold 설정이 제대로 읽어지는지 테스트"""
    
    config_path = Path("configs/yolo_config.yaml")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # reliability_threshold 읽기 테스트
        reliability_threshold = config.get('training', {}).get(
            'semi_supervised', {}
        ).get('uncertainty', {}).get('reliability_threshold', 0.8)
        
        print("✅ 설정 파일 읽기 성공!")
        print(f"📍 Config path: {config_path.absolute()}")
        print(f"🔧 reliability_threshold: {reliability_threshold}")
        print(f"📋 전체 uncertainty 설정:")
        
        uncertainty_config = config.get('training', {}).get(
            'semi_supervised', {}
        ).get('uncertainty', {})
        
        for key, value in uncertainty_config.items():
            print(f"   - {key}: {value}")
        
        # MC Dropout 관련 설정도 확인
        print(f"\n🤖 MC Dropout 설정:")
        model_config = config.get('model', {}).get('dropout', {})
        for key, value in model_config.items():
            print(f"   - {key}: {value}")
        
        # 기본값과 비교
        if reliability_threshold == 0.8:
            print(f"\n✅ reliability_threshold가 기본값 0.8로 설정됨")
        else:
            print(f"\n🔧 reliability_threshold가 사용자 설정값 {reliability_threshold}로 설정됨")
            
        return True
        
    except FileNotFoundError:
        print(f"❌ 설정 파일을 찾을 수 없습니다: {config_path}")
        return False
    except yaml.YAMLError as e:
        print(f"❌ YAML 파싱 오류: {e}")
        return False
    except Exception as e:
        print(f"❌ 설정 읽기 오류: {e}")
        return False

def test_mc_dropout_usage():
    """MC Dropout 다층 신뢰도 평가 사용법 예시"""
    print("\n" + "="*50)
    print("📚 MC Dropout 다층 신뢰도 평가 사용법 가이드")
    print("="*50)
    
    print("""
🔧 설정 파일에서 reliability_threshold 조정:
   src/configs/yolo_config.yaml:
   
   training:
     semi_supervised:
       uncertainty:
         reliability_threshold: 0.8  # 0.0-1.0 범위
   
🚀 코드에서 사용:
   
   # 기본 사용 (config에서 자동 읽기)
   results = detector.predict_with_uncertainty(image, config=config)
   
   # 수동 설정
   results = detector.predict_with_uncertainty(image, reliability_threshold=0.85)
   
📊 반환 결과:
   
   for result in results:
       stats = result['reliability_stats']
       print(f"총 검출: {stats['total_detections']}")
       print(f"고품질: {stats['high_quality']}")
       print(f"중품질: {stats['medium_quality']}")
       print(f"평균 신뢰도: {stats['avg_final_score']:.3f}")

🎯 권장 threshold 값:
   - 0.9: 매우 엄격 (고품질 위주)
   - 0.8: 균형잡힌 설정 (권장)
   - 0.7: 느슨한 설정 (양 위주)
   - 0.6: 매우 느슨한 설정
""")

if __name__ == "__main__":
    print("🧪 reliability_threshold 설정 테스트")
    print("=" * 50)
    
    success = test_reliability_threshold_config()
    
    if success:
        test_mc_dropout_usage()
        print("\n✅ 모든 테스트 완료!")
    else:
        print("\n❌ 테스트 실패") 