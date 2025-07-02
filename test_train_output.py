#!/usr/bin/env python3
"""
Train.py 출력 로깅 테스트 스크립트

수정된 train.py의 개선된 출력 로깅을 테스트합니다.
- tqdm leave=True로 epoch 완료 후에도 진행률 유지
- 구조화된 epoch 요약 출력
- 깔끔한 검증 및 체크포인트 로깅
"""

import sys
from pathlib import Path
import time
from tqdm import tqdm
import logging

# 현재 디렉토리를 path에 추가
sys.path.append(str(Path(__file__).parent / "src"))

def test_tqdm_output():
    """tqdm 출력 테스트"""
    print("🧪 tqdm 출력 테스트 시작")
    print("=" * 80)
    
    epochs = 3
    batches_per_epoch = 10
    
    for epoch in range(epochs):
        print(f"\n🔄 Starting Epoch {epoch+1}/{epochs}")
        
        # tqdm 설정 (수정된 방식)
        pbar = tqdm(
            range(batches_per_epoch), 
            desc=f"Epoch {epoch+1}/{epochs}", 
            dynamic_ncols=True, 
            leave=True,  # epoch 완료 후에도 진행률 유지
            unit="batch",
            colour="green",
            position=0,  # 출력 위치 고정
            ncols=120    # 고정 너비로 깔끔한 출력
        )
        
        for batch_idx in pbar:
            # 가상의 loss 계산
            fake_loss = 10.0 * (1 - (epoch * batches_per_epoch + batch_idx) / (epochs * batches_per_epoch))
            fake_box_loss = fake_loss * 0.4
            fake_cls_loss = fake_loss * 0.3
            fake_obj_loss = fake_loss * 0.3
            
            # Description 업데이트
            desc = f"Epoch {epoch+1}/{epochs} | Loss: {fake_loss:.4f}"
            desc += f" | Box: {fake_box_loss:.3f} | Cls: {fake_cls_loss:.3f} | Obj: {fake_obj_loss:.3f}"
            pbar.set_description(desc)
            
            # Postfix 업데이트
            if batch_idx % 3 == 0:  # 가끔 unlabeled loss 표시
                postfix_dict = {'unlabeled': f"{fake_loss * 0.2:.3f}"}
                pbar.set_postfix(postfix_dict)
            
            time.sleep(0.1)  # 시뮬레이션을 위한 지연
        
        # Epoch 완료 표시
        final_desc = f"Epoch {epoch+1}/{epochs} ✅ Completed"
        pbar.set_description(final_desc)
        pbar.refresh()
        # pbar.close()를 하지 않아서 진행률이 유지됨
        
        print()  # 줄바꿈으로 시각적 구분
        print("=" * 80)
        print(f"📊 EPOCH {epoch+1}/{epochs} SUMMARY")
        print("=" * 80)
        print(f"Average Loss: {fake_loss:.6f}")
        print(f"Learning Rate: {0.001 * (0.9 ** epoch):.8f}")
        
        progress_percent = (epoch + 1) / epochs * 100
        print(f"Training Progress: {progress_percent:.1f}% ({epoch + 1}/{epochs} epochs)")
        
        # MC Dropout 품질 메트릭 시뮬레이션
        if epoch % 2 == 0:  # 짝수 epoch마다
            print()
            print(f"🔬 [Epoch {epoch+1}] MC Dropout Quality Metrics:")
            print(f"  - avg_box_variance: {0.15 * (1 - epoch/epochs):.6f}")
            print(f"  - avg_confidence: {0.7 + 0.2 * epoch/epochs:.6f}")
            print(f"  - total_detections: {50 + epoch * 10}")
            print()
        
        # 검증 및 체크포인트 시뮬레이션
        if (epoch + 1) % 2 == 0:  # 2 epoch마다
            print("-" * 60)
            print(f"💾 CHECKPOINT & EVALUATION - Epoch {epoch+1}")
            print("-" * 60)
            
            # 가상의 검증 결과
            fake_map50 = 0.3 + 0.2 * (epoch + 1) / epochs
            fake_map95 = fake_map50 * 0.7
            
            print("📊 Validation Results:")
            print(f"   mAP@0.5: {fake_map50:.4f}")
            print(f"   mAP@0.5:0.95: {fake_map95:.4f}")
            
            if epoch == epochs - 1:  # 마지막 epoch에서 best 달성
                print(f"🎉 NEW BEST mAP@0.5: {fake_map50:.4f} (Previous: {fake_map50-0.1:.4f})")
                print(f"🎉 NEW BEST mAP@0.5:0.95: {fake_map95:.4f}")
            else:
                print(f"   Best mAP@0.5: {fake_map50+0.1:.4f} (Current: {fake_map50:.4f})")
                print(f"   Best mAP@0.5:0.95: {fake_map95+0.1:.4f} (Current: {fake_map95:.4f})")
            
            print("💾 Saving checkpoints...")
            print(f"   ✓ Latest: ./runs/train/test/latest_model.pt")
            
            if epoch == epochs - 1:  # 마지막 epoch에서만 best 저장
                print(f"   🏆 Best: ./runs/train/test/best_model.pt")
            
            print("-" * 60)
        
        time.sleep(0.5)  # epoch 간 간격
    
    print("\n🎉 학습 완료!")
    print("각 epoch의 진행률 표시가 유지되는 것을 확인할 수 있습니다.")

def test_logger_output():
    """로거 출력 테스트"""
    print("\n\n🧪 로거 출력 테스트")
    print("=" * 80)
    
    # 간단한 로거 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    
    # 테스트 로그 메시지들
    logger.info("🚀 Starting training...")
    logger.info("✓ Labeled 데이터: 1000 샘플, 50 배치")
    logger.info("✓ Unlabeled 데이터: 5000 샘플, 250 배치")
    logger.info("✓ Validation 데이터: 500 샘플, 25 배치")
    
    time.sleep(1)
    
    logger.info("=" * 80)
    logger.info("📊 EPOCH 1/10 SUMMARY")
    logger.info("=" * 80)
    logger.info("Average Loss: 5.234567")
    logger.info("Learning Rate: 0.00100000")
    logger.info("Training Progress: 10.0% (1/10 epochs)")
    
    time.sleep(1)
    
    logger.info("🔬 [Epoch 1] MC Dropout Quality Metrics:")
    logger.info("  - avg_box_variance: 0.150000")
    logger.info("  - avg_confidence: 0.700000")
    logger.info("  - total_detections: 50")

if __name__ == "__main__":
    print("🧪 Train.py 출력 로깅 개선 테스트")
    print("=" * 80)
    print("이 테스트는 수정된 train.py의 개선된 출력 로깅을 시뮬레이션합니다.")
    print("주요 개선사항:")
    print("  1. tqdm leave=True로 epoch 완료 후에도 진행률 유지")
    print("  2. 구조화된 epoch 요약 출력")
    print("  3. 깔끔한 검증 및 체크포인트 로깅")
    print("  4. 시각적 구분을 위한 구분선 추가")
    print("=" * 80)
    
    try:
        test_tqdm_output()
        test_logger_output()
        
        print("\n✅ 모든 테스트 완료!")
        print("실제 train.py 실행 시 위와 같은 개선된 출력을 볼 수 있습니다.")
        
    except KeyboardInterrupt:
        print("\n⚠️ 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}") 