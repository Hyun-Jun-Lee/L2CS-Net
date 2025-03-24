from l2cs import Pipeline, render
import cv2
from pathlib import Path
import torch
import numpy as np
import math
import json
from datetime import datetime
import time
import matplotlib.pyplot as plt

# 좌표를 저장할 리스트 생성
gaze_coordinates = []
measurement_started = False  # 측정 시작 여부
reference_pitch = None  # 기준점의 pitch
reference_yaw = None    # 기준점의 yaw

def calculate_gaze_coords(pitch, yaw, is_first_point=False):
    """
    캠 관점에서 시선 좌표 계산
    pitch: 상하 각도 (위쪽이 음수)
    yaw: 좌우 각도 (오른쪽이 양수)
    is_first_point: 첫 번째 측정점 여부
    """
    global reference_pitch, reference_yaw
    
    # 첫 번째 점이면 기준점으로 설정
    if is_first_point:
        reference_pitch = pitch
        reference_yaw = yaw
        return 0, 0  # 첫 번째 점은 원점으로
    
    # 기준점으로부터의 상대적인 각도 계산
    relative_pitch = pitch - reference_pitch
    relative_yaw = yaw - reference_yaw
    
    # 각도를 거리로 변환 (1도당 20cm로 계산하여 값을 크게 만듦)
    SCALE = 20.0  # 1도당 20cm
    
    # 좌우 움직임을 y축으로, 상하 움직임을 x축으로 설정
    y = relative_yaw * SCALE    # 좌우 좌표를 y축으로
    x = relative_pitch * SCALE  # 상하 좌표를 x축으로
    
    return x, y

CWD = Path(__file__).parent

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gaze_pipeline = Pipeline(
    weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
    arch='ResNet50',
    device=device  # GPU 사용 가능하면 GPU 사용
)

# 웹캠 초기화 및 설정
cap = cv2.VideoCapture(0)

# MJPG 코덱 설정 (먼저 설정)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
cap.set(cv2.CAP_PROP_FOURCC, fourcc)

# 다른 설정들
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)  # 180이 지원되지 않을 수 있어 60으로 조정
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 설정이 적용되었는지 확인
print(f"\n실제 카메라 설정:")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
print(f"해상도: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print(f"코덱: {int(cap.get(cv2.CAP_PROP_FOURCC))}")
print(f"버퍼 크기: {cap.get(cv2.CAP_PROP_BUFFERSIZE)}")

# 설정 확인
ret, test_frame = cap.read()
if ret:
    print(f"\n실제 프레임 크기: {test_frame.shape}")
    
# 지원되는 해상도 확인
def print_supported_resolutions():
    test_resolutions = [
        (640, 480),
        (1280, 720),
        (1920, 1080)
    ]
    print("\n지원되는 해상도 테스트:")
    for width, height in test_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"요청: {width}x{height} -> 실제: {actual_width}x{actual_height}")

print_supported_resolutions()

# 원하는 설정으로 다시 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 실제 FPS 계산을 위한 변수들
prev_frame_time = 0
curr_frame_time = 0
fps_list = []  # 최근 FPS 기록을 위한 리스트
FPS_WINDOW = 30  # FPS 평균을 계산할 윈도우 크기

# 처리 시간 측정을 위한 변수들
total_model_time = 0
total_render_time = 0
frame_count = 0
print_interval = 30  # 30프레임마다 평균 출력

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # 프레임 크기 조정
    frame = cv2.resize(frame, (960, 720))

    # FPS 계산
    curr_frame_time = time.time()
    if prev_frame_time != 0:
        fps = 1/(curr_frame_time-prev_frame_time)
        fps_list.append(fps)
        if len(fps_list) > FPS_WINDOW:
            fps_list.pop(0)
    prev_frame_time = curr_frame_time
    
    # 평균 FPS 계산
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    
    # 각 단계별 처리 시간 측정
    t1 = time.time()
    results = gaze_pipeline.step(frame)
    t2 = time.time()
    frame = render(frame, results)
    t3 = time.time()
    
    # 처리 시간 누적
    model_time = t2 - t1
    render_time = t3 - t2
    total_model_time += model_time
    total_render_time += render_time
    
    # 30프레임마다 평균 처리 시간 출력
    if frame_count % print_interval == 0:
        avg_model_time = (total_model_time / print_interval) * 1000  # ms로 변환
        avg_render_time = (total_render_time / print_interval) * 1000
        print(f"\n=== 성능 분석 (최근 {print_interval}프레임 평균) ===")
        print(f"모델 처리 시간: {avg_model_time:.1f}ms")
        print(f"렌더링 시간: {avg_render_time:.1f}ms")
        print(f"총 처리 시간: {(avg_model_time + avg_render_time):.1f}ms")
        print(f"현재 FPS: {avg_fps:.1f}")
        print("=====================================")
        # 누적값 초기화
        total_model_time = 0
        total_render_time = 0
    
    # 처리 시간 표시
    cv2.putText(frame, f'FPS: {avg_fps:.1f}', 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Model: {model_time*1000:.1f}ms', 
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Render: {render_time*1000:.1f}ms', 
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2)
    
    # Get gaze angles and convert to screen coordinates
    if results.pitch is not None and results.yaw is not None:
        for pitch, yaw in zip(results.pitch, results.yaw):
            # 각도 값 출력
            cv2.putText(frame, f'Angles - Pitch: {pitch:.1f}, Yaw: {yaw:.1f}', 
                       (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)

            # 측정 시작된 경우에만 좌표 저장
            if measurement_started:
                # 첫 번째 점인지 확인
                is_first_point = len(gaze_coordinates) == 0
                
                # 좌표 계산
                x, y = calculate_gaze_coords(pitch, yaw, is_first_point)
                
                # 좌표와 시간 저장
                gaze_coordinates.append((x, y, time.time()))
            
                # 화면에 정보 표시
                cv2.putText(frame, f'Gaze point: ({x:.1f}cm, {y:.1f}cm)', 
                        (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 2)
                
                # 측정 상태 표시
                status = "measurement in progress" if measurement_started else "press space bar to start measurement"
                cv2.putText(frame, f'Status: {status}', 
                        (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 2)
                
                # 저장된 좌표 개수 표시
                cv2.putText(frame, f'Saved coordinates: {len(gaze_coordinates)}', 
                        (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 2)
        
    # Display the frame
    cv2.imshow('L2CS Gaze Estimation', frame)
    
    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # q를 누르면 종료
        break
    elif key == ord(' '):  # 스페이스바를 누르면 측정 시작/종료
        measurement_started = not measurement_started  # 상태 토글
        if measurement_started:
            gaze_coordinates.clear()  # 새로운 측정 시작시 기존 데이터 삭제
            print("측정을 시작합니다.")
        else:
            # 측정이 끝나면 결과를 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f'gaze_coordinates_{timestamp}.json', 'w') as f:
                json.dump({
                    "coordinates": [
                        {"x": float(x), "y": float(y), "timestamp": float(t)}
                        for x, y, t in gaze_coordinates
                    ]
                }, f, indent=2)
            print(f"측정을 종료하고 결과를 저장했습니다: gaze_coordinates_{timestamp}.json")
            
            # 결과 시각화
            plt.figure(figsize=(8, 8))
            # 상하(pitch)를 x축으로, 좌우(yaw)를 y축으로 설정
            x_coords = [x for x, y, _ in gaze_coordinates]  # pitch (상하)
            y_coords = [y for x, y, _ in gaze_coordinates]  # yaw (좌우)

            # 데이터 범위 계산
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # 범위에 여유 공간 추가 (10% 정도)
            x_margin = (x_max - x_min) * 0.1 if len(x_coords) > 1 else 1.0
            y_margin = (y_max - y_min) * 0.1 if len(y_coords) > 1 else 1.0
            
            # 축 범위 설정
            plt.xlim(x_min - x_margin, x_max + x_margin)
            plt.ylim(y_min - y_margin, y_max + y_margin)

            # 산점도 그리기 (첫 번째 점은 제외)
            if len(x_coords) > 1:
                plt.scatter(x_coords[1:], y_coords[1:], alpha=0.5, c='blue', s=10, label='Gaze Points')

                # 시선 이동 경로 그리기 (첫 번째 점은 제외)
                plt.plot(x_coords[1:], y_coords[1:], 'b-', alpha=0.2, linewidth=1, label='Gaze Path')

                # 시작 위치 (녹색 별) - 두 번째 점
                plt.plot(x_coords[1], y_coords[1], 'g*', markersize=15, label='Start')
                
                # 끝 위치 (빨간 별)
                plt.plot(x_coords[-1], y_coords[-1], 'r*', markersize=15, label='End')

            plt.title('Gaze Coordinates Visualization')
            plt.xlabel('Down (-) / Up (+) in cm')     # x축은 상하
            plt.ylabel('Left (-) / Right (+) in cm')  # y축은 좌우
            plt.grid(True)

            # 중심점(0,0) 표시
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

            # 범례 표시
            plt.legend()

            # 플롯 저장
            plt.savefig(f'gaze_plot_{timestamp}.png')
            plt.close()
        
# Release everything when job is finished
cap.release()
cv2.destroyAllWindows()