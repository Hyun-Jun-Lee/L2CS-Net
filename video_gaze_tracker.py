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
import argparse

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

def save_gaze_data(gaze_data, output_path=None):
    """시선 추적 데이터를 JSON 파일로 저장"""
    if not gaze_data:
        print("저장할 데이터가 없습니다.")
        return
    
    # 데이터 형식 변환
    formatted_data = []
    start_time = gaze_data[0][2]  # 첫 번째 측정 시간
    
    for x, y, timestamp in gaze_data:
        relative_time = timestamp - start_time  # 시작 시간 기준 상대 시간
        formatted_data.append({
            "x": float(x),
            "y": float(y),
            "time": float(relative_time)
        })
    
    # 파일명 생성
    if output_path is None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"gaze_data_{current_time}.json"
    
    # JSON 파일로 저장
    with open(output_path, 'w') as f:
        json.dump(formatted_data, f, indent=2)
    
    print(f"데이터가 {output_path}에 저장되었습니다.")
    return output_path

def plot_gaze_path(gaze_data, output_path=None):
    """시선 경로를 그래프로 시각화하여 저장"""
    if not gaze_data:
        print("시각화할 데이터가 없습니다.")
        return
    
    # 데이터 추출
    x_coords = [point[0] for point in gaze_data]
    y_coords = [point[1] for point in gaze_data]
    
    # 그래프 생성
    plt.figure(figsize=(10, 8))
    plt.scatter(y_coords, x_coords, c=range(len(x_coords)), cmap='viridis', 
                alpha=0.7, s=10)
    
    # 선으로 연결
    plt.plot(y_coords, x_coords, 'b-', alpha=0.3)
    
    # 시작점과 끝점 표시
    plt.scatter(y_coords[0], x_coords[0], color='green', s=100, label='시작')
    plt.scatter(y_coords[-1], x_coords[-1], color='red', s=100, label='끝')
    
    # 그래프 설정
    plt.title('시선 추적 경로')
    plt.xlabel('좌우 (cm)')
    plt.ylabel('상하 (cm)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 축 방향 반전 (위쪽이 양수가 되도록)
    plt.gca().invert_yaxis()
    
    # 파일명 생성
    if output_path is None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"gaze_path_{current_time}.png"
    
    # 저장
    plt.savefig(output_path)
    plt.close()
    
    print(f"시선 경로가 {output_path}에 저장되었습니다.")
    return output_path

def main():
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='동영상 파일에서 시선 추적')
    parser.add_argument('--video', type=str, required=True, help='입력 동영상 파일 경로')
    parser.add_argument('--output', type=str, default=None, help='결과 저장 경로 (기본값: 자동 생성)')
    parser.add_argument('--save-video', action='store_true', help='처리된 동영상 저장 여부')
    parser.add_argument('--start-frame', type=int, default=0, help='시작 프레임 (기본값: 0)')
    parser.add_argument('--end-frame', type=int, default=-1, help='종료 프레임 (기본값: -1, 모든 프레임)')
    args = parser.parse_args()
    
    CWD = Path(__file__).parent

    # GPU 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 시선 추적 파이프라인 초기화
    gaze_pipeline = Pipeline(
        weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
        arch='ResNet50',
        device=device
    )

    # 동영상 파일 열기
    video_path = args.video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: 동영상 파일 '{video_path}'을 열 수 없습니다.")
        return
    
    # 동영상 정보 출력
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n동영상 정보:")
    print(f"해상도: {frame_width}x{frame_height}")
    print(f"FPS: {fps}")
    print(f"총 프레임 수: {total_frames}")
    print(f"재생 시간: {total_frames/fps:.2f}초")
    
    # 시작 프레임으로 이동
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        print(f"시작 프레임: {args.start_frame}")
    
    # 종료 프레임 설정
    end_frame = total_frames if args.end_frame == -1 else args.end_frame
    print(f"종료 프레임: {end_frame}")
    
    # 결과 동영상 저장 설정
    output_video = None
    if args.save_video:
        output_path = args.output or f"processed_{Path(video_path).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        print(f"처리된 동영상을 {output_path}에 저장합니다.")
    
    # 측정 자동 시작
    measurement_started = True
    gaze_coordinates.clear()
    
    # 처리 시간 측정을 위한 변수들
    total_model_time = 0
    total_render_time = 0
    frame_count = 0
    print_interval = 30  # 30프레임마다 평균 출력
    
    # 현재 프레임 번호
    current_frame = args.start_frame
    
    while cap.isOpened() and current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_frame += 1
        
        # 진행 상황 표시
        if frame_count % 100 == 0:
            progress = (current_frame - args.start_frame) / (end_frame - args.start_frame) * 100
            print(f"진행률: {progress:.1f}% ({current_frame}/{end_frame})")
        
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
            print("=====================================")
            # 누적값 초기화
            total_model_time = 0
            total_render_time = 0
        
        # 텍스트 정보 표시 제거 (사용자 요청에 따라)
        
        # Get gaze angles and convert to screen coordinates
        if results.pitch is not None and results.yaw is not None:
            for pitch, yaw in zip(results.pitch, results.yaw):
                # 첫 번째 점인지 확인
                is_first_point = len(gaze_coordinates) == 0
                
                # 좌표 계산
                x, y = calculate_gaze_coords(pitch, yaw, is_first_point)
                
                # 좌표와 시간 저장
                gaze_coordinates.append((x, y, time.time()))
                
                # 텍스트 정보 표시 제거 (사용자 요청에 따라)
        
        # 결과 동영상 저장
        if args.save_video and output_video is not None:
            output_video.write(frame)
        
        # 화면에 표시
        cv2.imshow('Video Gaze Tracking', frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 처리 완료 후 데이터 저장
    if gaze_coordinates:
        # JSON 데이터 저장
        json_path = args.output or None
        save_gaze_data(gaze_coordinates, json_path)
        
        # 시선 경로 그래프 저장
        plot_path = f"{Path(json_path).stem}_path.png" if json_path else None
        plot_gaze_path(gaze_coordinates, plot_path)
    
    # 자원 해제
    cap.release()
    if output_video is not None:
        output_video.release()
    cv2.destroyAllWindows()
    
    print("처리가 완료되었습니다.")

if __name__ == "__main__":
    main()
