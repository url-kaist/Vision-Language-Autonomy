import open3d as o3d
import numpy as np

def auto_level_pcd(input_path, output_path):
    print(f"1. 파일 읽는 중: {input_path}")
    pcd = o3d.io.read_point_cloud(input_path)
    
    if pcd.is_empty():
        print("!! 오류: 포인트 클라우드가 비어있거나 파일을 읽을 수 없습니다.")
        return

    print(f"   - 전체 포인트 개수: {len(pcd.points)}개")

    # 2. 지면 검출 (RANSAC 알고리즘)
    # distance_threshold: 평면으로 간주할 두께 (예: 0.1m 이내의 점들은 같은 평면으로 봄)
    # ransac_n: 평면 추정을 위해 샘플링할 점의 개수 (보통 3)
    # num_iterations: 반복 횟수 (클수록 정확하지만 느림)
    print("2. 가장 큰 평면(지면) 찾는 중...")
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.1,
                                             ransac_n=3,
                                             num_iterations=2000)
    
    # 평면 방정식: ax + by + cz + d = 0
    [a, b, c, d] = plane_model
    print(f"   - 검출된 평면 방정식: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # 3. 지면의 높이 계산
    # 지면으로 판별된 점들(inliers)만 추출
    ground_cloud = pcd.select_by_index(inliers)
    ground_points = np.asarray(ground_cloud.points)
    
    # 지면 점들의 평균 Z값 계산
    mean_z = np.mean(ground_points[:, 2])
    print(f"3. 현재 지면의 평균 높이(Z): {mean_z:.4f} m")

    # 4. 전체 포인트 클라우드 이동
    # 지면이 0이 되려면, 현재 지면 높이만큼 반대로 이동하면 됨
    shift_value = -mean_z
    print(f"   - Z축 이동량: {shift_value:.4f} m")
    
    # 변환 행렬 없이 단순 좌표 이동 (Translate)
    # (x, y는 그대로 0, z만 shift_value 만큼 이동)
    pcd.translate((0, 0, shift_value))

    # 5. 결과 저장
    print(f"4. 저장 중: {output_path}")
    o3d.io.write_point_cloud(output_path, pcd) # binary format으로 저장됨 (용량 절약)
    # 텍스트로 보고 싶다면: o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)
    
    print("완료되었습니다! 이제 지면의 높이는 약 0.0m 입니다.")

if __name__ == "__main__":
    # --- 사용자가 수정할 부분 ---
    input_file = "vla.pcd"       # 원본 파일 경로
    output_file = "vla_modified.pcd" # 저장할 파일 경로
    # -------------------------
    
    auto_level_pcd(input_file, output_file)