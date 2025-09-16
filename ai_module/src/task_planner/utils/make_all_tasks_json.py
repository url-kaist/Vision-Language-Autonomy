import os
import json

def merge_json_files(input_dir, output_file):
    merged_data = []

    # 예외 파일
    skip_files = {'output.json', 'saved_answers.json'}

    # 모든 json 파일 순회
    for filename in os.listdir(input_dir):
        if filename.endswith('.json') and filename not in skip_files:
            file_path = os.path.join(input_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        merged_data.extend(data)  # 리스트 형태면 병합
                    else:
                        merged_data.append(data)  # 단일 객체면 리스트로 추가
            except Exception as e:
                print(f"파일 읽기 실패: {filename} - {e}")

    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

# 사용 예시
input_directory = '/ws/external/ai_module/src/task_planner/output'  # JSON 파일들이 있는 폴더 경로
output_json = '/ws/external/ai_module/src/task_planner/output/all_tasks.json'
merge_json_files(input_directory, output_json)
print(f"총 {output_json} 생성 완료.")
