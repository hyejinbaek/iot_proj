# IoT 음성인식 AI도입 가능 여부 테스트

import json

# 명령어 매핑 딕셔너리
actions = {
    "디퓨저 켜줘": {"device": "디퓨저", "action": "ON"},
    "디퓨저 꺼줘": {"device": "디퓨저", "action": "OFF"},
    "조명 켜줘": {"device": "조명", "action": "ON"},
    "조명 꺼줘": {"device": "조명", "action": "OFF"},
}

def command_to_json(command: str):
    command = command.lower().strip()  # 소문자로 변환 후 공백 제거
    if command in actions:
        json_data = json.dumps(actions[command], ensure_ascii=False)
        return json_data
    else:
        return json.dumps({"error": "알 수 없는 명령어입니다."}, ensure_ascii=False)

# 실행 테스트
if __name__ == "__main__":
    while True:
        user_input = input("사용자: ")
        if user_input.lower() == "종료":
            print("프로그램 종료.")
            break
        result = command_to_json(user_input)
        print("JSON 출력:", result)
