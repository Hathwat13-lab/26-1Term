# Agent Setup Guide (Nam's AI Agent)

이 문서는 Nam 님의 컴퓨터에서 이 프로젝트를 처음 세팅하는 AI 에이전트를 위한 필수 지침서입니다.
에이전트는 이 파일을 읽는 즉시 아래의 지시사항을 순차적으로 수행하여 로컬 개발 환경을 구성하십시오.

## 🛠️ 필수 수행 단계 (Required Steps)

**1. 가상 환경 생성 (Virtual Environment Creation)**
* 프로젝트 루트 디렉토리에서 Python 가상 환경을 생성합니다. (만약 이미 `venv` 폴더가 존재한다면 건너뜁니다.)
* **명령어**: `python -m venv venv`

**2. 필수 패키지 설치 (Install Dependencies)**
* 생성된 가상 환경 내의 `pip`를 사용하여 `requirements.txt`에 명시된 라이브러리들을 설치합니다. 
* 에이전트가 직접 쉘 명령어를 실행할 때는 가상환경 경로를 명시하여 실행하십시오.
* **명령어 (Windows 기준)**: `.\venv\Scripts\pip install -r requirements.txt`
* **명령어 (Mac/Linux 기준)**: `./venv/bin/pip install -r requirements.txt`

**3. 가상 환경 활성화 안내 (User Prompting)**
* 사용자(Nam 님)가 에디터 터미널을 직접 열 경우, 전역 파이썬(예: conda)이 연결되어 가상환경이 자동으로 켜지지 않을 수 있습니다.
* 세팅이 완료되면 에이전트는 사용자에게 반드시 다음을 안내하십시오: *"터미널에서 직접 코드를 실행하실 때는 `.\venv\Scripts\activate` (Mac/Linux는 `source venv/bin/activate`)를 실행하여 `(venv)`가 표시되는지 확인해 주세요!"*

**4. 컨텍스트 파악 (Context Synchronization)**
* 위 세팅이 모두 끝나면, 루트 디렉토리에 있는 **`project_context.md`** 와 **`dev_initial_memo.md`** 를 읽고 프로젝트의 현재 진행 상황과 기획 의도를 파악하십시오.
* 파악이 완전히 끝나면 사용자에게 **"현재 상황 파악 완료"**라고 대답한 뒤, 다음 작업(모듈 개발 등) 지시를 대기하십시오.
