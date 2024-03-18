## 보행데이터 활용 헬스케어 AI 해커톤 경진대회
### 📌 대회 소개
- **대회목표** : 근감소증을 가진 사람과 정상 군을 보행 걸음 신호만으로 특징을 추출하고 두 가지의 범주로 높은 정확도로 분류하라.
- **데이터 구성**
    - 보행시 5개 부위(허리, 왼손, 오른손, 왼발, 오른발)에 부착된 6축 IMU 센서 데이터(200Hz 샘플링)
    - 왼쪽부터 엑셀로미터 센서, 자이로스코프 센서 각 3개씩 6축으로 구성
 
    - | timestamp | ax | ay | az | gx | gy | gz |
      |-----------|----|----|----|----|----|----|

      
- **데이터 설명**
    - 보행 : 출발점에서 시작해 분기점을 돌아 다시 돌아오는 행위의 데이터
    - 10s : 10걸음만큼 직진으로 걸은 내용의 데이터
<!--
<p align="center" width="200%">
    <img width="35%" src="https://github.com/ssh6lq/Healthcare-AI-Hackathon-Competition-Using-Pedestrian-Data/assets/154342847/fc3caab6-c23e-4e9d-9938-b75172ebcc35.png width="200" height="400""> 
</p>
-->

---

### 📆 프로젝트 기간
- `2022.11.25 ~ 2022.11.26`

---

### ⚙️ 개발환경
- `Python`
- **Tool** : `tensorflow` `pytorch` `numpy` `pandas`
  
---

### ✔️ 역할
- 팀 : 석사1, 학부3
- 기술 서베이, 데이터 전처리, AI 모델링

---

### AI 모델링
- **LSTM-Autoencoder**

---

### 결과
- 최우수상
- https://www.youtube.com/watch?v=TXc53OP69oo
- 




