
## 📌 목차
1. [🖥️ 프로젝트 개요](#%EF%B8%8F-프로젝트-개요)
2. [🗂️ 프로젝트 범위](#%EF%B8%8F-프로젝트-범위)
3. [📖 사용 라이브러리](#-사용-라이브러리)
4. [⚙️ 프로세스](#%EF%B8%8F-프로세스)
5. [📊 결과물](#-결과물)
6. [🗃️ 자료](#%EF%B8%8F-자료)
<br>

## 🖥️ 프로젝트 개요
### :calendar: 분석 기간
  - 2025.01.02 ~ 2025.01.07(총 5일)

### 🧑‍🤝‍🧑 인원
  - 1명 (개인 프로젝트 진행)

### 🔖프로젝트 소개

> ***기상 정보를 활용한 공공자전거 수요분석 및 예측***

#### ✅ 추진 배경
- 공공자전거 이용의 증가와 함께, 기상정보를 활용한 수요 예측의 필요성이 점차 커지고 있음
- 기상정보는 자전거 이용에 직접적인 영향을 미치는 요소로 다양한 기상 변수들이 자전거 이용자의 선택에 영향을 줄 수 있음
- 공공자전거 시스템의 효율성을 높이기 위해, 기상정보를 활용하여 수요를 예측하고, 자전거 대여소의 운영 및 관리에 대한 전략을 수립하고자 함

#### ✅ 목적
- 운영 효율성을 높이고 경제적 이익을 극대화할 수 있는 방안을 모색
- 효과적인 환경 관리 정책 및 전략을 수립할 수 있도록 지원하며, 지속 가능한 발전을 위한 실질적인 방안을 제시

#### ✅ 기대효과 
- #### 자원 관리 효율성 증대
  >- 기상정보를 활용하여 자전거 수요를 예측하고, 효율적으로 배치함으로써 자원 낭비 감소
  >- 예측을 통해 유지보수 시기를 적절히 조정하여 자전거의 가용성을 높이고, 운영 비용을 절감

- #### 환경적 효과
  >- 공공자전거 이용이 증가함에 따라 자동차 이용이 줄어들고, 이는 대기 오염 감소에 기여

- #### 사회적 비용 절감
  >- 자전거 이용이 증가함에 따라 도로의 교통 혼잡이 줄어들고, 이는 교통사고 및 대기 오염을 감소시켜 사회적 비용을 절감
  >- 자전거 이용이 증가함에 따라 시민들의 신체 활동이 늘어나고, 이는 건강 증진 및 의료비 절감  

<br>

[📌 목차로 이동](#-목차)
<br><br>

## 🗂️ 프로젝트 범위
<div style="text-align: center;">
<table>
<tr><th colspan="2">과제 구분</th><th>내용</th></tr>
<tr><td rowspan="7">AI</td><td rowspan="7" align='center'>AI기반 공공자전거 수요분석, <br>예측모델 구현 및 시각화</td><td align='center'>원시 데이터 수집 및 데이터셋 구축</td></tr>
<tr><td align='center'>데이터 전처리, 표준화, 상관관계 분석 (EDA도구 활용)</td></tr>
<tr><td align='center'>예측모델 선정 및 학습</td></tr>
<tr><td align='center'>MSE, R-Squared 등 평가지표를 활용한 모델 성능 평가</td></tr>
<tr><td align='center'>웹 API 및 프로토타입 구충</td></tr>
<tr><td align='center'>예측모델 시각화 및 웹기반 시스템 구축</td></tr>
<tr><td align='center'>테스트</td></tr>
</table>
</div><br>

[📌 목차로 이동](#-목차)
<br><br>

## 📖 사용 라이브러리
|라이브러리|모델|기능|설명|
|:---:|:---:|:---:|:---:|
|numpy|-|데이터 수치 계산|데이터의 수치적 연산 수행|
|pandas|-|데이터 조작 및 분석|데이터 읽기, 쓰기, 필터링 등을 지원|
|matplotlib|-|데이터 시각화|다양한 유형의 그래프 제공|
|seaborn|-|Matplotlib기반 시각화|통계적 그래프 시각화 지원|
|sklearn|train_test_split|머신러닝|데이터를 훈련, 테스트 세트로 분류|
|sklearn|r2_score|머신러닝|결정 계수(R²)를 계산|
|sklearn|mean_squared_error|머신러닝|평균 제곱 오차(MSE)를 계산|
|sklearn|StandardScaler|머신러닝|데이터를 표준화|
|sklearn|LinearRegression|머신러닝|선형 회귀 모델을 구현하|
|sklearn|RandomForestClassifier|머신러닝|랜덤 포레스트 분류 모델을 구현|
|sklearn|RandomForestRegressor|머신러닝|랜덤 포레스트 회귀 모델을 구현|
|xgboost|XGBRegressor|회귀 및 분류|XGBoost를 기반으로 한 회귀 모델|
<br>

[📌 목차로 이동](#-목차)
<br><br>

## ⚙️ 프로세스
- #### 데이터 수집
  >- 서울열린데이터광장 서울시 공공자전거 이용정보(일별)
  >- 기상자료 개방포털 서울시 기상관측 데이터(일별)
- #### 데이터 전처리
  >- 데이터 표준화 및 전처리(결측치, 이상치 처리)
  >- 변수를 표준화 및 정규화하여 모델학습 효율성을 향상
  >- 상관관계 분석을 통해 독립변수 간의 관계를 확인
  >- 특성 중요도를 통해 중요도가 낮은 독립변수 제거
- #### 데이터 모델링
  >- 모델 성능 평가를 위해 MSE, r-squared 등의 지표를 사용
  >- 모델 비교 및 최적 모델 도출 (XGBoost 채택)
- #### 데이터 예측
  >- 다양한 기상요인에 따른 공공자전거 수요 예측
- #### 결과 시각화 및 분석
  >- 모델의 정확도 평가를 위해 실제값과 예측값을 비교
  >- 예측결과에 대한 분석 결과를 산점도로 시각화
  >- 실제값과 예측값 간의 차이를 바탕으로 잔차 분석 진행
  >- 기상요인에 따른 공공자전거 수요 예측을 통한 운영 효율성 향상 및 환경적 기여

<br>

[📌 목차로 이동](#-목차)
<br><br>

## 📊 결과물
<details>
  <summary><b>1. 데이터 수집</b> (👈 Click)</summary>
  <br>
  <li>
    서울시 공공자전거 일별 이용 현황 데이터(엑셀) : 2023년
  </li>
  <li>
    서울시 기상관측 일병 기상 데이터(엑셀) : 2023년
  </li>

  |공공자전거 데이터|기상관측 데이터|
  |:---:|:---:|
  |<img src="https://github.com/user-attachments/assets/4101d1ac-1d46-4f9a-9538-00282674518a" width="300" alt="데이터1">|<img src="https://github.com/user-attachments/assets/04e230dc-0a6b-4571-88b9-603fcef82327" width="300" alt="데이터2">|
  <br>
</details>
<details>
  <summary><b>2. 데이터 분석</b> (👈 Click)</summary>
  <br>
  <ol>
    <li>
      데이터 상관관계(Heatmap)
    </li><br>
    <img src="https://github.com/user-attachments/assets/83748058-4db5-4099-b0de-cfedcd2d1416" width="400" alt="히트맵"><br>
    <li>
      탐색적 데이터 분석
    </li>
    <ul>
      <li>
        결측치 및 중복값 통계
      </li><br>
      <img src="https://github.com/user-attachments/assets/fffa7e15-d5b1-407a-a0b8-653b064bb634" alt="결측치"><br><br>
      <li>
        주요 변수별 데이터 분포(Histogram)
      </li><br>
      <img src="https://github.com/user-attachments/assets/7f1327cc-4eb1-4e50-8856-543e1203ff8d" alt="분포도"><br><br>
      <li>
        데이터 전처리
      </li><br>
      <img src="https://github.com/user-attachments/assets/d8bb8085-fcdd-4dfa-b922-99d0044749f3" alt="분포도"><br>
    </ul>
  </ol>
</details>
<details>
  <summary><b>3. 데이터 학습 및 모델정의 </b> (👈 Click)</summary>
  <br>
  <ol>
    <li>예측 모델 선정</li>
    <ul>
      <li>결정계수 비교 : Ensemble 기법 중 하나인 XGBoost 모델 채택</li><br>
      <img src="https://github.com/user-attachments/assets/97b29c7c-8c93-4029-bd20-979bc6023a9b" alt="모델 선정">
    </ul>
    <li>모델 학습 및 시각화</li>
    <ul>
      <li>모델 학습</li><br>
      <img src="https://github.com/user-attachments/assets/c964e95e-fabd-4fab-af88-94468cb5cee4" alt="모델 학습"><br><br>
      <li>학습과정 시각화</li><br>
      <img src="https://github.com/user-attachments/assets/fdfa1423-93c2-4f1b-87e9-cfa683cb59c4" alt="모델 시각화"><br>
    </ul>
    <li>모델 예측</li>
    <ul>
      <li>예측값 vs 실제값 비교</li><br>
      <ul>
        <li>선 그래프 비교</li><br>
        <img src="https://github.com/user-attachments/assets/40662084-42f6-4e33-b72d-55dbb349ac7a" alt="선 그래프"><br><br>
        <li>산점도 분석</li><br>
        <img src="https://github.com/user-attachments/assets/3754e41b-4c30-4133-a514-149e82f98918" alt="산점도"><br><br>
        <li>잔차 분석</li><br>
        <img src="https://github.com/user-attachments/assets/c7cf2f15-b69d-424b-a46d-49bc50297f94" alt="잔차"><br><br>
      </ul>
    </ul>
  </ol>
  
  <br>
</details>
<details>
  <summary><b>4. 프로토타이핑(화면) </b> (👈 Click)</summary>
  <br>
  <ol>
  <li>모델 예측</li>
    <ul>
      <li>기상요인에 따른 공공자전거 이용건수 예측</li><br>
      <img src="https://github.com/user-attachments/assets/e0f4c19b-54a0-4452-a4ec-691b4443678f" alt="모델 예측"><br><br>
    </ul>
  <li>예측 결과</li>
    <ul>
      <li>기상요인에 따른 공공자전거 이용건수 예측</li><br>
      <img src="https://github.com/user-attachments/assets/75bf3c39-3b04-46cc-ae68-db4c27836cf4" alt="예측 자료"><br>
      <img src="https://github.com/user-attachments/assets/ab3156f7-e734-4f08-b18b-a8988d037e64" alt="예측 결과"><br>
    </ul>
  </ol>
</details>
<br>

[📌 목차로 이동](#-목차)
<br><br>

## 🗃️ 자료
[[📂 코드 및 자료]](https://drive.google.com/drive/folders/1LK1ONMXZGfyqcQqXVD0yvlwtoGq23Evx?usp=sharing)<br>

[📌 목차로 이동](#-목차)
