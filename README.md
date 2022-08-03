# - 가맹점 분류기
## - 프로젝트 소개
![프로젝트 소개](https://user-images.githubusercontent.com/97024674/182273714-e00c0fe4-bacf-446a-9e64-f9c153cb0d0c.png)

## * 데이터 수집 및 전처리
![데이터 소개](https://user-images.githubusercontent.com/97024674/182273836-e6958ba0-b951-4367-9bdb-b771f3e56855.png)
![데이터 Labeling](https://user-images.githubusercontent.com/97024674/182273877-48076fa2-cfcb-4332-b05e-8a3625d2c21c.png)
![데이터 전처리 1](https://user-images.githubusercontent.com/97024674/182273903-85320329-cb08-40f6-b767-8de04d800e03.png)
![데이터 전처리 2](https://user-images.githubusercontent.com/97024674/182273942-8f4701f1-ecfa-42e7-83a2-0fd84c55b51d.png)
![데이터 전처리 3](https://user-images.githubusercontent.com/97024674/182273977-a47eb134-1d74-4146-9fe4-7bf10c793d48.png)

## * 실험
![실험 소개](https://user-images.githubusercontent.com/97024674/182274033-6e1334e2-3c55-4fce-85b3-3e9022024c9a.png)

### - 사용한 모델
![Characer CNN (1d CNN)](https://user-images.githubusercontent.com/97024674/182274112-5d59c826-9343-410b-9233-39a898c23a46.png)
![KoBERT](https://user-images.githubusercontent.com/97024674/182274156-38a2f2b7-cc3b-4e8b-8721-53476817b5fc.png)
![KoELECTRA](https://user-images.githubusercontent.com/97024674/182274203-fc0a8f85-6ae5-49bf-aa4a-fde2f26e5c2e.png)
![RoBERTa](https://user-images.githubusercontent.com/97024674/182274251-bb3dd747-6d6a-4b65-9930-89df17c63c1e.png)

### - 실험 결과
![1d CNN_1](https://user-images.githubusercontent.com/97024674/182274315-3f9bf381-967b-43b0-9621-f66bdb4093e3.png)
![1d CNN_2](https://user-images.githubusercontent.com/97024674/182274351-91e12945-14fc-4eae-8a7d-81eea74a9aa0.png)
![1d CNN_3](https://user-images.githubusercontent.com/97024674/182274408-8e7987c6-f2e1-408e-ab08-d651f6ca40b7.png)
![1d CNN_4](https://user-images.githubusercontent.com/97024674/182274437-10120f9e-7d0e-46ca-a3cf-6c64e2cbd7b5.png)
![1d CNN_5](https://user-images.githubusercontent.com/97024674/182274460-6d2e364d-c72b-464c-8c8d-6ff173b59f25.png)
![Pretrained Model 1](https://user-images.githubusercontent.com/97024674/182292833-c1f95d2e-bbe9-4dc6-9543-c82717e65d0d.png)
![Pretrained Model 2](https://user-images.githubusercontent.com/97024674/182292881-d52da320-ec1d-4b33-91c7-add99fd60cb9.png)
![Pretrained Model_3](https://user-images.githubusercontent.com/97024674/182274573-6717dc57-a542-4fff-a43f-bc7bad8fa9bf.png)
![Pretrained Model_4](https://user-images.githubusercontent.com/97024674/182274602-cc64b0a5-230c-4351-8c1e-b1bf88c46c1b.png)

## - 데모
`Streamlit`, `FastAPI`로 웹서비스 구현
## -Streamlit

### INTRO
INTRO 에서 hyperparameter 별 모델 성능 및 학습 과정에서의 metric 변화를 볼 수 있습니다.
![image](https://user-images.githubusercontent.com/97024674/182293427-03885000-3b03-413f-9917-e47f37061025.png)

### 간단 시연
Side Bar 에서 모델을 선택한 뒤, 업체명을 검색하면 예측된 업종과 그 확률을 보여줍니다.
![image](https://user-images.githubusercontent.com/97024674/182293631-d35efd5f-66a5-40dc-92e5-82279ec6f08b.png)

### 가계부
Side Bar 에서 모델을 선택하고, 거래일자, 업체명, 사용금액이 들어있는 csv 파일을 업로드하면 다양한 기능을 사용할 수 있습니다.
![image](https://user-images.githubusercontent.com/97024674/182293902-e3b7d35d-9f7d-47a1-82f4-63681409e9d6.png)


1. 전체 기간 동안 업종별 사용 금액 및 소비 금액 그래프
![image](https://user-images.githubusercontent.com/97024674/182293965-41cd357f-e398-4b48-bc9a-95f1aecdb387.png)
선택한 업종에 대해서만 보는 것도 가능합니다.
![image](https://user-images.githubusercontent.com/97024674/182294039-41bcce5a-ebe0-4678-b817-b6623ad3e8f3.png)

2. 업종 별, 요일 별 분석
 ![image](https://user-images.githubusercontent.com/97024674/182294123-3f3ce8e6-6f6b-4982-be2c-2b6474d293ab.png)
 마찬가지로 선택한 업종에 대해서만 보는 것도 가능합니다.
![image](https://user-images.githubusercontent.com/97024674/182295178-ca3a0aa1-c077-4949-848a-5c434f9015e1.png)

3. 전월 대비 분석
![image](https://user-images.githubusercontent.com/97024674/182295529-4270bee9-f075-400e-920e-1ae72c6d0e8f.png)
![image](https://user-images.githubusercontent.com/97024674/182295684-fbf4d8c5-523f-479e-a8ab-9642bfe4f316.png)


### - Directory Structure
<pre>
<code>
merchandise_classifier
├── fastapi
├── model
└── streamlit
</code>
</pre>


