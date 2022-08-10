# Streamlit Directory Structure
<pre>
<code>
...
└── streamlit
      ├── INTRO.py
      └── pages
             ├── 1_🗃️_간단_시연.py
             └── 2_💸_가계부.py
</code>
</pre>
*****

## Streamlit

### 실행 방법

<pre>
<code>
streamlit run streamlit/INTRO.py
</code>
</pre>

### INTRO
* INTRO 에서 hyperparameter 별 모델 성능 및 학습 과정에서의 metric 변화를 볼 수 있습니다.
![image](https://user-images.githubusercontent.com/97024674/182293427-03885000-3b03-413f-9917-e47f37061025.png)
***
### 간단 시연
* Side Bar 에서 모델을 선택한 뒤, 업체명을 검색하면 예측된 업종과 그 확률을 보여줍니다.
![image](https://user-images.githubusercontent.com/97024674/182293631-d35efd5f-66a5-40dc-92e5-82279ec6f08b.png)
***
### 가계부
* Side Bar 에서 모델을 선택하고, 거래일자, 업체명, 사용금액이 들어있는 csv 파일을 업로드하면 다양한 기능을 사용할 수 있습니다.
![image](https://user-images.githubusercontent.com/97024674/182293902-e3b7d35d-9f7d-47a1-82f4-63681409e9d6.png)


 - 전체 기간 동안 업종별 사용 금액 및 소비 금액 그래프
![image](https://user-images.githubusercontent.com/97024674/182293965-41cd357f-e398-4b48-bc9a-95f1aecdb387.png)
선택한 업종에 대해서만 보는 것도 가능합니다.
![image](https://user-images.githubusercontent.com/97024674/182294039-41bcce5a-ebe0-4678-b817-b6623ad3e8f3.png)

 - 업종 별, 요일 별 분석
 ![image](https://user-images.githubusercontent.com/97024674/182294123-3f3ce8e6-6f6b-4982-be2c-2b6474d293ab.png)
 마찬가지로 선택한 업종에 대해서만 보는 것도 가능합니다.
![image](https://user-images.githubusercontent.com/97024674/182295178-ca3a0aa1-c077-4949-848a-5c434f9015e1.png)

 - 전월 대비 분석
![image](https://user-images.githubusercontent.com/97024674/182295529-4270bee9-f075-400e-920e-1ae72c6d0e8f.png)
![image](https://user-images.githubusercontent.com/97024674/182295684-fbf4d8c5-523f-479e-a8ab-9642bfe4f316.png)

*****
