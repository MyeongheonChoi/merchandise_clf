### Model Directory Structure
<pre>
<code>
...
└── model
      ├── config
      │     ├── inference_config.yaml
      │     ├── preprocess_config.yaml
      │     ├── split_config.yaml
      │     └── train_config.yaml
      ├── data
      │     ├── cased_test.csv
      │     ├── cased_train.csv
      │     ├── uncased_test.csv
      │     ├── uncased_train.csv
      │     ├── nonumber_test.csv
      │     ├── nonumber_train.csv
      │     └── preprocessing
      ├── modules
      │     ├── dataset.py
      │     ├── losses.py
      │     ├── metrics.py
      │     ├── optimizer.py
      │     ├── preprocess.py
      │     ├── split.py
      │     ├── trainer.py
      │     └── utils.py
      ├── train.py
      ├── inference.py
      └── inference_streamlit.py
</code>
</pre>

*****
## - 데이터 수집 및 전처리
![데이터 소개](https://user-images.githubusercontent.com/97024674/182273836-e6958ba0-b951-4367-9bdb-b771f3e56855.png)
![데이터 Labeling](https://user-images.githubusercontent.com/97024674/182273877-48076fa2-cfcb-4332-b05e-8a3625d2c21c.png)
![데이터 전처리 1](https://user-images.githubusercontent.com/97024674/182273903-85320329-cb08-40f6-b767-8de04d800e03.png)
![데이터 전처리 2](https://user-images.githubusercontent.com/97024674/182273942-8f4701f1-ecfa-42e7-83a2-0fd84c55b51d.png)
![데이터 전처리 3](https://user-images.githubusercontent.com/97024674/182273977-a47eb134-1d74-4146-9fe4-7bf10c793d48.png)
***
## - 실험
![실험 소개](https://user-images.githubusercontent.com/97024674/182274033-6e1334e2-3c55-4fce-85b3-3e9022024c9a.png)
***
### - 사용한 모델
![Characer CNN (1d CNN)](https://user-images.githubusercontent.com/97024674/182274112-5d59c826-9343-410b-9233-39a898c23a46.png)
![KoBERT](https://user-images.githubusercontent.com/97024674/182274156-38a2f2b7-cc3b-4e8b-8721-53476817b5fc.png)
![KoELECTRA](https://user-images.githubusercontent.com/97024674/182274203-fc0a8f85-6ae5-49bf-aa4a-fde2f26e5c2e.png)
![RoBERTa](https://user-images.githubusercontent.com/97024674/182274251-bb3dd747-6d6a-4b65-9930-89df17c63c1e.png)
***
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
*****
