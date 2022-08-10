### - Model Directory Structure
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
