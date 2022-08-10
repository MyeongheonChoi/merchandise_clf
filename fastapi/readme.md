# FastAPI Directory Structure
<pre>
<code>
...
└── fastapi
      └── main.py
</code>
</pre>
*****

### 실행 방법

<pre>
<code>
uvicorn fastapi.main:app --host=0.0.0.0 --port=8000
</code>
</pre>

-만약 host=127.0.0.1 을 사용한다면 streamlit에서 backend url을 localhost -> 127.0.0.1 로 
