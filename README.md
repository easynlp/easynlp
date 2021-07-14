# easynlp

start server:
`uvicorn server.server:app --port 1234`

get responses:

```python
import requests
r = requests.post('http://localhost:1234/classification', json={'text': ['some text about football.'],
                                                                'labels': ['sports', 
                                                                           'weather',
                                                                           'business']
                                                                })
assert r.status_code == 200
assert r.json() == {'classification': ['sports']}
```
