import io
import json
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from app.main import app

client = TestClient(app)

def make_png(color=(120, 140, 160)):
    arr = np.full((300, 300, 3), color, dtype=np.uint8)
    arr[40:100, 40:120] = [200, 60, 60]
    arr[140:200, 160:240] = [60, 200, 60]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG')
    return buf.getvalue()

png1 = make_png((100, 120, 140))
png2 = make_png((130, 150, 170))

queries = [
    ('Detect all vehicles.', False),
    ('Find all aircraft.', False),
    ('Find all buildings.', False),
    ('Segment all buildings.', False),
    ('How many vehicles are present?', False),
    ('What changed between these two images?', True),
    ('Which buildings have changed?', True),
    ('What percentage of the image changed?', True),
]

print('=' * 80)
print('SATQUERY AI — ACCEPTANCE TEST QUERIES VERIFICATION')
print('=' * 80)

for q, dual in queries:
    files = {'image': ('t1.png', png1, 'image/png')}
    if dual:
        files['image2'] = ('t2.png', png2, 'image/png')
    
    resp = client.post('/api/v1/analyze', files=files, data={'query': q})
    assert resp.status_code == 200, f'Failed: {resp.status_code} {resp.text}'
    d = resp.json()
    print(f'\nQUERY: "{q}"')
    print(f'  Status Code : {resp.status_code}')
    print(f'  Request ID  : {d.get("request_id")}')
    print(f'  Task/Op     : {d.get("task")} / {d.get("operation")}')
    print(f'  Models Used : {d.get("models_used")}')
    print(f'  Stats       : {d.get("statistics")}')
    print(f'  Detections  : {len(d.get("detections", []))}')
    print(f'  Masks       : {len(d.get("masks", []))}')
    print(f'  Changes     : {len(d.get("changes", []))}')
    ans = d.get('answer', '').replace('\n', ' ')
    print(f'  Answer      : {ans[:100]}...')

print('\n' + '=' * 80)
print('ALL 8 ACCEPTANCE QUERIES EXECUTED AND VERIFIED SUCCESSFULLY!')
print('=' * 80)
