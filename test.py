# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests
import base64
from io import BytesIO
from PIL import Image

model_inputs = {'prompt': 'the source of future growth dramatic, elaborate emotive metallic Baroque and Rococo styles to emphasise death as a transcendental, sharp focus, epic scene, 8K'}

res = requests.post('http://localhost:8000/', json = model_inputs)

image_byte_string = res.json()["image_base64"]

image_encoded = image_byte_string.encode('utf-8')
image_bytes = BytesIO(base64.b64decode(image_encoded))
image = Image.open(image_bytes)
image.save("output.jpg")