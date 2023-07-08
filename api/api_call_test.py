import requests
import base64

image_path = "artifacts/test.jpg"
image_path = "artifacts/PSHIPERCELULARIDADE20200802-10.JPG"

# Open the image file and read the contents
with open(image_path, "rb") as image_file:
    image_bytes = image_file.read()

base64_image = base64.b64encode(image_bytes).decode("utf-8")

url = "http://localhost:5000/predict"
# url = "http://localhost:5001/image-query"

data = {"image": { "data": base64_image, "format":"pn"} }

headers = {"Content-Type": "application/json"}
print("Sending request for model...")
# print(f"Data: {data}")
r = requests.post(url, json=data, headers=headers)
print(f"Response: {r.text}")
print(f"Response: {r.status_code}")
