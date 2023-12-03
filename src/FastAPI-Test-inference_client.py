import requests

local = "127.0.0.1"

port = "8000"

url = f"http://{local}:{port}"


# Basic GET request
response = requests.get(f'{url}/healthcheck')
print(response.json())

body = {
        "instances": [
            {"values": [0.1, 0.2, 0.3]},
            {"values": [0.6, 0.2, 1.3]}
        ],
        "parameters": {"param1": 1.54},
    }

# POST request with body
response = requests.post(
    f"{url}/predict",
    json=body,
)
print(response.json())