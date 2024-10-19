import requests

# Base URL of the Flask server
BASE_URL = "http://127.0.0.1:5000"  # Adjust the URL to your Flask server address

# 3. Sending a POST request to /post_example with JSON data
def send_post_example(data):
    response = requests.post(f"{BASE_URL}/get_gsnet_grasp", json=data)
    print("POST /get_gsnet_grasp response:", response.json())

# Example of how the client interacts with the server
if __name__ == "__main__":
    # Call the functions to interact with the Flask server
    send_post_example({"name": "Han", "age": 22})  # Sends a POST request with JSON data
