import os
import requests


def get_data(url, output_path):
    # output_path örnek: data/external/customer.zip
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.isfile(output_path):
        print("Data already installed.")
        return

    response = requests.get(url, allow_redirects=True)

    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded file saved to {output_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
