from utils.get_data import get_data
from utils.zip import extract_zip
from models.init import main

if __name__ == "__main__":
    url = "https://www.kaggle.com/api/v1/datasets/download/vetrirah/customer"
    try:
        get_data(url, 'data/external/customer.zip')
    except:
        print("An exception occurred")
    extract_zip('data/external/customer.zip')
    # run model
    main()
