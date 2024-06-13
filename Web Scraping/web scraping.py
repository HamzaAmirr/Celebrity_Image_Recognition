import bs4
import requests
from selenium import webdriver
import os
import time
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import base64
from PIL import Image

# Create a directory to save images
folder_name = 'Mr Beast'
if not os.path.isdir(folder_name):
    os.makedirs(folder_name)

def download_image(url, folder_name, i):
    """
    Download an image from a URL or a base64-encoded string and save it to a specified folder.

    :param url: The URL or base64-encoded string of the image.
    :param folder_name: The name of the folder where the image will be saved.
    :param i: The index of the image in the list of images.
    """
    # Check if the URL is a base64-encoded string
    if url.startswith('data:image/'):
        # Decode the base64-encoded string
        format, imgstr = url.split(';base64,')
        ext = format.split('/')[-1]
        image_data = base64.b64decode(imgstr)

        # Write the decoded data to a file
        with open(os.path.join(folder_name, f"{i}.{ext}"), "wb") as f:
            f.write(image_data)
    else:
        # Download the image from a URL
        response = requests.get(url)
        if response.status_code == 200:
            with open(os.path.join(folder_name, f"{i}.jpg"), 'wb') as file:
                file.write(response.content)

chrome_path = r'E:\Image recognition\chromedriver-win64\chromedriver.exe'
service = Service(executable_path=chrome_path)
driver = webdriver.Chrome(service=service)

search_URL = "https://www.google.com/search?q=Jimmy+Donaldson&source=lnms&tbm=isch"
driver.get(search_URL)

a = input("Waiting...")

# Scroll to the top of the page
driver.execute_script("window.scrollTo(0, 0);")

page_html = driver.page_source
pageSoup = bs4.BeautifulSoup(page_html, 'html.parser')
containers = pageSoup.findAll('div', {'class':"eA0Zlc WghbWd FnEtTd mkpRId m3LIae RLdvSe qyKxnc ivg-i PZPZlf GMCzAd"})

len_containers = len(containers)
print(f'Found {len_containers} images')

for i in range(1, len_containers+1):
    if i % 25 == 0:
        continue
    xPath = """//*[@id="rso"]/div/div/div[1]/div/div/div[%s]"""%(i)
    previewImageXPath = """/html/body/div[4]/div/div[13]/div/div[2]/div[2]/div/div/div/div/div[1]/div/div/div[%s]/div[2]/h3/a/div/div/div/g-img/img"""%i
    previewImageElement = driver.find_element(By.XPATH, previewImageXPath)
    previewImageURL = previewImageElement.get_attribute("src")

    # Scroll the preview image into view
    previewImageElement = driver.find_element(By.XPATH, previewImageXPath)
    driver.execute_script("arguments[0].scrollIntoView(true);", previewImageElement)
    time.sleep(0.5)  # Add a short delay

    driver.find_element(By.XPATH, xPath).click()

    # Download the image
    try:
        download_image(previewImageURL, folder_name, i)
        print(f"Downloaded element {i} out of {len_containers + 1}. URL: {previewImageURL}")
    except:
        print(f"Couldn't download an image {i}, continuing downloading the next one")

while True:
    time.sleep(1)