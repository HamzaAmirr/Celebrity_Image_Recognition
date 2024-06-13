# Google Images Scraper

The Python script "web scraping.py" is designed to scrape images of a search result from Google Images using Selenium and BeautifulSoup libraries. Here's a breakdown of how it works:

## 1. Importing Libraries

The script starts by importing necessary libraries such as `bs4` for web scraping, `requests` for making HTTP requests, `webdriver` from Selenium for browser automation, `os` for file and directory operations, `time` for adding delays, `base64` for decoding base64-encoded images, and `Image` from PIL for handling images.

## 2. Folder Creation

It defines a folder name where the downloaded images will be saved. If the folder doesn't exist, it creates one.

## 3. Download Image Function

The `download_image` function is responsible for downloading images from URLs or base64-encoded strings. It checks if the URL starts with `data:image/`, indicating a base64-encoded image. If so, it decodes the image and saves it to the specified folder. Otherwise, it downloads the image from the URL using `requests.get()` and saves it.

## 4. Chrome Driver Path

The "chrome_path" variable contains the path to the Chrome driver. Change it accordingly. Make sure the Chrome Driver you have is the same as the version of Chrome on your system. You can download Chrome Driver from [here](https://googlechromelabs.github.io/chrome-for-testing/).

## 5. Search URL

The "search_URL" variable contains a search query. Change it according to your need.

## 6. User Input

The `a = input("Waiting...")` lets the script wait for user input to continue, allowing the user to scroll as far below as desired to download the images.

## 7. XPath Extraction

Once the page loads, use the inspect option built-in Chrome to extract the XPath and the previewImageXPath.

### Finding the XPath Variable


![image](https://github.com/HamzaAmirr/Deep-Learning/assets/122119582/eb4431e1-3d82-4721-8ede-c05bf2d0a844)


This XPath will be the "xPath" variable. Copy at least two of the XPaths and identify which number is changing, then replace that number with `%s`.

### Finding the previewImageXPath Variable

![image](https://github.com/HamzaAmirr/Deep-Learning/assets/122119582/640545c2-6835-4202-922d-1c1b4fa41310)

This XPath will be the "previewImageXPath" variable. Copy at least two of the XPaths and identify which number is changing, then replace that number with `%s`.

Once both these XPaths are given, give the input, and you will see that the images are being downloaded one by one in the folder specified.

Output would be something like this:

![image](https://github.com/HamzaAmirr/Deep-Learning/assets/122119582/3d5ea393-44d4-4065-8c27-509fba1e5cc8)

Note: The folder where images would be downloading is made in the location from where the script is running. Mostly it is in the C drive (for Windows). To locate the folder, simply search the folder name in the Windows search, and you’ll easily find it.
