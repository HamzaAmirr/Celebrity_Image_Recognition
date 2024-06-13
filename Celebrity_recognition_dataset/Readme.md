## Celebrities Included:

  - Brad Pitt
  - Chris Hemsworth
  - Christiano Ronaldo
  - Elon Musk
  - Eminem
  - Garry Kasparov
  - Jeff Bezos
  - Leonardo di Caprio
  - Lionel Messi
  - Magnus Carlsen
  - Mark Zuckerberg
  - Mike Tyson
  - Muhammad Ali
  - Novak Djokovic
  - Shahrukh Khan
  - Tom Holland
  - Will Smith

### Image Count:
  - The count of each celebrity's images is recorded in the CSV file named "Data count" located in the same folder.

## 2. Image Scraping

- A Python script was created to scrape images loaded from a particular search query for each celebrity.
- Libraries used: bs4, requests, selenium, os, time, and PIL.
- An average 250 images per celebrity were downloaded using the script 

## 3. Image Preprocessing

- Each downloaded image was manually edited to isolate the celebrity from the background.
- Methods used for isolation:
  - Cropping out other people from the picture.
  - Using the erase tool in images on Windows 10 to remove other people.
  - Cropping out a part of the picture and removing other people using the erase tool.
- The method chosen for each image was decided based on intuition without a set rule.

The combination of scraping and preprocessing created a dataset suitable for training the Celebrity Image Recognition model.
