'''
crawl: https://www.tracuuphapluat.info/
'''
import requests
from bs4 import BeautifulSoup

url = "https://www.tracuuphapluat.info/2013/07/toan-van-luat-phong-chong-thien-tai-nam.html"

response = requests.get(url)
print(response.text)
# soup = BeautifulSoup(response.content, "html.parser")
soup = BeautifulSoup(response.read().decode('utf-8'))
text = soup.p
print(text)
