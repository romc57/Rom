from selenium import webdriver
from selenium.webdriver.chrome.options import Options

opt = Options()
opt.add_argument(r'user-data-dir=chromeHome')
opt.add_argument(r'--headless')
wassup_url = r'https://web.whatsapp.com/send?phone=+972507148309&text&app_absent=0'
driver = webdriver.Chrome(r'/home/rom/Downloads/chromedriver_linux64/chromedriver', options=opt)
driver.get(wassup_url)

