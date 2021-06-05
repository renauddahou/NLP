import requests
from bs4 import BeautifulSoup




class Blog_Data:
  def __init__(self,url):
    _res = requests.get(url)
    _html_page = _res.content
    self.soup = BeautifulSoup(_html_page, 'html.parser')

  def text_prep(self,req):
    text = self.soup.find_all(text=True)
    # set([t.parent.name for t in text])
    Text=""
    for t in text:
      if t.parent.name in req:
        Text+=t

    Text=' '.join(Text.split()[:200])

    return Text