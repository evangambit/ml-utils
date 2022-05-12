import os
import time
import requests
import sqlite3
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from urllib.parse import urldefrag

class Throttler:
  def __init__(self, qps):
    self.wait_time = 1. / qps
    self.last_call = time.time() - self.wait_time
  def throttle(self):
    next_time = self.last_call + self.wait_time
    time.sleep(max(next_time - time.time(), 0.0))
    self.last_call = time.time()

"""
Base class for crawling and resuming a crawl part-way through.

Subclasses should implement "process_page" and (optionally) initialize

Example:

```
import re
class WikiCrawler(CreepyCrawly):
  def __init__(self, seedUrls, qps, dirname):
    super().__init__(seedUrls, qps, dirname)

  def process_page(self, page_url, html, soup):
    urls = self.get_all_urls(page_url, soup)
    for url in urls:
      if re.match(r"https://en.wikipedia.org/wiki/.+", url):
        self.add_page(url)
    
    title = soup.find_all('h1', {'id': 'firstHeading'})[0]
    # do stuff with the page
```


"""
class CreepyCrawly:
  def __init__(self, seedUrls, qps, dirname):
    """
    This is a javadoc style.

    @param seedUrls ([str]): list of urls to start crawling from
    @param qps (float): querys per second
    @param qps (str): output directory name
    """
    self.dirname = dirname
    self.throttler = Throttler(qps)
    if os.path.exists(dirname):
      self.conn = sqlite3.connect(os.path.join(dirname, 'db.db'))
      self.c = self.conn.cursor()
    else:
      os.mkdir(dirname)
      self.conn = sqlite3.connect(os.path.join(dirname, 'db.db'))
      self.c = self.conn.cursor()
      self.c.execute('CREATE TABLE pages (url BLOB PRIMARY KEY, visited INTEGER)')
      self.c.execute('CREATE INDEX visistedIndex ON pages(visited, url)')
      for url in seedUrls:
        self.c.execute('INSERT INTO pages (url, visited) VALUES (?, 0)', (url,))
      self.initialize()
  
  def initialize(self):
    """
    Optional initialization code (e.g. create relevant SQL tables)
    """
    pass
  
  def process_page(self, url, html, soup):
    raise NotImplementedError('')
  
  def next_page(self):
    self.c.execute('SELECT url FROM pages WHERE visited = 0 LIMIT 1')
    r = self.c.fetchone()
    if r is None:
      return None
    return r[0]
  
  def get_all_urls(self, current_url, soup, ignore_query = False, add_domain_if_necessary = True):
    urls = set([a.get('href') for a in soup.find_all('a')])
    if None in urls:
      urls.remove(None)
    if '' in urls:
      urls.remove(None)
    
    # Remove "#foo" from url
    urls = [urldefrag(url)[0] for url in urls]

    if add_domain_if_necessary:
      # Add domain if necessary
      parsedUrl = urlparse(current_url)
      domain = parsedUrl.scheme + '://' + parsedUrl.netloc
      tmp = []
      for url in urls:
        if url == '':
          continue
        if url[:2] == '//':
          tmp.append(parsedUrl.scheme + ':' + url)
        elif url[0] == '/':
          tmp.append(domain + url)
        else:
          tmp.append(url)
      urls = tmp
    
    if ignore_query:
      tmp = []
      for url in urls:
        parsed = urlparse(url)
        tmp.append(parsed.scheme + '://' + parsed.netloc + parsed.path)
      urls = tmp
    
    return urls    
    
  def mark_page_finished(self, url):
    self.c.execute("UPDATE pages SET visited = 1 WHERE url = ?", (url,))

  def add_page(self, url):
    self.c.execute("INSERT OR IGNORE INTO pages (url, visited) VALUES (?, 0)", (url,))
  
  def scrape(self):
    url = self.next_page()
    if url is None:
      return False
    self.throttler.throttle()
    r = requests.get(url)
    soup = BeautifulSoup(r.content)
    self.process_page(url, r.content, soup)
    self.mark_page_finished(url)
    return True

