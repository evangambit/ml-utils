import os
import time
import requests
import sqlite3
from urllib.parse import urlparse
from urllib.parse import urldefrag

import os
import re
import sqlite3
import time
from urllib.parse import urldefrag, urlparse

from requests_html import HTMLSession

class Throttler:
  def __init__(self, qps):
    self.wait_time = 1. / qps
    self.last_call = time.time() - self.wait_time

  def throttle(self):
    next_time = self.last_call + self.wait_time
    time.sleep(max(next_time - time.time(), 0.0))
    self.last_call = time.time()

"""
Example

class WikiCrawler(CreepyCrawly):
  def initialize(self):
    self.c.execute("CREATE TABLE IF NOT EXISTS pages (title STRING)")

  def process_page(self, page_url : str, response):
    urls = self.get_all_urls(page_url, response)
    for url in urls:
      if re.match(r"https://en.wikipedia.org/wiki/.+", url):
        self.add_page(url)
    
    title = response.html.find('h1#firstHeading', first=True).text
    self.c.execute("INSERT INTO pages (title) VALUES (?)", (title,))

crawler = WikiCrawler(
  dirname='crawl-tmp',
  qps=1.0,
  commitEvery=10,
)
crawler.add_page("https://en.wikipedia.org/wiki/Milton_Friedman")

while crawler.scrape():
  pass

"""
class CreepyCrawly:
  def __init__(self, dirname : str, qps : float, seedUrls : [str] = [], commitEvery : int = 1):
    self.it = 0
    self.commitEvery = commitEvery
    self.dirname = dirname
    self.throttler = Throttler(qps)
    self.session = HTMLSession()
    if os.path.exists(dirname):
      self.conn = sqlite3.connect(os.path.join(dirname, 'db.db'))
      self.c = self.conn.cursor()
    else:
      os.mkdir(dirname)
      self.conn = sqlite3.connect(os.path.join(dirname, 'db.db'))
      self.c = self.conn.cursor()
      self.c.execute('CREATE TABLE urls (url BLOB PRIMARY KEY, visited INTEGER)')
      self.c.execute('CREATE INDEX visistedIndex ON urls(visited, url)')
      for url in seedUrls:
        self.c.execute('INSERT INTO urls (url, visited) VALUES (?, 0)', (url,))
      self.initialize()
  
  def initialize(self):
    """
    Optional initialization code (e.g. create relevant SQL tables)
    """
    pass
  
  def process_page(self, url, response):
    raise NotImplementedError('')

  def next_page(self):
    self.c.execute('SELECT url FROM urls WHERE visited = 0 LIMIT 1')
    r = self.c.fetchone()
    if r is None:
      return None
    return r[0]
  
  def get_all_urls(self, current_url, response, ignore_query = False, add_domain_if_necessary = True):
    urls = set([a.attrs.get('href', None) for a in response.html.find('a')])
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
    
    return set(urls)    
    
  def mark_page_finished(self, url : str):
    self.c.execute("UPDATE urls SET visited = 1 WHERE url = ?", (url,))

  def add_page(self, url : str):
    self.c.execute("INSERT OR IGNORE INTO urls (url, visited) VALUES (?, 0)", (url,))
  
  def scrape(self):
    self.it += 1
    if self.it % self.commitEvery == 0:
      self.commit()
    url = self.next_page()
    if url is None:
      return False
    self.throttler.throttle()
    try:
      r = self.session.get(url)
    except KeyboardInterrupt as e:
      raise e
    except:
      print(f'Error getting "{url}"')
      self.mark_page_finished(url)
      return True
    self.process_page(url, r)
    self.mark_page_finished(url)
    return True

  def commit(self):
    self.conn.commit()
