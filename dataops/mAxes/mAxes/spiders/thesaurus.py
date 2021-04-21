import scrapy
import re
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
#from mAxes.embeds.RoBERTa import *

adjectives = pd.read_csv('mAxes/anchors.csv')
adjectives = adjectives['lex'].values
adjectives = ['weightless', 'hostile']

N=5

# dfO = pd.DataFrame(columns=['axis'])
# dfO.to_csv('mAxes/kao-axes-syns.csv', index=False, encoding='utf-8')

df0 = pd.DataFrame(columns=['root', 'lex'])
df0.to_csv('mAxes/axes-syns.csv', index=False, encoding='utf-8')

class spider(scrapy.Spider):

    name = 'thes'
    start_urls = ['https://thesaurus.com/browse/'+adj for adj in adjectives]

    data = {}

    def parse(self, response):
        title = response.css('h1.css-cs9r8m-Heading').extract()
        title = BeautifulSoup(title[0], 'html.parser')
        title = [i.text for i in title.findAll('h1')][0]

        synonyms = response.css('ul.css-17d6qyx-WordGridLayoutBox') .extract()
        synonyms = BeautifulSoup(synonyms[0], 'html.parser')
        synonyms = [i.text for i in synonyms.findAll('li')]
        print(len(synonyms))

        data = np.array(' '.join([title]+synonyms[:N])).reshape(1,-1)
        data = pd.DataFrame(data, columns=list(dfO))

        data.to_csv('mAxes/kao-axes-syns.csv', index=False, header=False, mode='a', encoding='utf-8')

class thesSpider(scrapy.Spider):

    name = 'thes2'
    start_urls = ['https://thesaurus.com/browse/'+adj for adj in adjectives]

    data = {}

    def parse(self, response):
        print('=======]open [=======')
        data = response.css('body').extract()
        print(len(data))
        data = BeautifulSoup(data[0], 'html.parser')

        title = data.findAll('div', {'id': 'headword'})
        title = title[0].findAll('h1')
        title = title[0].text

        meanings = data.findAll('div', {'id': 'meanings'})
        syns = meanings[-1].findAll('li')
        syns = [syn.text for syn in syns]

        syns = [[title, syn] for syn in syns]
        syns = np.array(syns)

        output = pd.DataFrame(syns, columns=list(df0))
        output.to_csv('mAxes/axes-syns.csv', index=False, header=False, encoding='utf-8', mode='a')

        print('=======]close[=======')


