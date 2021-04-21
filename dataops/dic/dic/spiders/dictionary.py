import scrapy
import re
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

RESTART = True


# adjectives = pd.read_csv('nn.csv')
# adjectives = adjectives['lex'].unique()[:3]
#adjectives=['cutthroat', 'gentle']

#adjectives = pd.read_csv("/Volumes/V'GER/comp_ling/Meta4/RSA/data/config/KAO-ANIMALS.csv")
# adjectives = pd.read_csv("/Volumes/V'GER/comp_ling/Meta4/RSA/dataops/dic/re-search.csv")
# adjectives = adjectives['lex'].unique()

adjectives =['small',
 'busy',
 'lazy',
 'intimidating',
 'blind',
 'night-time',
 'free',
 'liberty',
 'restrained',
 'awkward',
 'free-spirited',
 'fast',
 'friendly',
 'social',
 'unfriendly',
 'scaly',
 'naive',
 'vicious',
 'rapid',
 'cold',
 'clean',
 'awesome']

adjectives = adjectives[:3]

df_path = 'data.csv'
df0 = pd.DataFrame(columns=['lex', 'entry', 'text'])
if RESTART:
    df0.to_csv(df_path, index=False, encoding='utf-8')


class defSpider(scrapy.Spider):

    name = 'defs'
    start_urls = ['https://www.yourdictionary.com/'+adj for adj in adjectives]

    data = {}

    def parse(self, response):
        print('\n')
        title = response.css('.source-heading').extract()
        title = BeautifulSoup(title[0], 'html.parser')
        title = title.text.split()[0]

        spans = response.css('.single-definition-box').extract()
        spans = [BeautifulSoup(str(span), 'html.parser') for span in spans]
        spans = [span for span in spans if span.find('div', {'class': 'pos'}).text == 'adjective']
        spans = [span.find('span', {'class': 'definition flex-align-self-center'}).text.replace('. ', '.') for span in spans]
        spans = [[title, 'def', span] for span in spans]

        df = pd.DataFrame(np.array(spans).reshape(-1,3), columns=list(df0))
        df.to_csv(df_path, index=False, header=False, mode='a', encoding='utf-8')

        print('===========]{}|{}[==========='.format(title, len(spans)))

class sentSpider(scrapy.Spider):

    name = 'sents'
    start_urls = ['https://sentence.yourdictionary.com/'+adj for adj in adjectives]

    data = {}

    def parse(self, response):
        print('\n')
        data = response.css('.content-grid').extract()
        data = BeautifulSoup(data[0], 'html.parser')

        title = data.findAll('h1')
        title = title[0].text.split()[0]

        sents = data.findAll('div', {'class': 'sentence component'})
        sents = [[title.lower(), 'example', sent.text] for sent in sents]

        print('===========]{}|{}[==========='.format(title, len(sents)))

        df = pd.DataFrame(np.array(sents).reshape(-1, 3), columns=list(df0))
        df.to_csv(df_path, index=False, header=False, mode='a', encoding='utf-8')



class sent2Spider(scrapy.Spider):

    name = 'sents2'
    start_urls = ['https://sentence.yourdictionary.com/'+adj for adj in adjectives]

    data = {}

    def parse(self, response):
        print('\n')
        data = response.css('.content-grid').extract()
        # data = response.css('body').extract()
        data = BeautifulSoup(data[0], 'html.parser')
        print(len(data), len(data.findAll('div', {'class': 'sentence component'})), len(data.findAll('div', {'class':'head-word'})), len(data.findAll('span')))

        print('==========][==========')

        # title = data.findAll('.h1')
        # title = title[0].text.split()[0]
        # title='youdidit'
        #
        # sents = data.findAll('div', {'class': 'sentence-container'})
        # print(sents.children)
        # #sents = [[title.lower(), 'example', sent.text] for sent in sents]
        #
        # print('===========]{}|{}[==========='.format(title, len(sents)))

        # df = pd.DataFrame(np.array(sents).reshape(-1, 3), columns=list(df0))
        # df.to_csv(df_path, index=False, header=False, mode='a', encoding='utf-8')

