import scrapy
import re
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
#from mAxes.embeds.RoBERTa import *

remove_chaff = '.mw-parser-output table.biota-infobox{text-align:center;width:200px;font-size:100%}.mw-parser-output table.biota-infobox th.section-header{text-align:center}.mw-parser-output table.biota-infobox td.section-content{text-align:left;padding:0 0.25em}.mw-parser-output table.biota-infobox td.list-section{text-align:left;padding:0 0.25em}.mw-parser-output table.biota-infobox td.taxon-section{text-align:center;padding:0 0.25em}.mw-parser-output table.biota-infobox td.image-section{text-align:center;font-size:88%}.mw-parser-output table.biota-infobox table.taxonomy{margin:0 auto;text-align:left;background:transparent;padding:2px}.mw-parser-output table.biota-infobox table.taxonomy tr{vertical-align:top}.mw-parser-output table.biota-infobox table.taxonomy td{padding:1px}'

df = pd.read_csv("/Volumes/V'GER/comp_ling/Meta4/RSA/data/config/KAO-ANIMALS.csv")
df = df.replace('buffalo', 'bison')
animals = df['lex'].unique()

output = pd.DataFrame(columns=['lex', 'text'])
output.to_csv('mAxes/articles.txt', index=False, encoding='utf-8', sep='\t')



class spider(scrapy.Spider):

    name = 'wiki'
    start_urls = ['https://en.wikipedia.org/wiki/'+name for name in animals]

    data = {}

    def parse(self, response):
        #Get article title
        title = response.css('#firstHeading').extract()
        title = BeautifulSoup(title[0], 'html.parser')
        title = [i.text for i in title.findAll('h1')][0]
        print('======+======\n', title)

        #Get and process article text
        text = response.css('#mw-content-text').extract()
        text = BeautifulSoup(text[0], 'html.parser')
        text = text.findAll('p')
        text = ' '.join([tag.text for tag in text]).replace('\n', ' ').replace('\t', ' ').replace(remove_chaff, '').replace('killer whale', 'orca').replace('Killer whale', 'orca').lower()

        while '  ' in text:
            text = text.replace('  ', ' ')

        title.replace('Killer whale', 'orca').replace('killer whale', 'orca')
        #If we're saving everything . . . otherwise, we're going to have to
        # figure out how to do our roberta bit IN line. . .  sooo, step 2 :)
        output = pd.DataFrame(np.array([title.lower(), str(text)]).reshape(-1,2), columns=['lex','article'])
        output.to_csv('mAxes/articles.txt', mode='a', index=False, header=False, encoding='utf-8', sep='\t')
