import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


plt.style.use('ggplot')

import nltk

#Reading data 
df = pd.read_csv('../data/Reviews.csv') # ".." to move up a directory then search

#print(df.head())

df = df.head(500) # getting the first 500 test 


#Checking the data distribution:

if "__name__" == "__main__":
	ax = df['Score'].value_counts().sort_index().plot(kind='bar', 
		title ='Count of Reviews by Stars',
		figsize = (10,5))

	ax.set_xlabel('Review Stars')
	plt.show()

	#Mostly positive reviews 


	#Basic NLP:
	example = df['Text'][50]
	print(example)

	#Tokenising 
	tokens = nltk.word_tokenize(example) # splits it into tokens ("words")

	tags = nltk.pos_tag(tokens[:10]) # part of speech tags
	print(tags)

	entity = nltk.chunk.ne_chunk(tags)
	print(entity.pprint()) 




