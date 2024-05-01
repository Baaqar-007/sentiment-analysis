#Valence Aware Dictionary and Sentiment analysis 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 



from nltk.sentiment import SentimentIntensityAnalyzer 
from tqdm.notebook import tqdm
from intro import df

sia = SentimentIntensityAnalyzer()

# print(sia.polarity_scores('I am very angry')) # returns a dictionary
# print(sia.polarity_scores('I am so delighted'))

res = {}
for i,row in tqdm(df.iterrows(), total = len(df)):
	text = row['Text']
	my_id = row['Id']
	res[my_id] = sia.polarity_scores(text)

vaders = pd.DataFrame(res).T

vaders = vaders.reset_index().rename(columns={'index':'Id'})
vaders = vaders.merge(df, how = 'left')

#Plot vaders result 

# ax = sns.barplot(data=vaders, x = "Score", y = "compound")
# ax.set_title('Compund Score by Amazon Star Review')
# plt.show() 

fig,axs = plt.subplots(1,3, figsize =(15,5))
sns.barplot(data = vaders, x = 'Score', y = 'pos', ax = axs[0])
sns.barplot(data = vaders, x = 'Score', y = 'neu', ax = axs[1])
sns.barplot(data = vaders, x = 'Score', y = 'neg', ax = axs[2])

axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')

plt.show()

""" Doesn't take into consideration the relationship between words,
using a Pretraind Model like Roberta is preferred """