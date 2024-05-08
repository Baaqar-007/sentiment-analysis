#Transformer based Pretrained Model 
# Hugging Face

from intro import df
from transformers import AutoTokenizer 
from transformers import AutoModelForSequenceClassification 
from scipy.special import softmax 
from tqdm.notebook import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 

sia = SentimentIntensityAnalyzer()



MODEL = f"cardiffnlp/twitter-roberta-base-sentiment" # Transfer learning
tokeniser = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# example = df['Text'][50]
# encoded_text = tokeniser(example,return_tensors='pt') # 1,0 embeddings
# #print(encoded_text)

# output = model(**encoded_text)
# scores = output[0][0].detach().numpy()

# scores = softmax(scores)
# scores_dict = {
# 	'roberta_neg' : scores[0],
# 	'roberta_neu' : scores[1],
# 	'roberta_pos' : scores[2]
# }

def polarity_scores_roberta(example):
	encoded_text = tokeniser(example,return_tensors='pt') # 1,0 embeddings
	#print(encoded_text)

	output = model(**encoded_text)
	scores = output[0][0].detach().numpy()

	scores = softmax(scores)
	scores_dict = {
		'roberta_neg' : scores[0],
		'roberta_neu' : scores[1],
		'roberta_pos' : scores[2]
	}
	return scores_dict 

res = {}
for i,row in tqdm(df.iterrows(), total = len(df)):
	try:
		text = row['Text']
		my_id = row['Id']
		vader_result = sia.polarity_scores(text)
		# renaming keys 
		vader_result2 ={}
		for key,value in vader_result.items():
			vader_result2[f"vader_{key}"] = value
		roberta_result = polarity_scores_roberta(text)
		res[my_id] = vader_result2 | roberta_result
	except RuntimeError:
		print(f"Broke for id {my_id}")


results = pd.DataFrame(res).T

results = results.reset_index().rename(columns={'index':'Id'})
results = results.merge(df, how = 'left')

#Comparison plot 
sns.pairplot(data=results,vars =['vader_neg', 'vader_neu', 'vader_pos',
       'roberta_neg', 'roberta_neu', 'roberta_pos'],
       hue ='Score',
       palette = 'tab10')
plt.show()

#Reviewing examples

# Positive 1 star reviews: (Vader performs poorely with sarcastic , nuanced reviews)
x = results.query('Score == 1').sort_values('roberta_pos', ascending = False)['Text'].values[0]
print(x)
y = results.query('Score == 1').sort_values('vader_pos', ascending = False)['Text'].values[0]
print(y)

# Negative 5 star reviews:
x = results.query('Score == 5').sort_values('roberta_neg', ascending = False)['Text'].values[0]
print(x)
y = results.query('Score == 5').sort_values('vader_neg', ascending = False)['Text'].values[0]
print(y)
