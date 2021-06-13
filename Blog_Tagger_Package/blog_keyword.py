# -*- coding: utf-8 -*-
"""Keyword extraction with ALBERT

#Extracting keywords for tagging of blog post
"""
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from transformers import TFAutoModel,AutoTokenizer
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from web_data import Blog_Data

nltk.download('stopwords')


"""### step 1: candidate selection from text (blog post)

#### Dealing with unigram and bigrams candidate tokens only 
"""

def tokens(Text_data,stop_words):

  vector=CountVectorizer(ngram_range=(1,2), stop_words=stop_words).fit([Text_data])
  all_candidate_tokens=vector.get_feature_names()
  return all_candidate_tokens

"""#Candidate tokens are generated.
<h2>Now we have to keep important candidate tokens only.</h2>

<p>Candidate token selection using POS tagging, as we know that tags are going to be important words of the blog content.therefore it will probably be a noun.so we opt for POS tagging using spacy to eliminate tokens except ( POS: Noun)</p>
"""



class Blog_Tagger:
  def __init__(self,Text_data,maxlen):
    
    nlp=spacy.load('en_core_web_sm')
    Text_data=' '.join(Text_data.split()[:maxlen])
    self.Text_data=Text_data
    self.all_candidate_tokens=tokens(Text_data,stop_words=set(stopwords.words('english')))
    self.doc=nlp(self.Text_data)


  def token_embedding_gen(self,model,tokenizer):
    nouns = set()
    candidate_token_embeddings={}
    for token in self.doc:
        if token.pos_ == "NOUN":
            nouns.add(token.text)

    noun_phrases = set(chunk.text.strip().lower() for chunk in self.doc.noun_chunks)
    all_noun_candidate_tokens=nouns.union(noun_phrases)

    present_candidate_tokens=list(filter(lambda candidate_tokens : candidate_tokens in all_noun_candidate_tokens, self.all_candidate_tokens))

    """### Now let's create embeddings for text(blog post) and all filtered imp candidate tokens 
      <p>we use autoclass of the hugging face to call ALBERT model for creating embeddings</p>"""

    for token in present_candidate_tokens:
      token_id=tf.constant(tokenizer.encode(token,add_special_tokens=True))[None, :]
      candidate_token_embeddings[token]=model(token_id)['pooler_output'][0]

    text_tokens=tokenizer(self.Text_data,padding=True,return_tensors="tf")
    self.blog_text_embedding=model(text_tokens['input_ids'])['pooler_output']
    
    self.candidate_token_embeddings=candidate_token_embeddings

  """### Now using candidate embeddings and blog text embeddings we will use similarity measuring metric 
   cosine similarity from sklearn package to decide which tokens are best matched to the blog text"""

  def tag_gen(self,k):
    score={}
    for token,token_embed in self.candidate_token_embeddings.items():
      score[token]=cosine_similarity(np.array(token_embed).reshape(1,-1),np.array(self.blog_text_embedding))[0][0]  #compared to blog_text

    k_tag_score=sorted(score)
    return k_tag_score[-k:]



## Driver code for testing


# if __name__=='__main__':
#   data=Blog_Data("https://influencermarketinghub.com/12-best-food-blogs/")
#   Text_data=data.text_prep(req=['h1', 'h2', 'h3', 'h4', 'p'])
#   tagger=Blog_Tagger(Text_data,maxlen=<int num>)  #no of strings
#   model=TFAutoModel.from_pretrained('albert-base-v2')
#   tokenizer=AutoTokenizer.from_pretrained('albert-base-v2')
#   tagger.token_embedding_gen(model,tokenizer)
#   top_tokens=tagger.tag_gen(n)

#   print(top_tokens)






