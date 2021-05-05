## Preparation and Structuring Dataset 

### resume data
# pip install tika
### Extracting data for developer hiring"""
from tika import parser
from spacy.matcher import Matcher
import spacy
import numpy as np 
import pandas as pd

nlp=spacy.load('en_core_web_sm')
matcher=Matcher(nlp.vocab)

## buzz words

#for ai
pattern_1= [{"LOWER": "machine"}, {"LOWER": "learning"}]
pattern_2= [{"LOWER": "deep"}, {"LOWER": "learning"}]
pattern_3= [{"LOWER": "nlp"}]
pattern_4= [{"LOWER": "big"}, {"LOWER": "data"}]
pattern_5= [{"LOWER": "reinforcement"}, {"LOWER": "learning"}]

#for app dev
pattern_6= [{"LOWER": "android"}, {"LOWER": "studio"}]
pattern_7= [{"LOWER": "kotlin"}]
pattern_8= [{"LOWER": "react-native"}]
pattern_9= [{"LOWER": "ios"}]
pattern_16= [{"LOWER": "android"}]            
#for web dev

pattern_10= [{"LOWER": "frontend"}]
pattern_11= [{"LOWER": "backend"}]
pattern_12= [{"LOWER": "react"}]
pattern_13= [{"LOWER": "nodejs"}]
pattern_14= [{"LOWER": "javascript"}]
pattern_15= [{"LOWER": "HTML"}]

matcher.add("AI_TEST_PATTERNS_1",[pattern_1])
matcher.add("AI_TEST_PATTERNS_2",[pattern_2])
matcher.add("AI_TEST_PATTERNS_3",[pattern_3])
matcher.add("AI_TEST_PATTERNS_4",[pattern_4])
matcher.add("AI_TEST_PATTERNS_5",[pattern_5])
matcher.add("App_TEST_PATTERNS_6",[pattern_6])
matcher.add("App_TEST_PATTERNS_7",[pattern_7])
matcher.add("App_TEST_PATTERNS_8",[pattern_8])
matcher.add("App_TEST_PATTERNS_9",[pattern_9])
matcher.add("wb_TEST_PATTERNS_10",[pattern_10])
matcher.add("wb_TEST_PATTERNS_11",[pattern_11])
matcher.add("wb_TEST_PATTERNS_12",[pattern_12])
matcher.add("wb_TEST_PATTERNS_13",[pattern_13])
matcher.add("wb_TEST_PATTERNS_14",[pattern_14])
matcher.add("wb_TEST_PATTERNS_15",[pattern_15])
matcher.add("App_TEST_PATTERNS_16",[pattern_16])





class Resume_Extractor:
  doc_list=[]

  def __init__(self,resumelist):
    
    for resume in resumelist:
      raw = parser.from_file(resume)
      # print(raw['content'])
    
      self.content=raw['content']
      self.doc=nlp(self.content)

      Resume_Extractor.doc_list.append(self.doc)

  def __call__(self):
    matches=[]
    for n_doc in Resume_Extractor.doc_list:
      match=matcher(n_doc)
      matches.append(match)
    return matches,Resume_Extractor.doc_list

resumelist=['My Resume.pdf','sukesh.pdf']
resume_obj=Resume_Extractor(resumelist)
matches,doclist=resume_obj()
# print(Resume_Extractor.doc_list)
# for match_id,pos1,pos2 in matches: 
#   print(f'{nlp.vocab.strings[match_id]} : {doc[pos1:pos2].text}')








class Feature_Matrix():
  def __init__(self,n_resumes):
    feat={'ML':0,'DL':0,'NLP':0,'BD':0,'RF':0,'CV':0,'frontend':0,'backend':0,'react':0,'javascript':0,
          'HTML':0,'nodejs':0,'kotlin':0,'android':0,'ios':0,'react-native':0,'android studio':0}
    self.feat=feat
    self.n_resumes=n_resumes
    
    self.FM=np.zeros((self.n_resumes,len(self.feat)))
      
      
  def feature_gen(self,matches,doclist):

    for (i,match),doc in zip(enumerate(matches),doclist):
      print(f'for resume {i} \n')
      for match_id,pos1,pos2 in match: 
          print(f'{nlp.vocab.strings[match_id]} : {doc[pos1:pos2].text}')

          if doc[pos1:pos2].text.lower() == 'machine learning':
            self.feat['ML']+=1

          elif doc[pos1:pos2].text.lower() == 'nlp':
            self.feat['NLP']+=1

          elif doc[pos1:pos2].text.lower() == 'deep learning':
            self.feat['DL']+=1

          elif doc[pos1:pos2].text.lower() == 'big data':
            self.feat['BD']+=1

          elif doc[pos1:pos2].text.lower() == 'reinforcement learning':
            self.feat['RF']+=1

          elif doc[pos1:pos2].text.lower() == 'CV':
            self.feat['CV']+=1
          
          elif doc[pos1:pos2].text.lower() == 'html':
            self.feat['HTML']+=1

          elif doc[pos1:pos2].text.lower() == 'react':
            self.feat['react']+=1

          elif doc[pos1:pos2].text.lower() == 'javascript':
            self.feat['javascript']+=1

          elif doc[pos1:pos2].text.lower() == 'frontend':
            self.feat['frontend']+=1

          elif doc[pos1:pos2].text.lower() == 'backend':
            self.feat['backend']+=1
            
          elif doc[pos1:pos2].text.lower() == 'nodejs':
            self.feat['nodejs']+=1

          elif doc[pos1:pos2].text.lower() == 'kotlin':
            self.feat['kotlin']+=1   
          
          elif doc[pos1:pos2].text.lower() == 'android':
            self.feat['android']+=1

          elif doc[pos1:pos2].text.lower() == 'android studio':
            self.feat['android studio']+=1

          elif doc[pos1:pos2].text.lower() == 'ios':
            self.feat['ios']+=1

          elif doc[pos1:pos2].text.lower() == 'react-native':
            self.feat['react-native']+=1

          else:
            print(f'{doc[pos1:pos2]} has no matches \n')


      for j in range(len(self.feat)):
          if j==0:
            self.FM[i,j]=self.feat['ML']
          elif j==1:
            self.FM[i,j]=self.feat['DL']
          elif j==2:
            self.FM[i,j]=self.feat['NLP']
          elif j==3:
            self.FM[i,j]=self.feat['BD']
          elif j==4:
            self.FM[i,j]=self.feat['RF']
          elif j==5:
            self.FM[i,j]=self.feat['CV']
          elif j==6:
            self.FM[i,j]=self.feat['HTML']
          elif j==7:
            self.FM[i,j]=self.feat['javascript']
          elif j==8:
            self.FM[i,j]=self.feat['backend']
          elif j==9:
            self.FM[i,j]=self.feat['frontend']
          elif j==10:
            self.FM[i,j]=self.feat['react']
          elif j==11:
            self.FM[i,j]=self.feat['nodejs']
          elif j==12:
            self.FM[i,j]=self.feat['kotlin']
          elif j==13:
            self.FM[i,j]==self.feat['android']
          elif j==14:
            self.FM[i,j]=self.feat['ios']
          elif j==15:
            self.FM[i,j]=self.feat['android studio']
          elif j==16:
            self.FM[i,j]=self.feat['react-native']
          else:
            self.FM[i,j]=0

    return self.FM

      
  def class_label(self):
      #for AI
      target_data=[]
      ai_score=(self.feat['ML']+self.feat['DL']+self.feat['NLP']+self.feat['BD']+self.feat['RF']+self.feat['CV'])/(sum(self.feat.values()))

      #for web dev

      wb_score=(self.feat['frontend']+self.feat['backend']+self.feat['react']+self.feat['javascript']+self.feat['HTML']+self.feat['nodejs'])/(sum(self.feat.values()))

      #for app dev

      app_score=(self.feat['kotlin']+self.feat['android']+self.feat['ios']+self.feat['react-native']+self.feat['android studio'])/(sum(self.feat.values()))
      
      if ai_score>wb_score and ai_score>app_score:
        target_data.append('AI developer')
      elif wb_score>ai_score and wb_score>app_score:
        target_data.append('web developer')
      elif app_score>ai_score and app_score>wb_score:
        target_data.append('app developer')

      
      return target_data




if __name__=="__main__":

  arr_obj=Feature_Matrix(2)
  data=arr_obj.feature_gen(matches,doclist)


for i,arr in enumerate(data):
  for j,element in enumerate(arr):
    data[i,j]=element/sum(arr)

arr_obj.class_label()

