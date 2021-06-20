# Blog Tagger 

This Package can be used to extract keywords from a page to create tags for any blogs,news or any textual information available on the web page.
These tags highlights the topic content by providing a glance of large volume of texts embedded in a page.Tag generation is an important feature
in many sectors of IT such as Amazon uses tags for customer segmentation.

## Prerequisites:

Installed transformer model.<br>

Follow the instruction given below to install albert-base model ,you can change the model but it might need some customization in source code. 
so albert model is adviced here to download.

`model=TFAutoModel.from_pretrained('albert-base-v2')` <br>
`tokenizer=AutoTokenizer.from_pretrained('albert-base-v2')` <br>



## Usage Instructions

1. Clone the repository on local system 

2. Collect web data <br>

- For example <br>

  `from web_data import Blog_Data` <br>
  `data=Blog_Data("https://influencermarketinghub.com/12-best-food-blogs/")` pass website <br>
  `Text_data=data.text_prep(req=['h1', 'h2', 'h3', 'h4', 'p'])` pass tags <br>

3. Use main class Blog tagger to generate top k tags. <br>

- For example <br>

  `tagger=Blog_Tagger(Text_data,maxlen=<int num>)` <br>
  `tagger.token_embedding_gen(model,tokenizer)` <br>
  ` top_tokens=tagger.tag_gen(k)` <br>
