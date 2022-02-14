# Extending-Neural-Keyword-Extraction-with-TF-IDF-tagset-matching



# Citation

If you use our code please cite our work. 

@inproceedings{koloski-etal-2021-extending,
    title = "Extending Neural Keyword Extraction with {TF}-{IDF} tagset matching",
    author = "Koloski, Boshko  and
      Pollak, Senja  and
      {\v{S}}krlj, Bla{\v{z}}  and
      Martinc, Matej",
    booktitle = "Proceedings of the EACL Hackashop on News Media Content Analysis and Automated Report Generation",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.hackashop-1.4",
    pages = "22--29",
    abstract = "Keyword extraction is the task of identifying words (or multi-word expressions) that best describe a given document and serve in news portals to link articles of similar topics. In this work, we develop and evaluate our methods on four novel data sets covering less-represented, morphologically-rich languages in European news media industry (Croatian, Estonian, Latvian, and Russian). First, we perform evaluation of two supervised neural transformer-based methods, Transformer-based Neural Tagger for Keyword Identification (TNT-KID) and Bidirectional Encoder Representations from Transformers (BERT) with an additional Bidirectional Long Short-Term Memory Conditional Random Fields (BiLSTM CRF) classification head, and compare them to a baseline Term Frequency - Inverse Document Frequency (TF-IDF) based unsupervised approach. Next, we show that by combining the keywords retrieved by both neural transformer-based methods and extending the final set of keywords with an unsupervised TF-IDF based technique, we can drastically improve the recall of the system, making it appropriate for usage as a recommendation system in the media house environment.",
}
