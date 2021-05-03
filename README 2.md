# Unsupervised Topic and Aspect Modelling on Massive Video Transcriptions from YouTube
(Unüberwachte Themen- und Aspektmodellierung von umfangreichen Videotranskriptionen)

Author:           Jason Thies
Supervisor:       Prof. Dr. Georg Groh
Advisors:         Lukas Stappen, M.Sc.
                  Hagerer Gerhard, M.Sc.
Submission Date:  March 15, 2021

This paper addressed the uncertainty of an excellent topic modelling approach for real-world textual data. The inherent colloquialism, improper grammar, and speech recognition errors make a transcription data set uniquely challenging for topic mod- els. We analyze the performance of various topic models on their ability to extract generally coherent topics that represent the underlying corpus well. We have com- pared commonly used topic models and other topic modeling approaches to extract coherent topics. Different coherence measures, as well as a word intrusion study, are used to evaluate the performances. Additionally, the classification performance of these models was assessed. The study shows that traditional topic models can extract representative topics. However, using word embedding models drastically im- proves the general coherence of the topics. If the number of topics is not predefined, graph connectivity can extract highly coherent topics from the corpus. Additionally, a semantic space can be used to extract meaningful topics and classify segments simultaneously. Overall, topics were more coherent when the model was applied to a preprocessed data set that includes only nouns, and classification was better when the predicted topics were more coherent. These findings suggest topic modeling needs only basic preprocessing to perform well. While traditional models can extract topics of extremely unstructured data, training word embeddings improves results tremendously. The embedding space can be modeled by an embedding graph to create a valid alternative to clustering approaches.



Overview of this repository:
- Word Instruction Task (Study):
      This sub folder contains the results of the human evaluation study,
      it includes a .xlsx file and two .csv files of the results.

- visuals:
      This subfolder contains all graphs and scores from the topic models,
      as well as all the visuals in the thesis.

- Thesis:
      The actual thesis and a handwritten declaration is located in this subfolder.
      The handwritten declaration is part of the COVID submission procedure
      (see: https://www.in.tum.de/en/current-students/coronavirus/).

- src:
      This folder contains all the python source code to repeat the studies,
      use the requirements file to download all necessary libraries.

- data:
      All pre-calculated topic models and training set of MuSe - CaR
      is saved in this subfolder.

- bertEmbeddings.txt
      This file provides a link to a Google Drive from where the
      BERT embeddings can be downloaded.
      Put the .pickle files into the "data" subfolder.

- computeBertEmbeddings.ipynb:
      A Google Colab notebook that can be used to calculate:
      'train_vocab_emb_dict_11.pickle' and 'train_vocab_emb_dict_12.pickle',
      which are already in the Google Drive.




Installation Instructions:


1. Clone Repository


2. Create virtual environment (this project runs on Python 3.6):
    conda create --name thesisEnv python=3.6


3. Activate virtual environment:
    conda activate thesisEnv


3. Fetch requirements:
    pip3 install -r requirements.txt


4. run main.py:
    python main.py --pds XX --tm YY


Arguments:
- (--pds) is used to select the preprocessed data set:
    Just Nouns: JN
    Fully Preprocessed: FP

- (--tm) is used to set the topic model:
    NMF TF: Baseline
    Re-Ranking Words: RRW
    Topic Vector Similarity: TVS
    Graph Connectivity: k-components
    BERT: BERT
    Semantic Space using Pooled Word2Vec: avg_w2v
    Semantic Space using Doc2Vec: doc2vec

- (--mp) is used to calculated miscellaneous prints (already in the visuals folder)
    either 'segment_size' or 'common_words'

- (--bert) defines the BERT embedding type, (--tm) must be set to BERT
    for this to work, options are:
    'normal_11', 'normal_12', 'preprocessed_sentence_11', preprocessed_sentence_12'
    --> make sure to download the BERT embedding before hand!
