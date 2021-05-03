# Unsupervised Graph-based Topic Modeling from Video Transcriptions
(repo in inprogress)
This is the repository of the paper
"Unsupervised Graph-based Topic Modeling from Video Transcriptions"
by Jason Thies, Lukas Stappen, Gerhard Hagerer, Björn W. Schuller, and Georg Groh.


Abstract:
To unfold the tremendous amount of audiovisual data uploaded daily to social media platforms, effective topic modeling techniques are needed. Existing work tends to apply variants of topic models on text data sets. In this paper,  we aim at developing a topic extractor on video transcriptions. The model improves coherence by exploiting neural word embeddings through a graph-based clustering method. Unlike typical topic models, this approach works without knowing the true number of topics. Experimental results on the real-life multimodal dataset MuSe-CaR demonstrates that our approach extracts coherent and meaningful topics, outperforming baseline methods. Furthermore, we successfully demonstrate the generalisability of our approach on a pure text review dataset.





Overview of this repository:
- visuals:
      This subfolder contains all graphs and scores from the topic models.

- src:
      This folder contains all the python source code to repeat the studies,
      use the requirements file to download all necessary libraries.

- data:
      All pre-calculated topic models and training set of MuSe - CaR
      is saved in this subfolder.




Installation Instructions:


1. Clone Repository


2. Create virtual environment (this project runs on Python 3.6):
    conda create --name unsupervised_graph-based python=3.6


3. Activate virtual environment:
    conda activate unsupervised_graph-based


3. Fetch requirements:
    pip3 install -r requirements.txt


4. run main.py:
    python main.py --pds XX --tm YY


Arguments:
- (--pds) is used to select the preprocessed data set:
    Just Nouns: JN

- (--tm) is used to set the topic model:
    Clustering-Based Baselines: TVS
    K-Components: k-components
