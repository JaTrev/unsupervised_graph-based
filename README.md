# Unsupervised Graph-based Topic Modeling from Video Transcriptions
This is the repository of the paper
"Unsupervised Graph-based Topic Modeling from Video Transcriptions"
by Jason Thies, Lukas Stappen, Gerhard Hagerer, Björn W. Schuller, and Georg Groh.

In this paper,  we aim at developing a topic extractor on video transcriptions. 
The model improves coherence by exploiting neural word embeddings through a graph-based clustering method. 
Unlike typical topic models, this approach works without knowing the true number of topics. 
Experimental results on the real-life multimodal dataset MuSe-CaR demonstrates that our approach extracts coherent and 
meaningful topics, outperforming baseline methods. 
Furthermore, we successfully demonstrate the generalisability of our approach on a pure text review dataset.



Overview of this repository:
- visuals:
      This folder contains all graphs and scores from the topic models.

- src:
      This folder contains all the python source code for the study,
      use the requirements file to download all necessary libraries.

- data:
    This folder includes the training data set (including the labels) of MuSe - CaR 
    as well as the CitySearch Car Review data set (training and test set) from 
    ([http://www.cs.cmu.edu/~mehrbod/RR/][http://www.cs.cmu.edu/~mehrbod/RR/]). 
    All existing pre-calculated models are in this folder.
  


Installation Instructions:


1. Clone Repository:
    git clone ...


2. Create virtual environment (this project runs on Python 3.6):
    conda create --name unsupervised_graph-based python=3.6


3. Activate virtual environment:
    conda activate unsupervised_graph-based


3. Fetch requirements:
    pip3 install -r requirements.txt


4. run main.py:
    python main.py --data_set XX --tm YY


Arguments:
- (--data_set) is used to select the preprocessed data set:
    MuSe-CaR: MUSE
    Citysearch Corpus: CRR

- (--tm) is used to set the topic model:
    Clustering-Based Baselines: TVS
    Graph-based Clustering (using K-Components): k-components