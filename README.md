# BioCIE
Biomedical Confident Itemsets Explanation

In this repository, you can find the source code of the BioCIE explanation method. The supplementary material for the paper "Explaining Black-box Models for Biomedical Text Classification" is also presented in this repository.

<h2>Training black-box classifiers</h2>
<p>We trained three black-box classification models on three biomedical text classification tasks.</p>
<p>Link to the datasets:
<br>
  
- BioText: https://biotext.berkeley.edu/
- AIMed: https://ftp.cs.utexas.edu/pub/mooney/bio-data/
- Hereditary Diseases (HD): https://www2.informatik.hu-berlin.de/~hakenber/corpora/</p>
<p>We split each dataset into separate training and test sets with a ratio of 90:10. The black-box classifiers were trained on the training sets, and the explanations were produced for the outcomes of the black-boxes on the test sets.</p>

<h3>BioBERT classifier</h3>
<p>We used the FARM package (<a href="https://github.com/deepset-ai/FARM">https://github.com/deepset-ai/FARM</a>) in python to implement the BioBERT based classification model. As BioBERT is a variation of BERT that is trained on large biomedical text corpora, we used the BERT transformer model with the BioBERT-base language model version 1.0 pretrained on the PubMed and PMC corpora, along with the default hyperparameter settings, unless it is stated. Different pretrained BioBERT language models can be downloaded from <a href="https://github.com/dmis-lab/biobert">https://github.com/dmis-lab/biobert</a>.
<br>
As the BioBERT classification model had been already pretrained on the PubMed and PMC corpora, we fine-tuned it on the training set. Maximum sequence length was set to 128, batch size was set to 32, and learning rate was set to 3e-5.</p>
<h3>LSTM classifier</h3>
<p>We used the Keras library of Tensorflow in python to implement a LSTM based text classification model. The model comprised of an embedding layer with 100 dimensions, a biderctional LSTM layer with 100 units, and a dense layer as the output layer with softmax activation. We used binary cross entropy as the loss function, and the Adam optimizer with the default settings.</p>
<h3>SVM classifier</h3>
<p>Using the Scikit-learn library (<a href="https://scikit-learn.org/stable/index.html">https://scikit-learn.org/stable/index.html</a>) in python, we implemented a SVM classifier with RBF kernel, and tf-idf weights as the input features. Stop-words were removed to reduce the size of vocabulary. Ngram range was set to (1, 2) in order to consider both unigrams and bigrams.</p>
