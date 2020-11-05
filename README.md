# Text-Summarization
## Survey Papers
* Text Summarization Techniques: A Brief Survey(2017)
    - https://arxiv.org/pdf/1707.02268.pdf
* A Survey on Methods of Abstractive Text Summarization(2014)
    - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.679.2132&rep=rep1&type=pdf
* Recent automatic text summarization techniques: a survey(2017)
* METHODOLOGIES AND TECHNIQUES FOR TEXT SUMMARIZATION: A SURVEY(2020)
    - http://www.jcreview.com/fulltext/197-1592984804.pdf
* A SURVEY OF RECENT TECHNIQUES IN AUTOMATIC TEXT SUMMARIZATION(2018)
    - https://www.iaeme.com/MasterAdmin/uploadfolder/IJCET_09_02_007/IJCET_09_02_007.pdf

## Extractive Summarization

### Graph-based Model
- TextRank: Bringing Order into Texts(2004) ✔️
    - https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf
    - 참고: https://lovit.github.io/nlp/2019/04/30/textrank/
### Neural Network
#### Reinforcement Learning
* [Ranking Sentences for Extractive Summarization with Reinforcement Learning(2018)](https://arxiv.org/pdf/1802.08636.pdf)

#### CNN
* [CLASSIFY OR SELECT: NEURAL ARCHITECTURES FOR EXTRACTIVE DOCUMENT SUMMARIZATION(2015)](https://arxiv.org/pdf/1611.04244.pdf)

#### Encoder-Decoder Model
* [SummaRuNNer: A Recurrent Neural Network based Sequence Model for Extractive Summarization of Documents(2017)](https://arxiv.org/abs/1611.04230) ✔️
* [Neural Latent Extractive Document Summarization(2018)](https://www.aclweb.org/anthology/D18-1088.pdf) 
* [Fine-tune BERT for Extractive Summarization(2019)](https://arxiv.org/abs/1903.10318) 
* [Extractive Summarization of Long Documents by Combining Global and Local Context(2019)](https://arxiv.org/abs/1909.08089)

#### Attention
* [Neural Summarization by Extracting Sentences and Words(2016)](https://arxiv.org/abs/1603.07252) ✔️
    - 좀 더 하이브리드에 가까운 것 같다(Extractive -> 그것을 가지고 Abstractive) 
* [AttSum: Joint Learning of Focusing and Summarization with Neural Attention(2016)](https://arxiv.org/abs/1604.00125)

## Abstractive Summarization
### Seq2Seq

### Attention
* [A Neural Attention Model for Abstractive Sentence Summarization(2015)](https://arxiv.org/abs/1509.00685) ✔️ 
* [Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond(2016)](https://arxiv.org/abs/1602.06023) ✔️
* [Abstractive Sentence Summarization with Attentive Recurrent Neural Networks(2017)](https://nlp.seas.harvard.edu/papers/naacl16_summary.pdf)
* [A Deep Reinforced Model for Abstractive Summarization(2017)](https://arxiv.org/abs/1705.04304) ✔️
* [Get To The Point: Summarization with Pointer-Generator Networks(2017)](https://arxiv.org/abs/1704.04368)  ✔️ 
* [Deep Communicating Agents for Abstractive Summarization(2018)](https://www.aclweb.org/anthology/N18-1150.pdf)
* [Bottom-Up Abstractive Summarization(2018)](https://arxiv.org/pdf/1808.10792.pdf)

#### Transformer
* [Text Summarization with Pretrained Encoders(2019)](https://arxiv.org/pdf/1908.08345.pdf)  ✔️ 
    - used BERT in Abstractive Summarization
* [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension(2019)](https://arxiv.org/pdf/1910.13461.pdf) ✔️
    * 201105
        * BERT의 인코더와 GPT의 디코더를 합친 형태의 모델
        * seq2seq denoising autoencoder 언어 모델이며, 1) noising function으로 text를 망가뜨리고, 2) 그걸 다시 원래 문장으로 만드는 과정을 학습하게 된다.
        * text generation뿐만 아니라 comprehension에도 효과가 있어 다양한 nlp 분야의 sota 달성
* [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization(2019)](https://arxiv.org/abs/1912.08777)  ✔️

## Evaluation
* An Evaluation for Various Text Summarization Algorithms on Blog Summarization Dataset(2018)
    - https://pdfs.semanticscholar.org/27a5/664e20cb3eb3e0503a9e5685075067e949a2.pdf
* Automatic Evaluation of Summaries Using N-gram Co-Occurrence Statistics

## Some other fields that could be helpful
### Grammatical Error Correction(GEC)
* [Improving grammatical error correction via pre-training a copy-augmented archi- tecture with unlabeled data(2019)](https://arxiv.org/abs/1903.00138)
* [A Neural Grammatical Error Correction System Built On Better Pre-training and Sequential Transfer Learning(2019)](https://arxiv.org/abs/1907.01256)
    - Kakao, 2019 ACL 2등
