# Text-Summarization
* Bold: Read
## 1) Survey Papers
|Paper|Summary|
|------------|--------------------|
|[Text Summarization Techniques: A Brief Survey(2017)](https://arxiv.org/pdf/1707.02268.pdf)| |
|[A Survey on Methods of Abstractive Text Summarization(2014)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.679.2132&rep=rep1&type=pdf)| |
|Recent automatic text summarization techniques: a survey(2017)| |
|[METHODOLOGIES AND TECHNIQUES FOR TEXT SUMMARIZATION: A SURVEY(2020)](http://www.jcreview.com/fulltext/197-1592984804.pdf)| |
|[A SURVEY OF RECENT TECHNIQUES IN AUTOMATIC TEXT SUMMARIZATION(2018)](https://www.iaeme.com/MasterAdmin/uploadfolder/IJCET_09_02_007/IJCET_09_02_007.pdf)| |

## 2) Single Document Summarization
### (1) Extractive Summarization
#### Graph-based Model
|Paper|Summary|Reference|
|------------|--------------------|---|
|[**TextRank: Bringing Order into Texts(2004)**](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)| | https://lovit.github.io/nlp/2019/04/30/textrank/|
|[**Sentence Centrality Revisited for Unsupervised Summarization(2019)**](https://arxiv.org/pdf/1906.03508.pdf)|TextRank + BERT + Directed Graph|

#### Autoencoder
|Paper|Summary|Reference|
|------------|--------------------|---|
|[Recursive Autoencoders for ITG-based Translation(2013)](https://aclanthology.org/D13-1054.pdf)| |
|[Extractive Summarization using Continuous Vector Space Models(2014)](https://aclanthology.org/W14-1504/)| |

#### Neural Network
|Paper|Summary|
|------------|--------------------|
|[CLASSIFY OR SELECT: NEURAL ARCHITECTURES FOR EXTRACTIVE DOCUMENT SUMMARIZATION(2015)](https://arxiv.org/pdf/1611.04244.pdf)| |
|[**Neural Summarization by Extracting Sentences and Words(2016)**](https://arxiv.org/abs/1603.07252)|* 좀 더 하이브리드에 가까운 것 같다<br/>Extractive -> 그것을 가지고 Abstractive) |
|[AttSum: Joint Learning of Focusing and Summarization with Neural Attention(2016)](https://arxiv.org/abs/1604.00125)|
|[**SummaRuNNer: A Recurrent Neural Network based Sequence Model for Extractive Summarization of Documents(2017)**](https://arxiv.org/abs/1611.04230)| |
|[Neural Latent Extractive Document Summarization(2018)](https://www.aclweb.org/anthology/D18-1088.pdf)| |
|[Fine-tune BERT for Extractive Summarization(2019)](https://arxiv.org/abs/1903.10318)| |
|[Extractive Summarization of Long Documents by Combining Global and Local Context(2019)](https://arxiv.org/abs/1909.08089)| |
|[Unsupervised Extractive Summarization by Pre-training Hierarchical Transformers(2020)](https://arxiv.org/abs/2010.08242)| |


### (2) Abstractive Summarization
#### Attention
|Paper|Summary|
|------------|--------------------|
|[**A Neural Attention Model for Abstractive Sentence Summarization(2015)**](https://arxiv.org/abs/1509.00685)| |
|[**Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond(2016)**](https://arxiv.org/abs/1602.06023)| |
|[**Abstractive Sentence Summarization with Attentive Recurrent Neural Networks(2017)**](https://nlp.seas.harvard.edu/papers/naacl16_summary.pdf)| |
|[**Get To The Point: Summarization with Pointer-Generator Networks(2017)**](https://arxiv.org/abs/1704.04368)| |
|[Deep Communicating Agents for Abstractive Summarization(2018)](https://www.aclweb.org/anthology/N18-1150.pdf)| |
|[Bottom-Up Abstractive Summarization(2018)](https://arxiv.org/pdf/1808.10792.pdf)| |
|[**Text Summarization with Pretrained Encoders(2019)**](https://arxiv.org/pdf/1908.08345.pdf)|used BERT in Abstractive Summarization|

## 3) Multi-Document Summarization
|Paper|Summary|
|------------|--------------------|
|[**GENERATING WIKIPEDIA BY SUMMARIZING LONG SEQUENCES(2018)**](https://arxiv.org/abs/1801.10198)|* Extractive(중요한 정보 뽑기) + Abstractive(wiki article 생성)<br/>* T-ED라는 트랜스포머에서 디코더만 취한 모델 구조 제안 -> 긴 시퀀스에 잘 작동|
        
## 4) Long Document Summarization
|Paper|Summary|
|------------|--------------------|
|[A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents(2018)](https://www.aclweb.org/anthology/N18-2097.pdf)| |
|[Deep Communicating Agents for Abstractive Summarization(2018)](https://www.aclweb.org/anthology/N18-1150/)| |
|[**Extractive Summarization of Long Documents by Combining Global and Local Context(2019)**](https://www.aclweb.org/anthology/D19-1298.pdf)| |

## 5) Language Models
**task-specific한 언어 모델을 학습하기 보다는 general하게 사용될 수 있는(downstream task) 언어 모델을 학습 하는 것이 트렌드**
|Paper|Summary|
|------------|--------------------|
|[**BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension(2019)**](https://arxiv.org/pdf/1910.13461.pdf)|* BERT의 인코더와 GPT의 디코더를 합친 형태의 모델<br/>* seq2seq denoising autoencoder 언어 모델이며,<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1) noising function으로 text를 망가뜨리고<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2) 그걸 다시 원래 문장으로 만드는 과정을 학습하게 된다.<br/>* text generation뿐만 아니라 comprehension에도 효과가 있어 다양한 nlp 분야의 sota 달성|
|[**PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization(2019)**](https://arxiv.org/abs/1912.08777)| |
|[**Big Bird: Transformers for Longer Sequences(2020)**](https://papers.nips.cc/paper/2020/file/c8512d142a2d849725f31a9a7a361ab9-Paper.pdf)<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- [PPT](https://github.com/hyunbool/Text-Summarization/blob/master/%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%85%E1%85%B5/bigbird.pdf)| |

## 6) Reinforcement Learning
|Paper|Summary|
|-----|--------|
|[**A Deep Reinforced Model for Abstractive Summarization(2017)**](https://arxiv.org/abs/1705.04304)| |
|[Improving Abstraction in Text Summarization(2018)](https://www.aclweb.org/anthology/D18-1207.pdf)|- ML+RL ROUGE+Novel, with LM<br/>-꼭 읽어보기|
|[Deep Communicating Agents for Abstractive Summarization(2018](https://www.aclweb.org/anthology/N18-1150.pdf)|- DCA<br/>-읽어보기|
|[Ranking Sentences for Extractive Summarization with Reinforcement Learning(2018)](https://www.aclweb.org/anthology/N18-1158.pdf)| |
|[Ranking Sentences for Extractive Summarization with Reinforcement Learning(2018)](https://arxiv.org/pdf/1802.08636.pdf)| |
|[Reward Learning for Efficient Reinforcement Learning in Extractive Document Summarisation(2019)](https://www.ijcai.org/Proceedings/2019/0326.pdf)| |
|[Better rewards yield better summaries: Learning to summarise without references(2019)](https://arxiv.org/pdf/1909.01214.pdf)| |
|[Fine-tuning language models from human preferences(2019)](https://arxiv.org/pdf/1909.08593.pdf)| |
|[**Learning to summarize from human feedback(2020)**](https://arxiv.org/pdf/2009.01325.pdf)| |
|[**The Summary Loop: Learning to Write Abstractive Summaries Without Examples(2020)**](https://www.aclweb.org/anthology/2020.acl-main.460.pdf)|1. key term 마스킹: M<br/>- 마스킹은 tf-idf 이용해 k개 단어<br/>2. 원문에 대해 summarizer 이용해 요약: S<br/>3. M과 S 이용해 coverage로 마스킹 된 문서에 key term 채우기: F<br/>4. 원문과 F 비교해 coverage score 비교<br/>5. 요약문에 대한 fluency score 계산<br/>- 언어 모델의 probability로 fluency 계산<br/>6. 점수들 가지고 summarizer optimization - 여기에서 RL 사용|

## 7) Autoencoder
|Paper|Summary|
|-----|--------|
|[SummAE: Zero-Shot Abstractive Text Summarization using Length-Agnostic Auto-Encoders(2019)](https://arxiv.org/abs/1910.00998)| |
|[Sample Efficient Text Summarization Using a Single Pre-Trained Transformer(2019)](https://arxiv.org/abs/1905.08836)| |
|[MeanSum: A Neural Model for Unsupervised Multi-document Abstractive Summarization(2019)](https://arxiv.org/abs/1810.05739)| |

## 8) Metrics
* CNN/Daily Mail: [Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond(2016)](https://arxiv.org/abs/1602.06023)
    * with human ratings: [The price of debiasing automatic metrics in natural language evaluation(2018)](https://www.aclweb.org/anthology/P18-1060.pdf)
    
## 9) Evaluation
|Paper|Summary|
|-----|--------|
|[An Evaluation for Various Text Summarization Algorithms on Blog Summarization Dataset(2018)]( https://pdfs.semanticscholar.org/27a5/664e20cb3eb3e0503a9e5685075067e949a2.pdf)| |
|Automatic Evaluation of Summaries Using N-gram Co-Occurrence Statistics| |

## Some other fields that could be helpful
### Grammatical Error Correction(GEC)
|Paper|Summary|
|-----|--------|
|[Improving grammatical error correction via pre-training a copy-augmented archi- tecture with unlabeled data(2019)](https://arxiv.org/abs/1903.00138)| |
|[A Neural Grammatical Error Correction System Built On Better Pre-training and Sequential Transfer Learning(2019)](https://arxiv.org/abs/1907.01256)|Kakao, 2019 ACL 2등|
### Content Selection
|Paper|Summary|
|-----|--------|
|[**Exploring Content Selection in Summarization of Novel Chapters(2020)**](https://arxiv.org/abs/2005.01840)| |

### Factual Correctness in Summarization
|Paper|Summary|
|-----|-------|
|[Optimizing the Factual Correctness of a Summary: A Study of Summarizing Radiology Reports(2020)](https://arxiv.org/pdf/1911.02541.pdf)| |
