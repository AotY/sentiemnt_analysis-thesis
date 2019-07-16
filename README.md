# Sentiment Analysis Theis

This is the code for my thesis (Research and Application of Sentiment Analysis Based on Transformer). 

## Abstract

Sentiment analysis is one of the research hotspots in natural language processing. It is the pipeline of analyzing, processing, summarizing and reasoning subjective texts with emotional inclination. The word embedding model that has emerged in recent years has been widely used in natural language processing. However, the current word embedding model pays more attention to the context of words, ignoring some special cases in the language, such as harmonics phenomenon. On the other hand, the rise of the Transformer Network has further solved the problem of long dependence in the text, but how to apply it to the sentiment analysis task effectively has become a hotspot problem.

Firstly, this thesis studies the harmonics phenomenon in Chinese proposes to add Pinyin information to the semantic space of words and uses TF-IDF values to evaluate the importance of each word, implements the word embedding model of joint Pinyin and TF-IDF weighted. The addition of Pinyin allows word in a similar context of Pinyin to share some semantic information, solving the semantic similarity problem of harmonics. Using the TF-IDF value enforces model to pay more attention to important words while learning, then get a better semantic representation. Secondly, this paper studies the Transformer network and proposes two sentiment analysis models based on Transformer: Transformer_Adaptive and Transformer_Gumbel. The Transformer_Adaptive model uses an adaptive way to learn the weight distribution and then derives the sentence representation from the Transformer's outputs. The Transformer_Gumbel model uses a differentiable sampling method (GumbelSoftmax) to get the sentence representation from the Transformer's outputs. The performance of sentiment classification can be improved by the sentence representation of the Transformer_Adaptive and Transformer_Gumbel models.

The results show that the F1 values of the Transformer_Adaptive and Transformer_Gumbel models are increased by 0.46% and 1.51%, respectively, over the larger dataset. After initializing with the pre-trained word vector, the F1 value is increased by 0.89% and 2.11% compared to the baseline model initialized in the same manner. The model proposed by this thesis not only improves the performance of sentiment analysis tasks to a certain extent but has an applying value.

## Requirement

- Python 3.5+
- Pytorch 0.4+

## Data

### 1. For Training  Embedding

reviews-qa: 

> https://github.com/brightmart/nlp_chinese_corpus 百科类问答json版(baike2018qa) + 社区问答json版(webtext2019zh) 
>
> https://github.com/SophonPlus/ChineseNlpCorpus weibo_senti_100k + simplifyweibo_4_moods + yf_dianping + yf_amazon + dmsc_v2

wiki-news:

> https://github.com/brightmart/nlp_chinese_corpus 维基百科json版(wiki2019zh) +  新闻语料json版(news2016zh)

### 2. For Training Sentiment Model

ChnSentiCorp_htl_all(CSCH), OS10, online_shopping_10_cats(dmsc_v2), yf_amazon: https://github.com/SophonPlus/ChineseNlpCorpus


## Model
### 1.BiGRU

```
SAModel(
  (encoder): RNNEncoder(
    (embedding): Embedding(15004, 100, padding_idx=0)
    (dropout): Dropout(p=0.7)
    (rnn): GRU(100, 64, bidirectional=True)
    (reduce_state): ReduceState()
    (linear_final): Linear(in_features=128, out_features=3, bias=True)
  )
)
```

### 2.CNN

```
SAModel(
  (encoder): CNNEncoder(
    (embedding): Embedding(15004, 100, padding_idx=0)
    (conv1): Conv2d(1, 100, kernel_size=(3, 100), stride=(1, 1))
    (conv2): Conv2d(1, 100, kernel_size=(4, 100), stride=(1, 1))
    (conv3): Conv2d(1, 100, kernel_size=(2, 100), stride=(1, 1))
    (dropout): Dropout(p=0.7)
    (linear_final): Linear(in_features=300, out_features=3, bias=True)
  )
)
```



### 3.Transformer_Avg

```
SAModel(                                                                                                                               [25/329]
  (encoder): BERTCM(
    (bert): BERT(
      (embedding): Embedding(15004, 100, padding_idx=0)
      (pos_embedding): Embedding(91, 100, padding_idx=0)
      (dropout): Dropout(p=0.7)
      (transformer_blocks): ModuleList(
        (0): TransformerBlock(
          (attention): MultiHeadedAttention(
            (linear_layers): ModuleList(
              (0): Linear(in_features=100, out_features=100, bias=True)
              (1): Linear(in_features=100, out_features=100, bias=True)
              (2): Linear(in_features=100, out_features=100, bias=True)
            )
            (output_linear): Linear(in_features=100, out_features=100, bias=True)
            (attention): Attention()
            (dropout): Dropout(p=0.1)
          )
          (feed_forward): PositionwiseFeedForward(
            (w_1): Linear(in_features=100, out_features=400, bias=True)
            (w_2): Linear(in_features=400, out_features=100, bias=True)
            (dropout): Dropout(p=0.7)
            (activation): GELU()
          )
          (input_sublayer): SublayerConnection(
            (norm): LayerNorm()
            (dropout): Dropout(p=0.7)
          )
          (output_sublayer): SublayerConnection(
            (norm): LayerNorm()
            (dropout): Dropout(p=0.7)
          )
          (dropout): Dropout(p=0.7)
        )
      )
    )
    (norm): LayerNorm()
    (linear_final): Linear(in_features=100, out_features=3, bias=True)
  )
)
```



### 4.Transformer_Max

```
SAModel(                                                                                                                              
  (encoder): TransformerCM(
    (transformer): Transformer(
      (embedding): Embedding(15004, 100, padding_idx=0)
      (pos_embedding): Embedding(91, 100, padding_idx=0)
      (dropout): Dropout(p=0.7)
      (transformer_blocks): ModuleList(
        (0): TransformerBlock(
          (attention): MultiHeadedAttention(
            (linear_layers): ModuleList(
              (0): Linear(in_features=100, out_features=100, bias=True)
              (1): Linear(in_features=100, out_features=100, bias=True)
              (2): Linear(in_features=100, out_features=100, bias=True)
            )
            (output_linear): Linear(in_features=100, out_features=100, bias=True)
            (attention): Attention()
            (dropout): Dropout(p=0.1)
          )
          (feed_forward): PositionwiseFeedForward(
            (w_1): Linear(in_features=100, out_features=400, bias=True)
            (w_2): Linear(in_features=400, out_features=100, bias=True)
            (dropout): Dropout(p=0.7)
            (activation): GELU()
          )
          (input_sublayer): SublayerConnection(
            (norm): LayerNorm()
            (dropout): Dropout(p=0.7)
          )
          (output_sublayer): SublayerConnection(
            (norm): LayerNorm()
            (dropout): Dropout(p=0.7)
          )
          (dropout): Dropout(p=0.7)
        )
      )
    )
    (norm): LayerNorm()
    (linear_final): Linear(in_features=100, out_features=3, bias=True)
  )
)
```



### 5.Transformer_Adaptive

```
SAModel(                                                                                                                               
  (encoder): TransformerCM(
    (transformer): Transformer(
      (embedding): Embedding(15004, 100, padding_idx=0)
      (pos_embedding): Embedding(91, 100, padding_idx=0)
      (dropout): Dropout(p=0.7)
      (transformer_blocks): ModuleList(
        (0): TransformerBlock(
          (attention): MultiHeadedAttention(
            (linear_layers): ModuleList(
              (0): Linear(in_features=100, out_features=100, bias=True)
              (1): Linear(in_features=100, out_features=100, bias=True)
              (2): Linear(in_features=100, out_features=100, bias=True)
            )
            (output_linear): Linear(in_features=100, out_features=100, bias=True)
            (attention): Attention()
            (dropout): Dropout(p=0.1)
          )
          (feed_forward): PositionwiseFeedForward(
            (w_1): Linear(in_features=100, out_features=400, bias=True)
            (w_2): Linear(in_features=400, out_features=100, bias=True)
            (dropout): Dropout(p=0.7)
            (activation): GELU()
          )
          (input_sublayer): SublayerConnection(
            (norm): LayerNorm()
            (dropout): Dropout(p=0.7)
          )
          (output_sublayer): SublayerConnection(
            (norm): LayerNorm()
            (dropout): Dropout(p=0.7)
          )
          (dropout): Dropout(p=0.7)
        )
      )
    )
    (norm): LayerNorm()
    (linear_final): Linear(in_features=100, out_features=3, bias=True)
  )
)
```



### 6.Transformer_Gumbel

```
SAModel(                                                                                                                             
  (encoder): TransformerCM(
    (transformer): Transformer(
      (embedding): Embedding(15004, 100, padding_idx=0)
      (pos_embedding): Embedding(91, 100, padding_idx=0)
      (dropout): Dropout(p=0.7)
      (transformer_blocks): ModuleList(
        (0): TransformerBlock(
          (attention): MultiHeadedAttention(
            (linear_layers): ModuleList(
              (0): Linear(in_features=100, out_features=100, bias=True)
              (1): Linear(in_features=100, out_features=100, bias=True)
              (2): Linear(in_features=100, out_features=100, bias=True)
            )
            (output_linear): Linear(in_features=100, out_features=100, bias=True)
            (attention): Attention()
            (dropout): Dropout(p=0.1)
          )
          (feed_forward): PositionwiseFeedForward(
            (w_1): Linear(in_features=100, out_features=400, bias=True)
            (w_2): Linear(in_features=400, out_features=100, bias=True)
            (dropout): Dropout(p=0.7)
            (activation): GELU()
          )
          (input_sublayer): SublayerConnection(
            (norm): LayerNorm()
            (dropout): Dropout(p=0.7)
          )
          (output_sublayer): SublayerConnection(
            (norm): LayerNorm()
            (dropout): Dropout(p=0.7)
          )
          (dropout): Dropout(p=0.7)
        )
      )
    )
    (norm): LayerNorm()
    (linear_final): Linear(in_features=100, out_features=3, bias=True)
  )
)
```




