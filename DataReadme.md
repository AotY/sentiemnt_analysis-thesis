## 数据描述

1. **ChnSentiCorp_htl_all**.csv CSCH

> label1 表示正向评论，0 表示负向评论
>
> review评论内容
>
> label:
>
> ​	0 ~ 2400  31%
>
> ​	1 ~ 5300 68%
>
> len: 120 conver 80%,
>
> ​	150 conver 85%, 
>
> ​	170 covner 88%
>
> voab:
>
> ​	7000 conver 95.8%
>
> ​	8000 conver 96.3%



| cnn                                                          | rnn                                                          | bert_avg                                                     | bert_max                                                     | bert_sample                                                  | bert_weight                                                  | self_attn |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | --------- |
| ![chn-cnn](https://ws4.sinaimg.cn/large/006tKfTcly1g1e8jiovhpj30w20bctd0.jpg) | ![chn-rnn](https://ws2.sinaimg.cn/large/006tKfTcly1g1e8oofgjmj30ww0bqaes.jpg) | ![chn-bert_avg](https://ws4.sinaimg.cn/large/006tKfTcly1g1e8jmfiapj30v00c0gpz.jpg) | ![chn-bert_max](https://ws2.sinaimg.cn/large/006tKfTcly1g1e8q04fqsj30ty0bygpv.jpg) | ![chn-bert_smaple](https://ws4.sinaimg.cn/large/006tKfTcly1g1e8oquj2pj30ww0cg0xe.jpg) | ![chn-bert_weight](https://ws2.sinaimg.cn/large/006tKfTcly1g1e96nexwlj30w60ca78s.jpg) |           |

 

2. **online_shopping_10_cats** OS10

   >label1 表示正向评论，0 表示负向评论
   >
   >review评论内容
   >
   >label:
   >
   >​	0 ~ 3w  49%
   >
   >​	1 ~ 3.1w 51%
   >
   >len: 
   >
   >​	75 conver 87%
   >
   >​	85 conver 89%
   >
   >​	90 conver 90%
   >
   >voab:
   >
   >​	10000 conver 96%
   >
   >​	12000 conver 97%

   

3. **yf_amazon**

> rating  评分，[1,5] 之间的整数      
>
> comment  评论内容
>
> label:
>
> ​	0 ~ 16w 5.6%
>
> ​	1 ~ 47w 16%
>
> ​	2 ~ 230w 78%
>
> len:
>
> ​	85 conver 94.5%
>
> ​	100 conver 96%
>
> 
>
> vocab:
>
> ​	1.5w 96.6%
>
> ​	2w conver 97.2%



| cnn  | rnn                                                          | bert_avg                                                     | bert_max                                                     | bert_sample                                                  | bert_weight                                                  | self_attn |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | --------- |
|      | ![yf_amazon_rnn](https://ws2.sinaimg.cn/large/006tKfTcly1g1pihqwrgrj30xw0d2aej.jpg) | ![yf_amazon_bert_avg](https://ws4.sinaimg.cn/large/006tKfTcly1g1o3yyzgpij311c0dadl5.jpg) | ![yf_amazon_bert_max](https://ws2.sinaimg.cn/large/006tKfTcly1g1pigckf1lj30ww0cyq7d.jpg) | ![yf_amazon_bert_sample](https://ws2.sinaimg.cn/large/006tNc79ly1g1qxlgalgdj30zw0d2gr2.jpg) | ![yf_amazon_bert_weight](https://ws1.sinaimg.cn/large/006tKfTcly1g1pihmbbzqj30wa0ce0wq.jpg) |           |

 



4. **dmsc_v2**

> rating  评分，[1,5] 之间的整数      
>
> comment  评论内容
>
> label:
>
> ​	0 ~ 19w 14%
>
> ​	1 ~ 47w 36%
>
> ​	2 ~ 64w 49%
>
> len:
>
> ​	75 conver 93%
>
> ​	80 conver 94.2%
>
> ​	85 conver 95.4%
>
> vocab:
>
> ​	1.2w 96.1%
>
> ​	1.5w 96.7%
>
> ​	2w conver 97.3%





| cnn                                                          | rnn                                                          | bert_avg                                                     | bert_max                                                     | bert_sample                                                  | bert_weight                                                  | self_attn |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | --------- |
| ![dmsc_v2-cnn](https://ws4.sinaimg.cn/large/006tKfTcly1g1ilib8kf9j30z40den3m.jpg) | ![dmsc_v2-rnn](https://ws1.sinaimg.cn/large/006tKfTcly1g1ilichhs7j30vu0d00yv.jpg) | ![dmsc_v2-bert_avg](https://ws4.sinaimg.cn/large/006tKfTcly1g1imzs3id2j30wu0cwdjv.jpg) | ![dmsc_v2-bert_max](https://ws2.sinaimg.cn/large/006tKfTcly1g1ilij7k36j30ym0cudln.jpg) | ![dmsc_v2-bert_sample](https://ws4.sinaimg.cn/large/006tKfTcly1g1ilinguthj30wq0d644s.jpg) | ![dmsc_v2-bert_weight](https://ws2.sinaimg.cn/large/006tKfTcly1g1ilipsxunj30yo0cydlt.jpg) |           |

 









































**weibo_senti_100k**

> label  1 表示正向评论，0 表示负向评论  
>
> review  微博内容
>
> label:
>
> ​	0 ~ 6000
>
> ​	1 ~ 6000
>
> len:
>
> ​	70 conver 83%
>
> ​	80 conver 90%
>
> vocab:
>
> ​	9000 conver 87%
>
> ​	15000 conver 90%
>
> ​	





**simplifyweibo_4_moods**

> label  0 喜悦，1 愤怒，2 厌恶，3 低落  
>
> review  微博内容
>
> label:
>
> ​	0 ~ 20w 55%
>
> ​	1 ~ 5w 14%
>
> ​	2 ~ 5w 15 %
>
> ​	3 ~ 5w 15 %
>
> len:
>
> ​	95 conver 88%
>
> ​	108 conver 93%
>
> voab:
>
> ​	2w conver 94%
>
> ​	3w conver 95.1%	



| cnn                                                          | rnn                                                          | bert_avg                                                     | bert_max                                                     | bert_sample                                                  | bert_weight                                                  | self_attn |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | --------- |
| ![simplify-cnn](https://ws4.sinaimg.cn/large/006tKfTcly1g1f0gme5dlj30wq0ea44g.jpg) | ![Screen Shot 2019-03-25 at 15.22.35](https://ws2.sinaimg.cn/large/006tKfTcly1g1f2ihfxisj30w60do0xz.jpg) | ![simplify-bert_avg](../../VMShare/Thesis/images/experiments/simplify-bert_avg.png) | ![simplify-bert_max](https://ws2.sinaimg.cn/large/006tKfTcly1g1f0gis2o9j30wo0ean2z.jpg) | ![simplify-bert_sample_](https://ws4.sinaimg.cn/large/006tKfTcly1g1f0jeqtszj30xu0eajyu.jpg) | ![simplify-bert_weight](../../VMShare/Thesis/images/experiments/simplify-bert_weight.png) |           |

 

















