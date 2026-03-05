# DeepOFormer-G-T
it's suitable for life prediction under fatigue creep interaction
DeepOFormer概念的提出是来源于这篇文章：https://arxiv.org/abs/2503.22475/
这篇文章针对的是不同材料，然后给出了模型的解释以及实验分析与结果，我在阅读了之后，结合我个人之前做的流-热-固耦合仿真分析，适当的改动了一下这个模型。我把这个模型应用到了同一材料不同工况作用下的寿命预测模型进而给出了一个基于DeepOFormer的寿命预测模型，然后主干网络上考虑的不同考核点的坐标，也就是"G",geometry,引入tensor和transformer处理工况特征也就是"T"。/
我暂时是没有找到这篇论文的源程序，因此我考虑到这个架构本身也是基于DeepONet的一个变体架构，所以我自己就简单研究了一下DeepONet架构，然后写了自己的这个DeepOFormer_G/T模型，最后也是用于我自己去研究这个疲劳和蠕变交互作用下的一个寿命预测模型，不作学术论文课题用，如果有朋友需要借鉴参考请标注一下本次程序的出处和作者，谢谢！
