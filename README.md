# tianchi_zhhyjs
天池数字中国创新大赛 智慧海洋建设
相关数据可在官网下载
成绩为20/3275（结束后发现全部数据训练线上还可以提升一个百，大概能到第六左右，就很难受....，不过也说明这题抖动较大）



 
方案和模型思路以及使用的什么模型

    本方案主要分四个部分
        （1）数据预处理
            先对训练集和测试集数据的异常离散点进行处理，部分进行删除，然后对数据进行平滑处理，并对乱报点间隔时间数据进行处理使得间隔时间接近600秒，
      通过以上方法使得数据集基本合理。
         (2) 特征工程
            渔船位置统计特征
            点迹特征
            地理坐标组合特征
            比率特征
            速度分箱特征
            交叉特征
            时间分区特征
            位移、差分特征等
        （3）特征选择
            在充分调研拖网、围网、刺网的特点和区别后，有针对性进行特征挖掘，使得线上线下效果变化不大，保证鲁棒性。
        （4）模型选择
            考虑到海上执法的便捷、快速反应的现实需求，本方案选用了lightgbm5折交叉算法模型。
            整个运行时间在800秒左右，而大赛限定时间为14400秒；
            并对冗余的特征进行删减，同时保证较优成绩，精简代码，避免模型堆叠。
            目前只使用单模便能达到理想效果。模型效果和运算速度达到了很好的统一。代码非常简洁、速度非常快、预测效果达到了预期。



总结：
    
    模型帮助我们逼近数据的上限，特征工程做的好往往能有较大收益；
    训练数据越多往往效果越好；
