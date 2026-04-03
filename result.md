对应关系是这样的：

- [real_run](/Users/junye/Documents/code/visualstudio/cephalopod%20camouflage/outputs/real_run)
  第一版启发式模型，还是早期“环境特征 -> 连续皮肤更新”的版本。  
  结果特征：`uniform 0.718 / mottle 0.120 / disruptive 0.163`，`Final loss 0.08086`

- [real_run_bvam](/Users/junye/Documents/code/visualstudio/cephalopod%20camouflage/outputs/real_run_bvam)
  第二版，第一次切到论文里的 `BVAM/Turing` 核心，但渲染层还比较生硬。  
  结果特征：`BVAM C=0.293, D_A=0.118`，`Final loss 0.143`

- [real_run_bvam_v2](/Users/junye/Documents/code/visualstudio/cephalopod%20camouflage/outputs/real_run_bvam_v2)
  第三版，还是 `BVAM/Turing` 核心，但把皮肤着色改成更跟环境低频颜色一致。  
  结果特征：`BVAM C=0.293, D_A=0.118`，`Final loss 0.143`

- [real_run_hybrid](/Users/junye/Documents/code/visualstudio/cephalopod%20camouflage/outputs/real_run_hybrid)
  最新版，也是现在推荐你看的版本。  
  它用了混合模型：`inverse-Turing 参数拟合 + BVAM 纹理 + chromatophore/iridophore/leucophore 三层渲染`。  
  结果特征：`BVAM C=-0.009, D_A=0.192`，`Final loss 0.05741`

如果你只看一个，直接看 [real_run_hybrid](/Users/junye/Documents/code/visualstudio/cephalopod%20camouflage/outputs/real_run_hybrid)。

里面这些文件含义是：

- [octopus_skin.png](/Users/junye/Documents/code/visualstudio/cephalopod%20camouflage/outputs/real_run_hybrid/octopus_skin.png)
  最终章鱼皮肤图案
- [octopus_on_environment.png](/Users/junye/Documents/code/visualstudio/cephalopod%20camouflage/outputs/real_run_hybrid/octopus_on_environment.png)
  叠加到环境图上的效果
- [chromatophore_layer.png](/Users/junye/Documents/code/visualstudio/cephalopod%20camouflage/outputs/real_run_hybrid/chromatophore_layer.png)
  色素胞层
- [iridophore_layer.png](/Users/junye/Documents/code/visualstudio/cephalopod%20camouflage/outputs/real_run_hybrid/iridophore_layer.png)
  结构色层
- [leucophore_layer.png](/Users/junye/Documents/code/visualstudio/cephalopod%20camouflage/outputs/real_run_hybrid/leucophore_layer.png)
  白色反射层
- [diagnostics.png](/Users/junye/Documents/code/visualstudio/cephalopod%20camouflage/outputs/real_run_hybrid/diagnostics.png)
  参数和分层诊断图

如果你要，我可以下一步顺手帮你把旧的 `real_run*` 目录整理成带版本说明的名字。


发现看着不太像：
对，这张参考图把问题说得很清楚了：当前输出最大的问题已经不是颜色方程，而是“身体模型完全不对”。你给的图里更像乌贼/墨鱼类体型，主图案集中在外套膜，边缘有半透明鳍，腕部和头部的纹理分布也和外套膜不同；我现在的结果还是一个章鱼卡通轮廓，所以再怎么调色也不像真实头足类。

我先改形体层，不碰 BVAM 核心。这样能直接回答你这张参考图指出的问题：让输出至少先像“真实头足类”，而不是一个带纹理的章鱼图标。

