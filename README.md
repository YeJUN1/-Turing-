# Cephalopod Camouflage Prototype

这个原型现在采用混合模型：

1. `inverse-Turing` 风格的环境拟合
   从环境图提取亮度、边缘、纹理、频谱和方向性特征，估计 BVAM 参数 `C`、`D_A`、`n`。

2. `BVAM / Turing` 纹理生成
   用论文 *A Turing-based bimodal population code can specify Cephalopod chromatic skin displays* 中的两变量 reaction-diffusion 方程生成皮肤图案。

3. 多层皮肤渲染
   把生成的激活场投射到离散 `chromatophore` 阵列，再叠加 `iridophore` 和 `leucophore` 两层，得到更接近头足类皮肤机制的颜色输出。

4. 可选的参考图外形先验
   通过 `rembg` 从章鱼参考图中提取 silhouette，再把这个 mask 转成 `mantle / head_arms / axis` 等身体图层，替代固定程序模板。

5. 内置章鱼 `body prior` 模板库
   提供多种伏地章鱼姿态模板，并支持按环境特征自动选择模板。

## 当前算法

环境图进入后，脚本会：

- 提取视觉特征
- 推断 `uniform / mottle / disruptive`
- 用 inverse-Turing 风格映射估计 `BVAM` 参数
- 解 BVAM 方程生成纹理场
- 将纹理场映射到离散色素胞
- 渲染三层皮肤：
  `chromatophore`：黄 / 棕 / 黑色素层
  `iridophore`：蓝绿 / 金色结构反射层
  `leucophore`：高反射亮度底层

## 生物学与工程边界

这个项目是生物启发模型，不是真实章鱼脑和皮肤的逐细胞仿真。

已经直接借鉴的内容：

- 视觉输入经中央控制后驱动皮肤图案，而不是皮肤自己“看见”环境
- `uniform / mottle / disruptive` 可作为 body pattern 的高层描述
- 局部相互作用可以通过 Turing / Hopf 型 reaction-diffusion 生成 spots、stripes、mottle、blotch
- 头足类皮肤颜色不仅来自色素胞，还涉及 `iridophore` 和 `leucophore`

工程近似部分：

- 环境图到 BVAM 参数的映射是工程拟合，不是论文给出的完整视觉神经模型
- `reflectin / iridophore` 这里只做了近似的结构色渲染，不是严格光学仿真
- 如果不给 `--body-ref`，轮廓会从内置章鱼模板库中选择
- 即使给了 `--body-ref`，当前也只是 2D silhouette 先验，不包含 papillae、三维照明和真实肌肉驱动的姿态变形

## 运行

先激活环境：

```bash
conda activate cephalocam
```

运行静态伪装：

```bash
python octopus_camouflage.py \
  --env input/reef.png \
  --output-dir outputs/real_run_hybrid \
  --size 512 \
  --iterations 80
```

如果要直接从一张章鱼参考图提取外形先验：

```bash
python octopus_camouflage.py \
  --env input/reef.png \
  --body-ref "https://img95.699pic.com/photo/60020/1925.jpg_wh300.jpg!/fh/300/quality/90" \
  --output-dir outputs/real_run_body_ref_v1
```

`--body-ref` 同时支持本地路径和 URL。这个模式下，脚本会先做前景分割，再用分割结果驱动身体轮廓，而不是继续使用程序生成的伏地章鱼模板。

如果要使用内置模板库，并让程序自动按环境选择：

```bash
python octopus_camouflage.py \
  --env input/reef.png \
  --body-template auto \
  --output-dir outputs/real_run_template_auto_v2
```

如果要强制指定一个模板：

```bash
python octopus_camouflage.py --env input/reef.png --body-template reef_crouch
```

当前内置模板：

- `prone_spread`
- `photo_sprawl`
- `prone_tucked`
- `reef_crouch`
- `algae_reach`
- `crevice_anchor`
- `real_zhangyu_pose`

选择优先级：

- 给了 `--body-ref`：优先使用参考图外形先验
- 没给 `--body-ref`：使用 `--body-template`
- `--body-template auto`：根据环境特征自动选模板

其中 `real_zhangyu_pose` 是从真实参考图分割得到的 silhouette 模板，不是程序生成轮廓。

如果要切到论文里的动态 Hopf 一侧参数：

```bash
python octopus_camouflage.py --env input/reef.png --dynamic
```

如果要减弱环境颜色辅助，只保留更偏纹理驱动的结果：

```bash
python octopus_camouflage.py --env input/reef.png --color-assist 0
```

## 输出

脚本会生成：

- `octopus_skin.png`
- `chromatophore_layer.png`
- `iridophore_layer.png`
- `leucophore_layer.png`
- `octopus_on_environment.png`
- `diagnostics.png`

如果使用了 `--body-ref`，还会额外生成：

- `body_ref_mask_raw.png`
- `body_ref_mask_clean.png`
- `body_ref_cutout.png`
- `body_ref_texture_prior.png`

实际输出目录会自动追加本地时间戳，格式是 `YYYYMMDD_HHMM`，例如：
`outputs/real_run_ref2_20260403_0038`

## 架构说明

更完整的数据流和算法层次图见：
[docs/camouflage_pipeline.md](/Users/junye/Documents/code/visualstudio/cephalopod%20camouflage/docs/camouflage_pipeline.md)

如果你要把当前项目上升成一篇“神经机制 + Turing 模式 + 神经网络控制器”的生物启发计算小论文，成稿草案见：
[docs/paper_design.md](/Users/junye/Documents/code/visualstudio/cephalopod%20camouflage/docs/paper_design.md)

## 导出 PDF

项目里已经带了论文 Markdown 到 PDF 的导出脚本，默认导出：
[paper_design.md](/Users/junye/Documents/code/visualstudio/cephalopod%20camouflage/docs/paper_design.md)
到
[paper_design.pdf](/Users/junye/Documents/code/visualstudio/cephalopod%20camouflage/docs/paper_design.pdf)

直接运行：

```bash
python scripts/export_paper_pdf.py
```

如果要自定义输入和输出：

```bash
python scripts/export_paper_pdf.py \
  --input docs/paper_design.md \
  --output docs/paper_design.pdf
```

这条导出链路使用：

- `pandoc` 负责 `Markdown -> HTML`
- 本地 Chrome headless 负责 `HTML -> PDF`

因此需要系统里能找到 `pandoc` 和 Google Chrome/Chromium。

## 参考文献

- Iskarous K, Mather J, Alupay J. *A Turing-based bimodal population code can specify Cephalopod chromatic skin displays*. arXiv. https://arxiv.org/abs/2205.11500
- Ishida T. *A model of octopus epidermis pattern mimicry mechanisms using inverse operation of the Turing reaction model*. PLoS ONE. https://pubmed.ncbi.nlm.nih.gov/34379702/
- Montague TG. *Neural control of cephalopod camouflage*. Current Biology. https://pubmed.ncbi.nlm.nih.gov/37875091/
- Messenger JB. *Cephalopod chromatophores: neurobiology and natural history*. Biological Reviews. https://doi.org/10.1017/S1464793101005772
- *Reconstruction of Dynamic and Reversible Color Change using Reflectin Protein*. Scientific Reports. https://www.nature.com/articles/s41598-019-41638-8
