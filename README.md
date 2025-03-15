# Karpathy GPT从零搭建资源链接

## 视频教程
1. Let's build GPT from scratch: https://www.youtube.com/watch?v=kCc8FmEb1nY
   - 时长：4小时44分钟
   - 发布日期：2023年4月
   - 描述：从头开始一步步构建GPT模型的完整教程

## GitHub仓库
1. build-nanogpt项目: https://github.com/karpathy/build-nanogpt
   - 描述：视频教程对应的代码，按照提交历史可以清晰看到从零开始构建的过程
   - 特点：代码整洁，注释详细，适合学习

2. nanoGPT项目: https://github.com/karpathy/nanoGPT
   - 描述：用于训练/微调中等规模GPT的最简单、最快的仓库
   - 主要用途：可以在单个GPU上训练小型GPT-2模型

3. llm.c项目: https://github.com/karpathy/llm.c
   - 描述：从头开始实现一个可以训练和推理GPT模型的C语言代码库
   - 特点：代码精简，可以在约90分钟内重现GPT-2 (124M)模型训练

## 训练与硬件
- GPT-2 (124M)模型的训练时间大约为:
  - 8x H100 GPU: ~40分钟
  - 单个A100 40GB: ~15小时
  - 单个消费级GPU (如RTX 4080): ~44小时

## 学习路径建议
1. 观看"Let's build GPT from scratch"视频，了解模型架构
2. 研究build-nanogpt代码库，跟随提交历史学习构建过程
3. 尝试使用nanoGPT训练小型模型
4. 探索llm.c了解更底层的实现
