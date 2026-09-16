# 拼音版（pocket-tts-zh-en-pinyin）AX650 适配报告（M0–M4，Python 闭环）

日期：2026-09-16 · 板端：AX650N（root@10.126.29.50）· 音色：Vivian.wav
工程：`/data/shared/huyuan/TTS_quant/pocket-tts-zh-en-pinyin-axera`

## 1. 与 zh-en 版的关键差异与格式审计

| 项 | 结论 |
|---|---|
| 交付格式 | **只有 int8 ONNX**（`step_model.onnx` 195MB，ORT 动态量化 QOperator：24×DynamicQuantizeLinear + 24×MatMulInteger） |
| 量化范围 | **仅 FlowLM 投影层**（in_proj/out_proj/linear1/linear2 × 6）；flow_net、mimi、attention 在文件中是 float |
| 结构 | 与 zh-en 版同构（I/O、cache 布局、节点命名一致），图手术脚本可复用 |
| 编码器 | `step_encoder.onnx` 为 fp32，但权重是新的（hash 不同），需重编 |
| 文本前端 | 中文→带声调拼音（pypinyin TONE3+轻声）+ tokens.txt（9279）＋英文 BPE；多音字显著更准 |

**Pulsar2 与 int8 输入**：Pulsar2 前端为 float+自插 QDQ 体系，不支持 QOperator 动态 int8 直接编译。
本工程无需该路径：上 NPU 的 `flow_net`/`mimi_conv` 在原文件中即 float；FlowLM 本就留 CPU（供应商 int8 终态）。
（如需 FlowLM 的 float 版：`W_fp=(W_int8−zp)×scale` 权重重构反量化，脚本可在旧模型 int8/fp32 对上先验证。）

## 2. 完成内容

| 阶段 | 产物 | 验证 |
|---|---|---|
| M0 基线 | `results/baseline.json` + `results/audio/vendor_int8_*.wav`（11 条） | 主机 int8 RTF ≈0.20 |
| M1 图手术 | 子图提取 / 窗口补丁 / 精简 AR / mimi 拆分 / 编码器 40f 档 | **逐位一致**（子图、补丁 no-op、拆分 max_abs=0；精简 AR maxdiff=0） |
| M2 量化 | `flow_net_step_fp32.axmodel`(36MB) · `mimi_conv_step.axmodel`(4.4MB) · `step_encoder_40f.axmodel`(24.8MB) | check2 全过 |
| M3 前端 | 拼音前端移植进板端运行时（tokens.txt+pypinyin+SPM） | 与 vendor demo **token 逐条相同**（唯一差异：我们对阿拉伯数字自动转写，属增强） |
| M4 板端闭环 | 板端本地包 `/root/pocket_tts_pinyin`，NPU 混合推理 | 见下 |

## 3. 板端指标（Python 混合运行时，NPU: flow_net FP32 + mimi_conv U16 + 编码器 40f）

- 平均 **RTF 0.798**（10 条测量项，加载 4.6s 不计），平均首帧 **205ms**
- 逐条 RTF 0.77~0.86（zh_long 19.2s 音频 RTF 0.807）

**回环 CER（board vs vendor 主机 int8）**：

| 文本 | board CER | vendor CER |
|---|---:|---:|
| zh_short / zh_notice / zh_long | 0% | 0% |
| en_short / en_long / mix_short | 0% | 0% |
| zh_duoyin（多音字） | **0%** | 0% |
| zh_number（中文数字） | 0% | 0% |
| zh_brand | 0% | 0% |
| zh_tongue（绕口令） | 25.0% | 31.25%（ASR 极限） |
| mix_long | 9.26% | 12.96% |

## 4. 重要发现（本模型专属）

1. **本模型对数值扰动远比 zh-en 敏感**：flow_net 烘焙图（7e-7 级常量折叠误差）在约 6 帧内即让波形与 vendor 分叉（cos 0.73→0.58）。
   - 因此板端 Python 提供两种 flow_net 模式：NPU axmodel（快，轨迹不同但 **CER=0%**）与 **CPU 原图（与 vendor 逐位一致）**；
   - 波形 cos 依旧不是验收指标 → 用 CER/试听。
2. 拼音前端对多音字有效：`zh_duoyin`（银行行长/音乐会/重新/重要）CER 0%，绕口令 CER 也优于字符版。
3. RTF 高于 zh-en（0.80 vs 0.67）主要来自板端负载与 flow 阶段；后续可套用 zh-en 的 C++ 加速方案（跨帧流水线+注意力融合）。

## 5. 复现

```bash
# 主机：基线 / 图手术 / 量化
python python/baseline_run_pinyin.py
python python/extract_subgraphs.py && python python/patch_flow_window.py
python python/make_flow_ar_onnx.py && python python/split_mimi_npu.py
python python/make_encoder_static.py && python python/prepare_flow_net_npu.py
python python/generate_calib.py --samples-per-run 24 && python python/generate_calib_split.py && python python/generate_calib_encoder.py
# Pulsar2: pulsar2_pinyin_flowfp32 / pulsar2_pinyin_check2 / pulsar2_pinyin_encoder

# 板端
python3 board/pocket_tts_axera.py --text "..." --reference models/Vivian.wav --output out.wav \
  --onnx-dir models --axmodel-dir models --cpu-model-dir models/cpu --mimi-split-dir models/mimi_split \
  --tokens-txt models/tokens.txt --bpe-model models/chn_jpn_yue_eng_ko_spectok.bpe.model \
  --threads 4 --prefill-threads 8 --npu-flow-net 1 --npu-mimi-conv 1 \
  --flow-ar-model flow_ar_step.onnx --flow-prefill-model flow_step_windowed.onnx \
  --mimi-tf-model mimi_transformer_step.onnx --encoder-dir models/encoder \
  --flow-net-model flow_net_step_fp32.axmodel --mimi-conv-model mimi_conv_step.axmodel
```
