# Pocket-TTS 中英双语 · 拼音版 · AX650 部署工具链

[Pocket-TTS 中英双语社区版（拼音版）](https://www.tulingyun.com/tts_clone.html)（Mimi 编解码器 + FlowLM/LSD，零样本声音克隆，
中文按**带声调拼音**建模、多音字更准）在 **AX650N** 上的适配工程。

> 本工程与 [pocket-tts-zh-en.AXERA](https://github.com/ZY-2012/pocket-tts-zh-en.AXERA)（**fp32 ONNX 起点**）不同：
> 供应商只提供 **int8 ONNX**（无 fp32 权重），全程需要在**仅有 int8 交付件**的前提下完成 NPU 适配。

推理包（axmodel + int8 ONNX + 运行时 + 词表/音色）见 Hugging Face：
**https://huggingface.co/HY-2012/pocket-tts-zh-en-pinyin.AXERA**

## int8-only 的适配策略（本工程核心）

| 事实 | 对策 |
|---|---|
| `step_model.onnx` 是 ORT **动态量化 QOperator**（24×`DynamicQuantizeLinear`+`MatMulInteger`） | Pulsar2 前端是 float+QDQ 体系，**不能直接吃**这种 int8 图 |
| 24 个量化矩阵乘**全部是 FlowLM 投影层**；`flow_net`、`mimi 卷积/transformer` 在文件里仍是 float | **混合路线**：FlowLM 留在 CPU int8（供应商终态）；float 部分正常走 Pulsar2 编译 axmodel |
| 音色编码器是 fp32 | 静态 40f 档 + Pulsar2 U16（同 zh-en 工程） |

如果将来必须把 FlowLM 也变成 float（例如要上 NPU 或自行重量化）：
**权重重构反量化** `W_fp = (W_int8 − zp) × scale`，把
`DynamicQuantizeLinear→MatMulInteger→Cast→Mul` 链替换为 float MatMul + 反量化权重常量。
该方法可用「同一模型的 int8/fp32 双份」先验证（本仓对应 zh-en 工程有此条件）。

## 实测指标（AX650N，参考音色 Vivian）

| 项目 | 数值 |
|---|---|
| 平均 RTF（11 条中英文本，Python 混合运行时） | **0.798** |
| 首帧延迟（含音色编码+prefill） | **205 ms** |
| 回环 CER（SenseVoiceSmall） | 常规中英文本 **0%**；多音字 0%；绕口令 25%（ASR 极限） |
| vendor int8（主机 CPU 对照） | RTF ≈0.20，同文本 CER 一致 |

## 运行时架构（混合 NPU / CPU）

| 阶段 | 后端 | 模型 |
|---|---|---|
| 音色编码 | **NPU** | `step_encoder_40f.axmodel`（静态 3.2s 档，24.8MB） |
| FlowLM prefill / AR 步 | CPU ORT **int8（供应商原图）** | `flow_step_windowed.onnx` / `flow_ar_step.onnx` |
| flow_net | **NPU FP32** | `flow_net_step_fp32.axmodel`（36MB） |
| Mimi transformer | CPU fp32 | `mimi_split/mimi_transformer_step.onnx` |
| Mimi 卷积解码 | **NPU U16** | `mimi_conv_step.axmodel`（4.4MB） |

文本前端：阿拉伯数字→中文 → 全角转半角/小写 → 中文转**带声调拼音**（pypinyin TONE3+轻声）→ `tokens.txt` 行号取 id；
英文/标点走 BPE。与 vendor `demo.py` 的 token **逐条对拍一致**（已覆盖多音字用例）。

## 主机侧复现

```bash
export PINYIN_TTS_ROOT=/path/to/pocket-tts-zh-en-pinyin   # 供应商 int8 权重目录
python python/baseline_run_pinyin.py                      # vendor 基线（金标准 wav + RTF）
python python/extract_subgraphs.py                        # 从 int8 图提取 flow/flow_net/mimi（逐位一致）
python python/patch_flow_window.py                        # flow KV 窗口位置补丁
python python/make_flow_ar_onnx.py                        # 精简 AR（烘焙 gates）
python python/split_mimi_npu.py                           # mimi -> transformer + conv（逐位一致）
python python/prepare_flow_net_npu.py                     # 烘焙 decode_steps（Pulsar2 需要）
python python/make_encoder_static.py                      # 编码器静态档
python python/generate_calib.py --samples-per-run 24      # 校准数据
python python/generate_calib_split.py && python python/generate_calib_encoder.py
# Pulsar2（三份配置：flow_net FP32 / mimi_conv U16 / encoder U16，check2）
```

板端（HF 包内 `run_ax650.sh` 一键；精确模式 `NPU_FLOW_NET=0` 让 flow_net 回 CPU 原图）：

```bash
bash run_ax650.sh "你好，世界。" out.wav
```

## 关键发现（本模型专属，务必注意）

1. **本模型对数值扰动极其敏感**：把 `flow_net` 的 `decode_steps` 烘焙为常量（常量折叠误差仅 ~7e-7）
   就会让波形在约 6 帧后与 vendor 分叉（PCM cos 0.73→0.58）；NPU FP32 同理。
2. 因此 **PCM 波形 cos 不是验收指标**：板端输出与 vendor 波形不同，但回环 CER 0%；
   验收用 **CER + 人工试听**。如果必须与 vendor 逐位一致，用精确模式（flow_net 走 CPU 原图）。
3. 拼音版对多音字确实更准：`银行行长/音乐会/重新/重要` 用例 CER 0%。

## 目录

```
configs/     # 基线/试听文本集
python/      # 拆图、补丁、静态化、校准、Pulsar2 配置、拼音前端、验证、CER
board/       # AX650 运行时（axengine + onnxruntime，含拼音前端）与批量驱动
scripts/     # ax650 量化脚本
docs/        # M0–M4 报告、基线指标、试听集 CER 表
```

## 模型与许可

- 上游模型权重：供应商社区版（拼音版，**CC BY-NC 4.0，仅限非商业**）；商用需另行授权。
- 本仓代码：Apache License 2.0。
- 声音克隆请确保已获得被克隆者授权。

## 参考 & 感谢

- **原工程**：[Pocket-TTS 中英双语社区版（拼音版，图灵云）](https://www.tulingyun.com/tts_clone.html) —— int8 ONNX 权重来源
- **上游架构**：[kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts) —— CALM / Lagrangian Self Distillation
- **拼音分词**：[pypinyin](https://github.com/mozillazg/python-pinyin)
- **转换与量化工具**：[AXERA-TECH/Magnetar](https://github.com/AXERA-TECH/Magnetar) —— 模型 → ONNX → Pulsar2 → AXMODEL 工作流
- **推理与评测栈**：[ONNX Runtime](https://github.com/microsoft/onnxruntime)、[AXERA pyaxengine](https://github.com/AXERA-TECH/pyaxengine)、Pulsar2（AX650 BSP）、[FunASR SenseVoiceSmall](https://github.com/modelscope/FunASR)（回环 CER）
- **板端评测框架**：[ZY-2012/Voice_Test.AXERA](https://github.com/ZY-2012/Voice_Test.AXERA)
- **参考音色**：`Vivian.wav`（来自原工程分发包）
- **姊妹工程**：[pocket-tts-zh-en.AXERA](https://github.com/ZY-2012/pocket-tts-zh-en.AXERA)（fp32 ONNX 起点 · 含 C++ 加速）
