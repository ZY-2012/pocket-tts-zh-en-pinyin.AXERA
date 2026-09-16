---
name: int8-onnx-axera
description: 只有 int8 ONNX（无 fp32 权重）的模型适配 AX650 板端的完整方法论——int8 格式鉴定（QOperator 动态量化 vs QDQ）、Pulsar2 能力边界、混合 NPU/CPU 拆分策略、权重重构反量化与验证、以及「量化扰动被自回归放大」的验收口径。当用户拿到供应商 int8 ONNX 要上 AXERA、问「int8 能不能走 Pulsar2」、需要把量化矩阵乘变回 float、或维护 pocket-tts-zh-en-pinyin.AXERA 工程时使用。样例工程：/data/shared/huyuan/TTS_quant/pocket-tts-zh-en-pinyin-axera。
---

# int8 ONNX → AX650：从「只有 int8 交付件」到板端推理

## 适用场景

- 供应商只给 **int8 ONNX**（无 fp32 权重/源码），要完成 AXERA（AX650）量化推理；
- 或已有 int8 图想确认「能否直接喂 Pulsar2」。

## 第 0 步：鉴定 int8 格式（决定一切）

```python
import onnx
from collections import Counter
m = onnx.load(path, load_external_data=False)
c = Counter(n.op_type for n in m.graph.node)
dt = Counter(onnx.TensorProto.DataType.Name(i.data_type) for i in m.graph.initializer)
print({k: c[k] for k in ("DynamicQuantizeLinear","MatMulInteger","QuantizeLinear",
                         "DequantizeLinear","QLinearConv","ConvInteger") if c[k]})
print(dict(dt))
```

| 特征 | 格式 | 说明 |
|---|---|---|
| `DynamicQuantizeLinear` + `MatMulInteger`（+Cast/Mul 缩放） | **QOperator 动态量化** | 激活 scale 运行时才计算；ORT/CPU 友好 |
| `QuantizeLinear`/`DequantizeLinear` 成对包裹 | QDQ（静态） | 各工具链通用，Pulsar2 前端有 QDQ 转换通道 |
| `QLinearConv`/`ConvInteger` | 卷积也量化了 | CPU EP 可能不支持 ConvInteger（ORT 报 NotImplemented） |

同时统计**量化节点落在哪些子图**（节点名前缀/位置），这决定第 1 步路线。

## 第 1 步：Pulsar2 能不能吃 int8？（结论与证据）

**QOperator 动态 int8 不能直接喂 Pulsar2。**
证据（npu-codebase）：
- 前端/量化链路围绕 **float ONNX + 自插 QDQ**（`frontend/transformations/qdq_conversion/*`、`quant/.../eliminate_qdq.py`、`collect_qdq_onnx_info.py`）；
- 全仓无 `MatMulInteger`/`DynamicQuantizeLinear` 输入支持（仅 PPQ 导出侧提到这些名字）；
- 直接 build 会报 shapefn/unsupported op 类错误。

**决策树**：

```text
量化矩阵乘集中在「本来就该留 CPU」的模块（递归 KV/敏感 AR）？
├─ 是 → 混合路线（推荐，零反量化）：
│      int8 部分 CPU ORT 原样跑（供应商终态，也是质量上限）
│      float 部分正常走 Pulsar2（本文档样例：flow_net FP32 axmodel + mimi_conv U16）
└─ 否（NPU 必须吃 int8 部分）→ 权重重构反量化成 float 图 → 再走 Pulsar2（见第 2 步）
```

样例（pocket-tts-zh-en-pinyin）：24 个量化矩阵乘全是 FlowLM 投影层 → FlowLM 留 CPU int8；
`flow_net`/`mimi` 在文件中本就是 float → 正常编译 axmodel。整个过程**不需要**反量化。

## 第 2 步（备选）：权重重构反量化

对每个 `MatMulInteger`：
```text
W_fp = (W_int8.astype(f32) − w_zero_point) × w_scale     # per-channel 时逐列广播
```
把 `DynamicQuantizeLinear → MatMulInteger → Cast → Mul(缩放链)` 替换为
`MatMul(x_fp32, W_fp_const)`；注意：
- `w_scale/w_zero_point` 可能是 initializer 也可能由节点计算，先按名字/生产者定位；
- 输出链上的 `Mul(y_scale × w_scale)` 等缩放要一起摘掉，否则数值翻倍；
- 完成后用 onnx.checker + **同模型 int8/fp32 双份对拍**（若拿得到 fp32）或单步输出 cos 验证；
- 反量化只还原「权重误差」，激活量化误差被丢弃（更准），最终质量以 CER/试听为准。

## 第 3 步：验收口径（int8/量化模型的铁律）

1. **PCM 波形 cos 不是验收指标**：任何量化/NPU 入环后，自回归会放大微小扰动，波形很快分叉。
   样例实测：flow_net 烘焙常量折叠误差仅 **7e-7**，6 帧后 PCM cos 掉到 0.73。
2. 验收 = **回环 CER/WER（ASR）+ 人工试听**；对照供应商 int8 在 CPU 上的输出（同 ASR 口径）。
3. 保留一个「**精确 CPU 回退**」开关（如 `--npu-flow-net 0` 用原图 ONNX），需要与供应商逐位一致时使用。
4. **不要对已有 int8 图再跑 `quantize_dynamic`**（双重量化，质量无收益）；只对自己的 float 子图量化。
5. 报告里同时给：RTF（不含加载）、首帧、CER、以及「与 vendor 输出的时长/帧数/EOS」对比。

## 样例工程与复现（pocket-tts-zh-en-pinyin-axera）

```bash
export PINYIN_TTS_ROOT=/path/to/pocket-tts-zh-en-pinyin   # 供应商 int8 目录
python python/baseline_run_pinyin.py        # vendor 金标准
python python/extract_subgraphs.py          # int8 融合图 -> 子图（逐位一致）
python python/patch_flow_window.py          # flow KV 窗口补丁
python python/make_flow_ar_onnx.py          # 精简 AR（烘焙 gates）
python python/split_mimi_npu.py             # mimi -> transformer(CPU)+conv(NPU)
python python/prepare_flow_net_npu.py       # 烘焙 decode_steps（Pulsar2 前置）
python python/make_encoder_static.py        # 编码器静态档（40f 档上限）
python python/generate_calib*.py            # 校准数据
# Pulsar2: flow_net(FP32) / mimi_conv(U16) / encoder(U16) check2
python python/validate_subgraphs.py && python python/validate_mimi_split.py
# 板端
bash run_ax650.sh "你好，世界。"        # NPU 混合（快）
NPU_FLOW_NET=0 bash run_ax650.sh ...   # 精确 CPU flow_net（与 vendor 逐位一致）
```

实测（AX650N，拼音版）：Python 混合 RTF **0.798**、首帧 205ms、常规中英 CER **0%**；
vendor int8（主机）RTF ≈0.20。

## 与其它 skill 的关系

| skill | 起点 | 场景 |
|---|---|---|
| `pocket-tts-zh-en-axera` | **fp32 融合 ONNX** | ONNX 图手术 + C++ 加速（含注意力融合、跨帧流水线） |
| **本 skill** | **int8 ONNX** | 量化格式鉴定 + 混合路线/反量化 + 验收口径 |
| `pocket-tts-axera-workflow` | PyTorch 源权重 | 从源头重导出 |
