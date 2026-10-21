# MeetScribe · 会议转写助手

本地运行的会议录音转写与纪要工具：实时分角色字幕、会后一键生成纪要，全程离线（ASR 与可选的大模型都在本机跑），录音与转写文件只存在自己的电脑上。

Windows 桌面程序（PySide6），支持线上/线下两种会议模式，可打包成独立 exe + 桌面快捷方式。

## 功能

**录音与字幕**
- 一键开始 / 暂停 / 结束；窗口置顶小条，录音时不挡会议软件
- 双声道采集：麦克风（线下现场多人）+ 系统声音回环（线上会议），两轨按会议时间对齐
- 分角色实时字幕：句级声纹特征（基频 + 谱质心）增量聚类，自动归为说话人 A/B/C/D，可改名并沉淀为长期声纹档案
- 字幕不等整句说完就出现；句中停顿 ≥0.6s 时用离线模型滑窗预校正，说错可重显示
- 自动增益（AGC）：麦克风电平过低时自动拉起来，否则声纹和精修都会失效
- 会议软件监控：检测到 Teams / 钉钉 / 飞书 / 腾讯会议 / 微信 在开会时自动开始录音

**两遍识别**
- 流式 Paraformer 出实时字幕；每句定稿后交离线 Paraformer 重识别替换
- 标点用 CT-Transformer；同音字纠错用词表 + 拼音匹配，可批量替换并写入全局替换表
- 识别错误可逐句手改，或一次批量纠正

**纪要与回放**
- 本机 Ollama 跑总结模型，一键生成会议纪要（主题 / 观点 / 争议 / 结论 / 待办），带进度与预览
- 会议管理页：搜索（标题 / 日期 / 转写正文）、批量删除、转写编辑、说话人改名、录音压缩（FLAC → OPUS）、跨分段连续回放
- 意外关闭可恢复：会话清单落盘，重启后能识别并标记异常会话

## 运行环境

- Windows 10/11，Python 3.13
- 依赖见 `.venv`（主要：PySide6、sherpa-onnx、sounddevice、soundcard、soundfile、numpy）
- 纪要功能需要本机装 [Ollama](https://ollama.com/) 并拉一个模型，例如：

```bash
ollama pull qwen2.5:3b-instruct   # 纠错
ollama pull qwen2.5:7b-instruct   # 纪要
```

## 模型（不在仓库里）

仓库不含模型权重，需自行下载后放到 `models/` 下：

| 目录 | 用途 |
| --- | --- |
| `models/paraformer_zh_en/` | 流式识别（encoder.int8.onnx / decoder.int8.onnx / tokens.txt） |
| `models/paraformer_offline/` | 句末精修（model.int8.onnx / tokens.txt） |
| `models/punct_zh_en/` | 标点恢复（model.int8.onnx） |

模型来自 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 发布的中英双语 Paraformer 与 CT-Transformer 标点模型。

## 使用

```bash
# 源码运行
python run_app.py

# 启动自检（验证模型、配置、识别、音频设备是否正常）
python run_app.py --selftest

# 打包成桌面 exe（产物在 release/MeetScribe/，并创建桌面快捷方式）
python build_exe.py
```

首次启动会生成 `config/settings.json`，可在「设置」页调整设备、识别参数、存储格式与大模型。

## 目录结构

```
meetscribe/
  asr/        流式与离线识别、精修队列、分角色聚类
  audio/      双轨录音、AGC、设备枚举、会话清单、回放
  llm/        Ollama 客户端、纠错、纪要生成
  text/       标点、同音字纠错、热词替换
  ui/         主窗口、录音页、设置页、管理页、悬浮条、主题
  session_tools.py  会后再处理（编辑/纠正/改名/压缩/恢复）
  paths.py    源码态与打包态的路径解析
config/       热词表、替换表、用户配置
phase1..3/    各阶段调试与回归脚本
```

## 说明

- 录音、转写、纪要默认存在 `data/sessions/<会话>/` 下，不会上传任何地方
- 大模型是可选的：关掉后仍可正常录音、转写、精修、回放
- 声纹聚类是轻量实现（句级 F0 + 谱质心），两人对话区分稳定；多人交叉抢话场景仍建议人工核对说话人

## 许可

未声明许可证，默认保留所有权利。如需引用或二次分发请先联系作者。
