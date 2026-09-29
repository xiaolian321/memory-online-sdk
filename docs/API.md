# Online Memory SDK 接口文档

进程内 Python SDK。**不负责生成辅导回复**；业务方用 `retrieve` 取记忆，用自有 LLM 生成 `reply`，再用 `commit_turn` 写入。

## 边界：谁调 LLM？

| 路径 | 是否调文本 LLM | 说明 |
|------|----------------|------|
| `retrieve` | **否** | 仅 embedding + 检索（GPU） |
| `commit_turn`（同步） | **否** | 写缓冲、计数 |
| 后台 observe / consolidate | **是** | 每 N 个学生轮 |
| `end_session` → profile | **是** | 会话摘要 + L2 特质 |
| 辅导 `reply` | **否（本 SDK）** | 由业务传入 `commit_turn(..., reply=...)` |

---

## 三层记忆（`memory_mode=layered` 默认）

| 层 | 含义 | 出现在 `prompt_block` |
|----|------|------------------------|
| L0 观察 | 单次作答事实 | 【最近证据】≤2 条 |
| L1 知识点卡 | struggling / learning / mastered | 【相关知识点】≤3 张 |
| L2 特质 | 跨题习惯 | 【学习者特质】≤3 条 |

---

## 调用时序

1. `retrieve(learner_id, session_id, query, question_text=...)`
2. 业务 LLM：`prompt_block` + 学生话 → **reply**
3. `commit_turn(..., query, reply, msg_id=唯一, used_item_ids=...)`
4. 会话结束：`end_session` → 可选 `wait_pending()`

---

## `retrieve`

### 输入

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `learner_id` | str | 是 | 学习者 ID |
| `session_id` | str | 是 | 会话 ID |
| `query` | str | 是 | 当前学生 utterance |
| `question_text` | str | 否 | 题面/任务描述 |
| `k` | int | 否 | 默认 5 |
| `force` | bool | 否 | True 时忽略闲聊门控仍检索 |

### 输出 `RetrieveResponse`

| 字段 | 说明 |
|------|------|
| `trace_id` | 追踪 ID |
| `intent` | task / concept / instruction / chitchat |
| `need_memory` | 是否建议检索 |
| `items[]` | 见下表 |
| `prompt_block` | 可直接拼进辅导 prompt 的中文块 |
| `timings_ms` | embed, intent, score, total |
| `n_candidates` | 候选数 |

**`items[]` 元素**

| 字段 | 说明 |
|------|------|
| `item_id` | 如 `topic:<uuid>`、`trait:<uuid>` |
| `layer` | trait / topic / observation |
| `type` | 同 layer 或 topic_card |
| `content` | 展示文本 |
| `status` | 知识点卡状态等 |
| `score` | 相关度 |
| `topic` | 标签 |
| `breakdown` | dense/bm25 等 |

---

## `commit_turn`

### 输入

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `learner_id`, `session_id` | str | 是 | |
| `query` | str | 是 | 学生话 |
| `reply` | str | 是 | **业务侧**生成的老师回复（SDK 不生成） |
| `msg_id` | str | 是 | 幂等键 |
| `question_text` | str | 否 | |
| `used_item_ids` | list | 否 | 本轮用到的记忆 ID |

### 输出 `CommitResponse`

| 字段 | 说明 |
|------|------|
| `turn_index` | 会话累计轮次 |
| `student_turn_id`, `tutor_turn_id` | 内部 turn_id |
| `counter` | 距下次后台抽取剩余学生轮数 |
| `extract_scheduled` | 是否已提交后台任务 |
| `pending_jobs` | 未完成后台任务数 |
| `duplicate_msg` | msg_id 重复 |

---

## 辅助接口

```python
client.end_session(learner_id, session_id)
client.wait_pending(timeout=120.0)
client.snapshot(learner_id)
client.snapshot_layers(learner_id)
client.close()
```

持久化：

- 会话：`store/<hash>/sessions/<session_id>.json`
- 三层：`observations.jsonl`, `topics.jsonl`, `traits.jsonl`, `session_digests.jsonl`

---

## 配置

文件：`config/memory_online.json`（可从 `memory_online.example.json` 复制）

| 键 | 含义 |
|----|------|
| `extract_model` | 观察/精炼/特质用的模型名 |
| `extract_every_n` | 每 N 个学生轮触发后台 LLM |
| `memory_mode` | `layered`（默认）或 `flat` |
| `kp_resolver` | `open` 或 `kg:<course_id>` |
| `embedding_model_id` / `embedding_model_dir` | 向量模型 |
| `embedding_device` | 如 `cuda:0` |
| `store_dir` | 默认 store 根目录 |

环境变量与安装（含 **embedding 使用项目目录 / 同服务器已有模型路径**）见 **[ENVIRONMENT.md](./ENVIRONMENT.md)**；pip 依赖见 **[requirements.txt](./requirements.txt)**。

| 变量 | 用途 |
|------|------|
| `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` | 后台抽取 LLM |
| `MEMORY_ONLINE_LLM_BASE_URL` | OpenAI 兼容 base URL |
| `MEMORY_ONLINE_EMBEDDING_DIR` | 本地 embedding 模型目录（可与 `embedding_model_dir` 二选一） |
| `MEMORY_ONLINE_ENV_FILE` | 可选 key 文件路径 |
| `MEMORY_ONLINE_LLM_CACHE` | LLM 磁盘缓存目录 |

---

## LLM 调用与替换

后台抽取**统一经过** [`se/core/memory/online/llm.py`](../se/core/memory/online/llm.py)：

| 函数 | 调用方 |
|------|--------|
| `chat_json_for_model` | `observe_window.py`, `consolidate.py`, `profile.py`,（flat 时 `extract_window.py`） |
| `chat_text_for_model` | 预留 |

**更换厂商 / 协议时建议只改一处：**

1. **`_chat_completions()`** — HTTP 请求入口（当前为 OpenAI SDK + 兼容 base_url）
2. 或整体替换 **`_openai_client()`** 的构造逻辑
3. 环境变量：`DEEPSEEK_API_KEY`、`OPENAI_API_KEY`、`MEMORY_ONLINE_LLM_BASE_URL`

**不要改** `retrieve` / `embedder.py` 来调文本 LLM。

模型名在 `config/memory_online.json` 的 `extract_model`（如 `deepseek-chat`）。

Prompt 模板目录：`se/core/memory/online/prompts/`（`observe_v1.txt`, `consolidate_v1.txt`, `profile_v1.txt`）。

LLM 结果缓存：[`se/core/memory/online/llm_cache.py`](../se/core/memory/online/llm_cache.py)，目录由 `MEMORY_ONLINE_LLM_CACHE` 控制。

---

## 错误行为

| 情况 | 行为 |
|------|------|
| 缺少 `msg_id` | `ValueError` |
| 重复 `msg_id` | `duplicate_msg=True` |
| 无 API Key 且 cache miss | 后台任务失败，`wait_pending` 可能 `RuntimeError` |
| 闲聊 + `force=False` | `items` 可能为空 |
