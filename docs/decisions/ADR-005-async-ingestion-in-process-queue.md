# ADR-005 · 异步摄入用进程内队列，接受单副本限制

**日期**：2026-10-12
**状态**：已采纳
**相关**：`backend/services/ingest_service.py`、`backend/repositories/ingest.py`、
`backend/api/upload.py::upload_documents_async`、`tests/test_ingest_async.py`

## 背景

同步上传端点等摄入完成才返回。10MB 文件在 CPU 上要做解析、分块、embedding、
写向量库，耗时以分钟计，HTTP 请求会超时。T0c 已把阻塞调用全部 offload，
事件循环不再停摆，但请求本身仍然要等这么久。

验收 C8 要求：上传后立即返回 `task_id`，摄入在后台进行，客户端查询进度。

## 选项

| | 进程内 `asyncio.Queue` + 常驻 worker | Celery / RQ | 数据库队列（`FOR UPDATE SKIP LOCKED`） |
|---|---|---|---|
| 新增组件 | 无 | broker + 独立 worker 进程 + 监控 | 无（复用 PG） |
| 多副本 | **不支持** | 支持 | 支持 |
| 文件交接 | 本机临时文件 | 需要共享存储 | 需要共享存储 |
| 进程重启后的排队任务 | **丢失** | broker 持久化则保留 | 保留 |
| 实现量 | 约 200 行 | 配置 + 部署 | 约 300 行 + 轮询 |

## 决策

采用**进程内 `asyncio.Queue` + 单个常驻 worker**。

1. 本项目单副本部署，多副本没有需求方。为一个不存在的需求引入 broker，
   换来一整套额外的部署与监控面。
2. 后两种方案的前提都是**共享存储**：上传落到 A 副本的临时盘，B 副本的
   worker 读不到。只换队列不换存储，多副本照样跑不通。而共享存储（对象
   存储）在本项目里没有其它用途。
3. 只开一个 worker：embedding 是 torch 前向计算，跑在默认线程池里。
   多 worker 并发只会争同一批 CPU 与线程，吞吐不升，还会挤占检索请求要用
   的线程池。

## 状态放在哪

- **PG `ingest_tasks`**：pending → running → done / error，权威终态。
  `pending → running` 用 `UPDATE ... WHERE status='pending'` 做 CAS，
  靠 rowcount 判断是否抢到，不需要锁。
- **Redis hash `ingest:{task_id}`**：阶段与百分比，TTL 1 小时。
  每批 embedding 更新一次，高频写不打数据库；丢了只是进度条不动。

## 代价与兜底

**进程重启后，排队中与执行中的任务无法继续**：临时文件路径只在内存队列里，
没有落库。

曾考虑过把卡住的 running 打回 pending 重跑（"摄入幂等，重跑安全"）。
幂等本身成立，但重跑需要原始文件，路径已随进程丢失，打回 pending 的任务
永远不会被处理。

最终兜底：清扫器把 pending / running 超过 `INGEST_STALE_SECONDS`（默认
30 分钟）的任务标 error，提示用户重新上传。按**年龄**而非"早于本进程启动"
判断：后者在多副本下会误杀别的副本队列里的活任务。清扫器启动时扫一次，
之后每 `INGEST_SWEEP_INTERVAL_SECONDS` 扫一次。

## 何时重新评估

出现以下任一情况时改为"对象存储 + 数据库队列"：

- 需要多副本部署
- 单次上传量大到重启丢任务不可接受
- 摄入需要与 Web 进程隔离（例如改用 GPU 机器做 embedding）
