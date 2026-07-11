# 图片发布 Server — API 文档

**Base URL:** `http://127.0.0.1:{port}`（默认端口 `18765`，可在 GUI 中修改）

> 每条记录独立维护两个平台的投稿状态：`pixiv_status` 和 `chichipui_status`，互不影响。

---

## 1. 查询队列列表

```
GET /api/queue
```

### Query Parameters

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `page` | int | 否 | `1` | 页码 |
| `per_page` | int | 否 | `20` | 每页条数 |
| `pixiv_status` | string | 否 | 无 | 按 Pixiv 状态筛选：`pending` / `uploading` / `uploaded` |
| `chichipui_status` | string | 否 | 无 | 按 Chichi-pui 状态筛选：`pending` / `uploading` / `uploaded` |

> 两个筛选条件可同时使用（AND 逻辑）。

### 响应示例

```json
{
  "page": 1,
  "per_page": 20,
  "total": 42,
  "items": [
    {
      "id": 1,
      "uuid": "a1b2c3d4-5678-90ab-cdef-1234567890ab",
      "image_path": "D:\\code\\image-maker\\data\\20260708\\ee721fed_003711-18b7de.jpg",
      "image_md5": "d41d8cd98f00b204e9800998ecf8427e",
      "json_path": "D:\\code\\image-maker\\data\\20260708\\20260708-003608-ee721fed-紫苑の天光.json",
      "image_base64": "/9j/4AAQSkZJRg...",
      "metadata": {
        "positive_prompt": "...",
        "negative_prompt": "..."
      },
      "pixiv_status": "pending",
      "chichipui_status": "uploaded",
      "created_at": "2026-07-08 12:00:00",
      "updated_at": "2026-07-08 12:00:00"
    }
  ]
}
```

### 字段说明

| 字段 | 说明 |
|------|------|
| `pixiv_status` | Pixiv 投稿状态：`pending` / `uploading` / `uploaded` |
| `chichipui_status` | Chichi-pui 投稿状态：`pending` / `uploading` / `uploaded` |
| `image_base64` | 图片的 base64 编码 |
| `metadata` | 匹配到的 JSON 元数据对象；未匹配到时为 `null` |
| `json_path` | 匹配到的 JSON 文件路径；未匹配到时为 `null` |

---

## 2. 获取最新一条待上传数据

```
GET /api/queue/latest-pending?platform={pixiv|chichipui}
```

### Query Parameters

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `platform` | string | 否 | `pixiv` | 指定平台：`pixiv` 或 `chichipui` |

返回该平台 `status = 'pending'` 的最新一条记录。

### 有数据时

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "id": 1,
    "uuid": "a1b2c3d4-...",
    "image_path": "...",
    "image_md5": "...",
    "json_path": "...",
    "image_base64": "/9j/4AAQ...",
    "metadata": { "prompt": "..." },
    "pixiv_status": "pending",
    "chichipui_status": "pending",
    "created_at": "2026-07-08 12:00:00",
    "updated_at": "2026-07-08 12:00:00"
  }
}
```

### 无待上传数据时（HTTP 200）

```json
{ "code": 1, "message": "no pending item for pixiv", "data": null }
```

---

## 3. 查询特定 UUID 数据

```
GET /api/queue/{uuid}
```

返回字段同列表中的单条 `item`。

### 未找到时

```json
{ "code": 404, "message": "uuid not found" }
```

---

## 4. 获取图片原始文件（Blob）

```
GET /api/queue/{uuid}/image
```

返回原始图片二进制数据，`Content-Type` 自动匹配图片格式（`image/jpeg` / `image/png` 等）。

前端可直接通过 `fetch().then(r => r.blob())` 获取 Blob 对象，无需 base64 解码转换。

---

## 5. 修改状态

```
PUT /api/queue/{uuid}/status
```

### Request Body (JSON)

```json
{
  "platform": "pixiv",
  "status": "uploading"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `platform` | string | 是 | 目标平台：`pixiv` 或 `chichipui` |
| `status` | string | 是 | 新状态：`pending` / `uploading` / `uploaded` |

### 成功响应

```json
{ "code": 0, "message": "ok", "uuid": "a1b2c3d4-...", "platform": "pixiv", "status": "uploading" }
```

### 错误响应

```json
{ "code": 400, "message": "invalid platform, use pixiv or chichipui" }
```

```json
{ "code": 400, "message": "invalid status" }
```

```json
{ "code": 404, "message": "uuid not found" }
```

---

## 油猴脚本使用示例（Blob 方式）

```javascript
const BASE = "http://127.0.0.1:18765";

// 1. 拉取 Pixiv 最新待上传图片元数据
const resp = await fetch(`${BASE}/api/queue/latest-pending?platform=pixiv`);
const item = await resp.json();
if (item.code !== 0) {
    console.log("无待上传图片:", item.message);
    return;
}
const data = item.data;

// 2. 获取图片 Blob
const blob = await fetch(`${BASE}/api/queue/${data.uuid}/image`).then(r => r.blob());

// 3. 构造 File 对象，赋给页面的 file input
const file = new File([blob], data.image_path.split(/[/\\]/).pop(), { type: blob.type });
const dt = new DataTransfer();
dt.items.add(file);
document.querySelector("input[type=file]").files = dt.files;

// 4. 上传成功后修改 Pixiv 状态
await fetch(`${BASE}/api/queue/${data.uuid}/status`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ platform: "pixiv", status: "uploaded" })
});
```

---

## 元数据 JSON 匹配规则

图片拖入队列时，自动在同目录查找匹配的元数据 JSON 文件：

| 步骤 | 示例 |
|------|------|
| 图片文件 | `ee721fed_003711-18b7de.jpg` |
| 取 `_` 分割第一段 → key | `ee721fed` |
| 扫描同目录 `.json` 文件 | `20260708-003608-ee721fed-紫苑の天光.json` |
| 取 `-` 分割第 3 段（索引 2）| `ee721fed` ✓ 匹配 |

---

## 状态流转

两个平台独立维护，各自的流转逻辑相同：

```
pending ──→ uploading ──→ uploaded
  ↑            │              │
  └────────────┘              │
  (可回退重传)                └── 两平台都 uploaded 后可在 GUI 中清除
```
