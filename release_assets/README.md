# 完整项目大文件恢复说明

GitHub 普通 Git 单文件上限为 100MiB，而本项目包含多个 0.12–4.06GB 的模型、FSDP 检查点和安装包。为确保整个项目可完整传递，同时保持源码和评测结果可以在仓库中直接浏览，这些超限文件存放在 GitHub Release：

- Tag：`full-project-snapshot-2026-08-28`
- Release：<https://github.com/ilovecplusplus230/-OPD/releases/tag/full-project-snapshot-2026-08-28>
- 完整清单：`LARGE_FILES_MANIFEST.json`

## 一键下载并还原

克隆仓库后，在仓库根目录执行：

```bash
python release_assets/restore_release_assets.py \
  --repo-root . \
  --assets-dir release_parts \
  --download
```

脚本会逐片下载、校验每个分片的 SHA256、重组原文件，再校验原文件 SHA256。已有且校验正确的文件会自动跳过。

## 只验证已有文件

```bash
python release_assets/restore_release_assets.py \
  --repo-root . \
  --assets-dir release_parts \
  --verify-only
```

## 使用 GitHub CLI 手动下载

```bash
mkdir -p release_parts
gh release download full-project-snapshot-2026-08-28 \
  --repo ilovecplusplus230/-OPD \
  --dir release_parts

python release_assets/restore_release_assets.py \
  --repo-root . \
  --assets-dir release_parts
```

不要单独解压某个分片；分片不是压缩包，必须由恢复脚本按照清单顺序重组。
