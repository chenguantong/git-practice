# git-practice

Git 练习仓库：用于熟悉 `git init`、提交、远程协作与分支操作的日常实验。

## 快速开始

```bash
git clone https://github.com/chenguantong/git-practice.git
cd git-practice
```

## 目录结构

```
.
├── README.md      # 项目说明
└── .gitignore     # 忽略规则
```

## 常用命令

| 目的 | 命令 |
| --- | --- |
| 查看状态 | `git status` |
| 暂存改动 | `git add .` |
| 提交 | `git commit -m "message"` |
| 推送 | `git push` |
| 查看历史 | `git log --oneline --graph` |

## 说明

- 默认分支为 `main`。
- 本机 Git 使用 `openssl` TLS 后端（`schannel` 在此环境下会报 `SEC_E_NO_CREDENTIALS`）。
