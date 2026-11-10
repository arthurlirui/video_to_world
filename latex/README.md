# Paint2World — LaTeX Paper (CVPR 2027)

CVPR 2027 官方模板（[cvpr-org/author-kit](https://github.com/cvpr-org/author-kit)）下的 Paint2World 论文。

## 目录结构

```
latex/
├── main.tex                # 论文主入口（编译此文件）
├── preamble.tex            # 额外宏包与自定义命令（hyperref 之前加载）
├── cvpr.sty                # CVPR 2027 官方样式
├── ieeenat_fullname.bst    # CVPR 2027 官方参考文献样式
├── references.bib          # 参考文献数据库
├── sec/                    # 各章节独立文件
│   ├── 0_abstract.tex      # 摘要
│   ├── 1_introduction.tex  # 1 引言
│   ├── 2_related_work.tex  # 2 相关工作
│   ├── 3_method.tex        # 3 方法（Stage -1 / -0.5 / 0 / 1 / 3.2）
│   ├── 4_experiments.tex   # 4 实验（含 6 个表格）
│   └── 5_conclusion.tex    # 5 结论
└── figures/                # 论文插图
```

## 编译

```bash
cd latex
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

输出 `main.pdf`（9 页，review 模式带行号与 "CONFIDENTIAL REVIEW COPY" 页眉）。

## 版式切换

`main.tex` 顶部三选一：

```latex
\usepackage{cvpr}              % Camera-ready（终稿）
\usepackage[review]{cvpr}      % 投稿评审版（当前默认）
\usepackage[pagenumbers]{cvpr} % arXiv 版（强制页码）
```

切换 hyperref 开关后需删除 `*.aux` 再编译。

## 验证状态

- 编译错误：0
- 未定义引用（citation/reference）：0
- Overfull/Underfull hbox：0
- 页数：9（双栏）
