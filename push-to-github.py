# -*- coding: utf-8 -*-
# ## 推送主分支
# !git config --global user.email "acemyzoe@outlook.com"
# 出现一些类似 warning: LF will be replaced by CRLF in <file-name>. 可启用如下设置。
# !git config --global core.autocrlf false
# 配置打印历史commit的快捷命令
# !git config --global alias.lg "log --oneline --graph --all"

# !git init

# !git add  *.md *.py

# !git rm --cached  .ipynb_checkpoints/* 

# !git commit -m "revise" 

# !git remote rm origin 

# !git remote add origin main

# !git pull  origin main 

# !git push  -f origin main 

# ## 创建分支

# !git checkout -b gh

# !git rm --cached -r *.md

# !git clean -df
# !rm -rf *.md

# !cp -r _book/* .

# !git add .

# !git reset

# !git pull origin gh

# !git commit -m 'add gh'

# !git push -u origin gh

# !git checkout gh

# ## 更新命令

# !git checkout main

# !git add .

# !git commit -m "revise"

# !git push -u origin master









