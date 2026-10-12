#!/bin/bash

# pythonパッケージインストール
if [ -f requirements.txt ]; then
    sudo uv pip install --system -r requirements.txt
fi

# Claude Codeのインストール
npm install -g @anthropic-ai/claude-code