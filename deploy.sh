#!/bin/bash
cd /opt/chatbot_env/git
echo "👉 Pulling latest code from GitHub..."
cp app.py ..
git reset --hard
git clean -fd
git pull origin main
echo "✅ Code updated at $(date)" >> deploy.log
