cd /opt/chatbot_env/git
echo "👉 Pulling latest code from GitHub..."
git reset --hard
git clean -fd
git pull origin main
cp *.py ..
echo "✅ Code updated at $(date)" >> deploy.log