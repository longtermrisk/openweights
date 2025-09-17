#!/bin/bash

# Configuration
WORK_DIR="/workspace/openweights"
DASHBOARD_DIR="$WORK_DIR/openweights/dashboard"
REPO_URL="https://github.com/longtermrisk/openweights"
REPO_BRANCH="main"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Clone repository if it doesn't exist
log "Setting up repository..."
if [ ! -d "$WORK_DIR" ]; then
    git clone -b $REPO_BRANCH $REPO_URL "$WORK_DIR"
    cd "$WORK_DIR"
else
    cd "$WORK_DIR"
    git pull origin $REPO_BRANCH
fi

# Setup Node.js
log "Setting up Node.js..."
if ! command -v nvm &> /dev/null; then
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
fi

# Install and use Node.js 20
log "Installing Node.js 20..."
nvm install 20
nvm use 20
nvm alias default 20

# Verify versions
log "Node.js version: $(node --version)"
log "npm version: $(npm --version)"

#!/bin/bash

# Update package list
apt update

# Install prerequisites
apt install -y software-properties-common curl

# Add deadsnakes PPA
add-apt-repository ppa:deadsnakes/ppa -y
apt update

# Install Python 3.11 and required tools
apt install -y python3.11 python3.11-distutils python3.11-venv

# Install pip for Python 3.11
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Update alternatives to make Python 3.11 the default
update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.11 1

# Set Python 3.11 as default
update-alternatives --set python /usr/bin/python3.11
update-alternatives --set python3 /usr/bin/python3.11
update-alternatives --set pip /usr/local/bin/pip3.11

# Verify installation
echo "Python version:"
python --version
echo "Python3 version:"
python3 --version
echo "Pip version:"
pip --version


# Initial frontend build
log "Building frontend..."
cd $DASHBOARD_DIR/frontend
npm install
npm run build

# Create static directory in backend
log "Moving frontend build to backend..."
mkdir -p ../backend/static
cp -r dist/* ../backend/static/

source deploy.sh
