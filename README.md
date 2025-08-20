# MindTunnel
Transform written content into immersive audio experiences. MindTunnel converts research papers, essays, and documents into high-quality audio files, allowing you to absorb knowledge during commutes, workouts, or any time reading isn't practical.


## Installation

### 1. Environment Setup

Create and activate a new conda environment:

```bash
conda create --name mindtunnel python=3.10 -y
conda activate mindtunnel
pip install -r requirements.txt
```

### 2. Google Cloud SDK Installation

**macOS:**
```bash
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/install_google_cloud_sdk.bash
bash install_google_cloud_sdk.bash
source ~/.zshrc
```

**Other platforms:** Follow the [official Google Cloud SDK installation guide](https://cloud.google.com/sdk/docs/install) for your operating system.

### 3. Authentication

Set up Google Cloud authentication:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project <your-project-name>
```

> **Note:** Replace `<your-project-name>` with your actual Google Cloud project ID.

## Usage

### Quick Start

1. **Collect content:**
   ```bash
   python crawler.py
   ```

2. **Convert to audio:**
   ```bash
   python batch_all.py
   ```

   > ⚠️ **Long Process Warning**: Batch conversion can take considerable time depending on the number and length of documents. We recommend running this in a terminal multiplexer like `tmux` or `screen`:
   > 
   > ```bash
   > tmux new-session -d -s mindtunnel
   > tmux send-keys -t mindtunnel "conda activate mindtunnel && python batch_all.py" Enter
   > ```

## Configuration

Ensure your Google Cloud project has:
- Text-to-Speech API enabled
- Appropriate billing configured
- Sufficient quota for your usage needs
