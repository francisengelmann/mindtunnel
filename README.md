# MindTunnel


## About

MindTunnel converts text (such as essays or papers) into an audio file -- listen to reearch papers or essays whenever you have time.

## Installation

First, set up the environment:
```bash
conda create --name mindtunnel python=3.10 -y
conda activate mindtunnel
pip install -r requirements.txt
```

Next, install google cloud sdk (example here is on Mac, likely sth similar on your OS)

```bash
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/install_google_cloud_sdk.bash
bash install_google_cloud_sdk.bash
source ~/.zshrc
```

Do the gcloud auth

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project <project_name>
```
## Running

First get the essays
```bash
python crawler.py
```

Then convert them. This will take a long time, so make sure to run it in tmux or similar.
```bash
python batch_all.py
```

