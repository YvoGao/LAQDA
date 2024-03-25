# Improve Meta-learning for Few-Shot Text Classification with All You Can Acquire from the Tasks

The code for LAQDA.


## quick start

### create ENV
```
conda create -n LAQDA python=3.7
source activate LAQDA
pip install -r requirements.txt
```

### run
**Noting:** before you start, you should download bert-base-uncased from https://huggingface.co/google-bert/bert-base-uncased, and change the path in the run.sh file to your own file path.
The specific parameters per dataset in the paper are consistent with run.sh.
```
sh run.sh
```


