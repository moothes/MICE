# MICE

Source code for our work submitted to journal.

### Preparing data

You can prepare your pathology data following the steps described in [this link](https://github.com/mahmoodlab/SurvPath).   
Genomics data can be downloaded from 
After that, your should generate a ```.csv``` file to include the follow-ups (e.g., OS) and path to corresponding ```.pt``` file for the patients.

### Running 
```python
# Pre-training
python main.py MICE --stage=pretrain

# Fine-tuning
python main.py MICE --stage=finetune

# Testing
python test.py MICE
```
