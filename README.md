# MICE

Source code for our work submitted to journal.

## Preparing data

For TCGA dataset, pathology images, mRNA expression data, and a clean version of clinical reports are publicly available at [pathology images](https://portal.gdc.cancer.gov/), [genomics data](https://www.cbioportal.org/), and [clinical reports](https://github.com/cpystan/Wsi-Caption), respectively.
The HANCOCK dataset is publicly available at [this link](https://hancock.research.fau.eu).

You can prepare your pathology data following the steps described in [this link](https://github.com/mahmoodlab/SurvPath) and use [UNI](https://github.com/mahmoodlab/UNI/tree/main) to extract patch embeddings.   
Then, leveraging the [BioBERT](https://github.com/dmis-lab/biobert) to extract features from clinical reports.     
Finally, your should generate a ```.csv``` file to include the follow-ups (e.g., OS) and path to corresponding multimodal data for all patients.


## Running 
```python
# Pre-training
python main.py MICE --stage=pretrain

# Fine-tuning
python main.py MICE --stage=finetune

# Testing
python test.py MICE
```
