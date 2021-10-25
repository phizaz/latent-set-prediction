# CLEVR dataset

## Retrieve
Download the CLEVR images (18 GB) https://drive.google.com/file/d/1shG9LOC51UbcqLx2G6-kLrDczh58HXkR/view?usp=sharing (We do not have the license to share, but we included for convenience of the reader)

Extract the archive into the directory `data/clevr_text`. In the final state, you should see the following directory structure:

```
data/clevr_text
- images
- clevr_train_train.csv
- clevr_train_val.csv
- clevr_val.csv
```

# MIMIC-CXR report dataset

## Retrieve

Download the corresponding .csv files (~30 MB) from https://drive.google.com/file/d/1aWifveUxx9n74SYjcMqdEfUR9wcEa4g2/view?usp=sharing 

Download the chest x-ray images (51 GB) https://drive.google.com/file/d/1TZkTeHRDtwQN8vdRxMgii30r_QMo7I_b/view?usp=sharing (We do not have the license to share, but we include for convenience of the reader)

Extract the both archives into the directory `data/mimic_cxr`. In the final state, you should see the following directory structure:

```
data/mimic_cxr
- images512
- mimic_reports_with_findings_ready.csv
- mimic-cxr-record-ap.csv
- mimic-cxr-record-front.csv
- mimic-cxr-record-pa.csv
- test.csv
- train.csv
- val.csv
```
