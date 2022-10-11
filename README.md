# Attention based deep learning on EHR data

Replication of recent deep learning models using attention. SARD [1], RETAIN[2] and a Variational GNN [3]. Developed using 
the OMOP common data model. The fitted models are stored in /SavedModels

### How to Install

    * Install python dependancies from the requirements.txt file
    * Install R requirements from renv.lock file


### How to Run

* First cohorts need to be defined, either with sql or an OHDSI tool like [ATLAS](http://github.com/OHDSI/Atlas).
* Then they need to be saved in a cohort table with the following columns:
  cohort_definition_id, subject_id, cohort_start_date, cohort_end_date. There will be two cohorts, the target and
  outcome cohort with different id's.
* You need to configure a config.yml file with your database path and access credentials (or use environment variables and ```Sys.getenv()```
* Then you can run the R script extractPLPData.R from the command line:  
  `Rscript extractPLPData.R cohortSchema cohortTable cohortId outcomeId pathToOutput pathToPython`
    * cohortSchema is the schema where you have saved your cohorts
    * cohortTable is the table with your cohorts
    * cohortId is the cohort_definition_id of your target cohort
    * outcomeId is the cohort_definition_id of your outcome cohort
    * pathToOutput is the folder where to save output, should be something like './data/task'
    * pathToPython is the path to your python environment
* Then you can run main.py, you need to modify the first lines in the main part to point to the saved data
    * The data needs to be saved in './data/task/' where task is the prediction task
    * You select one of the models in /models to run on the task. Currently support models are:
        - [RETAIN](http://arxiv.org/abs/1608.05745) [2] with some modifications
            - A bidirectional LSTM is used, and the continuous/non-temporal features are concatenated to visit
              embeddings
        - A Transformer from [1]. With addition of non-temporal features being concatenated to embeddings
        - SARD from [1]. With same addition as for the Transformer
        - [Variational GNN](http://arxiv.org/abs/1912.03761) [3]



References

1. Kodialam RS, Boiarsky R, Lim J, Dixit N, Sai A, Sontag D. Deep Contextual Clinical Prediction with Reverse
   Distillation. Proc AAAI Conf Artif Intell 2020;35:249-58 
2. Choi, Edward et al. 2016. ‘RETAIN: An Interpretable Predictive Model for Healthcare Using Reverse Time Attention Mechanism’. Advances in Neural Information Processing Systems: 3512–203.
3. Zhu, Weicheng, and Narges Razavian. 2021. ‘Variationally Regularized Graph-Based Representation Learning for Electronic Health Records’. In ACM CHIL 2021 - Proceedings of the 2021 ACM Conference on Health, Inference, and Learning, arXiv, 1–13. (March 5, 2021).

