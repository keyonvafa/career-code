# CAREER: Transfer Learning for Economic Prediction of Labor Sequence Data
Code for the paper: [CAREER: Transfer Learning for Economic Prediction of Labor Sequence Data.
](https://arxiv.org/abs/2202.08370)

<p align="center">
<img src="https://github.com/keyonvafa/career-code/blob/main/analysis/figs/CAREER_wide.png"   --width="370" height="370" />
</p>

## Getting Started

CAREER is a transformer-based model that learns a low-dimensional representation of an individual's job history to predict their next occupations. CAREER is first pretrained to learn representations on a large but noisy resume dataset. These representations are then "fine-tuned" to form predictions on survey datasets that may be more relevant to economists, such as NLSY or PSID.

The instructions below will first pretrain CAREER's representations on a resume dataset and then fine-tune these representations on a small survey dataset. This code assumes that you have access to resume data and to a survey dataset such as [NLSY](https://www.bls.gov/nls/nlsy97.htm) or [PSID](https://psidonline.isr.umich.edu).

### Software requirements and installation
Configure a virtual environment using Python 3.6+ ([instructions here](https://docs.python.org/3.6/tutorial/venv.html)).
Inside the virtual environment, use `pip` to install the required packages:

```{bash}
pip install -r requirements.txt
```
This code assumes that you have access to a GPU. Running the code without a GPU may be impractically slow. We use the neural sequence library [fairseq](https://github.com/pytorch/fairseq) to model job sequences. First, 
configure fairseq to be developed locally:
```{bash}
cd fairseq
pip install --editable ./
cd ..
```

Optionally, install NVIDIA's [apex](https://github.com/NVIDIA/apex) library to enable faster training
```{bash}
cd fairseq
git clone https://github.com/NVIDIA/apex
pip install -v --no-cache-dir \
  --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
cd ../..
```

## <a id="transfer_and_finetuning">Train CAREER on Resumes and Fine-Tune on Survey Dataset</a>
The instructions below will first pretrain CAREER's representations on a resume dataset and then fine-tune these representations on a small survey dataset. This code assumes that you have access to resume data and to a survey dataset such as [NLSY](https://www.bls.gov/nls/nlsy97.htm) or [PSID](https://psidonline.isr.umich.edu).

### <a id="data_formatting">Data formatting</a>
CAREER is trained on sequences of jobs and covariates to predict future jobs. CAREER uses two kinds of datasets: large, noisy resume datasets, which CAREER will use to learn an initial set of representations, and small, survey datasets upon which CAREER will fine-tune its job representations. Each kind of dataset should have training, validation, and test sequences of jobs and covariates. CAREER will train on all of the sequences in the training split; it will use the validation split for model selection; and it will be evaluated by its performance on the test split. 

Each split must have a sequence of jobs, and may contain as many of the supported covariates as desired: year, education, race/ethnicity, gender, and location. For example, the train split must contain the data file `train.job`, and it may contain as many of the following data files as desired: `train.year`, `train.education`, `train.ethnicity`, `train.gender`, and `train.location`. The same goes with the validation and test splits, replacing `train` with `valid` and `test`, respectively (e.g. resulting in the files `valid.job` and `test.job`). All splits must contain sequences of jobs; if a split contains other covariates as well, each covariate must be contained in all of the splits. Each file should end with a newline.

An example of data files in the correct format is [located here](https://github.com/keyonvafa/career-code/tree/main/sample-data) (these data files are for demonstration purposes only; they do not correspond to real sequences and they are too short for learning useful representations). Each row in a data file corresponds to one individual. In the job file, e.g. `train.job`, each row is a sequence of jobs. Jobs should be denoted with a classification code, such as [O*NET SOC](https://www.onetcenter.org/taxonomy/2019/list.html) or [occ1990dd](https://www.ddorn.net/data.htm). The exact coding scheme does not matter, as long as the coding is consistent. Use spaces to separate job timesteps; e.g., a sequence of two jobs should contain a job code, a space, and a job code. For example, the snippet below contains two sequences of jobs in the [O*NET SOC](https://www.onetcenter.org/taxonomy/2019/list.html) format: the first individual has recorded jobs for 8 timesteps, and the second individual has recorded jobs for 5 timesteps.

```
31-2021 31-2021 35-3011 27-1012 27-1012 27-1012 27-1012 27-1012
25-9044 23-1012 23-1011 11-1030 11-1030
```
Each covariate file should contain the same number of lines as the job file, since they should correspond to the same individuals. For example, if the first individual in the above example worked from 2001-2008 and the second individual worked from 2005-2009, the year file should look like:
```
2001 2002 2003 2004 2005 2006 2007 2008
2005 2006 2007 2008 2009
```
We consider education as another time-varying covariate, where each entry corresponds to the most recent educational degree. We consider the other covariates to be static. For static covariates, there should be one entry per line. For example, if the first individual in our ongoing example is female and the second individual is male, the gender file should look like:
```
female
male
```
[Refer here](https://github.com/keyonvafa/career-code/tree/main/sample-data) for an example of each data file.

### <a id="data_location">Data location</a>

For our experiments, we use all covariates except for race/ethnicity and gender for the resume data, and we use all covariates for the survey datasets. The data should be stored in `$RESUME_DATA_DIR` and `$SURVEY_DATA_DIR`. More specifically, replace the ellipses below with the location of the resume and survey datasets: 
```{bash}
RESUME_DATA_DIR=...
SURVEY_DATA_DIR=...
```
In our case, `$RESUME_DATA_DIR` should direct to a folder contain 12 files: `train.job`, `train.year`, `train.education`, `train.location`, and the same four file names but replacing `train` with `valid` and `test`. `$SURVEY_DATA_DIR` should direct to a folder containing 18 files; the same file names as `$RESUME_DATA_DIR`, in addition to `train.ethnicity`, `train.gender`, and repeated for the `valid` and `test` splits ([refer here](https://github.com/keyonvafa/career-code/tree/main/sample-data) for a complete example). The files in `$RESUME_DATA_DIR` and `$SURVEY_DATA_DIR` should contain sequences in the same formats; for example, if resumes are encoded with [O*NET SOC 2019](https://www.onetcenter.org/taxonomy/2019/list.html), so should the survey dataset. 

You will also need to define the following variables to direct to folders where you would like CAREER to save important files:

```{bash}
BINARY_DATA_DIR=...
SAVE_DIR=...
LOG_DIR=...
SURVEY_DATASET_NAME=...
```
Specifically, `$BINARY_DATA_DIR` will contain the binarized data used to train these models; `$SAVE_DIR` will contain the trained model parameters; and `$LOG_DIR` will contain the logs for [tensorboard](https://www.tensorflow.org/tensorboard). A possible default for these folders is `fairseq/data-bin`, `fairseq/checkpoints`, and `fairseq/logs`. Finally `$SURVEY_DATASET_NAME` should contain the name of the survey dataset that `$SURVEY_DATA_DIR` links to. This makes it possible to fine-tune CAREER on multiple survey datasets, like we do in the paper. For example, if you are using NLSY, you can define `SURVEY_DATASET_NAME=nlsy`. If you'd also like to fine-tune the model on PSID, you can redo these steps and set `SURVEY_DATASET_NAME=psid` (and change `$SURVEY_DATA_DIR` to contain the corresponding data).

### Preprocess and binarize datasets
The first step involves preprocessing the data to store them in a format that fairseq understands. Among other things, this creates shared dictionaries so that the encodings for the resume and fine-tuning datasets are consistent. This code may take a while to run, depending on the size of your dataset (it took us 30 minutes for a dataset with 24 million resumes). Run the following from the `fairseq` directory:
```{bash}
sh preprocess/preprocess_transfer_learning_datasets.sh \
  -r $RESUME_DATA_DIR -s $SURVEY_DATA_DIR \
  -b $BINARY_DATA_DIR -n $SURVEY_DATASET_NAME
```

### Pretrain CAREER on resumes
After the data is preprocessed and binarized, it's time to pretrain CAREER on the resumes dataset. You can do this by running the following code in the `fairseq` directory (make sure you run this on a machine with a GPU):
```{bash}
fairseq-train --task occupation_modeling \
  $BINARY_DATA_DIR/resume-pretraining \
  --arch career \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 \
  --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode eos \
  --max-tokens 16000 --update-freq 1 \
  --max-update 50000 --save-interval-updates 1000 \
  --save-dir $SAVE_DIR/resume-pretraining/career \
  --tensorboard-logdir $LOG_DIR/resume-pretraining/career \
  --fp16 --two-stage \
  --include-year --include-education --include-location
```
The number of updates that you need to run will depend on the size of your resume dataset. You can also manually inspect the validation loss and stop running when the validation loss seems to plateau. Here, we train CAREER for 50,000 steps (in the paper, we use 85,000 steps, but results are pretty stable after 30,000 steps or so). You can monitor the training results by checking the logged output or by using tensorboard:
```{bash}
tensorboard --logdir $LOG_DIR/resume-pretraining
```
If you'd like to train without covariates, you can remove the flags `--include-year` or `--include-education` or `--include-location`. 

### Fine-tune CAREER on a survey dataset
After you've pretrained CAREER on resumes, you can now fine-tune on the survey dataset by running the following command (again from the `fairseq` directory):
```{bash}
fairseq-train --task occupation_modeling \
  $BINARY_DATA_DIR/$SURVEY_DATASET_NAME \
  --finetune-from-model $SAVE_DIR/resume-pretraining/career/checkpoint_best.pt \
  --arch career \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 \
  --clip-norm 0.0 \
  --lr 0.0001 --lr-scheduler inverse_sqrt --warmup-updates 500 \
  --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode eos \
  --max-tokens 16000 --update-freq 1 \
  --max-update 5000 \
  --save-dir $SAVE_DIR/$SURVEY_DATASET_NAME/career-transferred \
  --tensorboard-logdir $LOG_DIR/$SURVEY_DATASET_NAME/career-transferred \
  --fp16 --two-stage \
  --include-year --include-education --include-location \
  --include-ethnicity --include-gender \
  --no-epoch-checkpoints 
```
Again, the number of steps to train for depends on the size of the dataset. Since a survey dataset is typically much smaller than a resume dataset, fine-tuning should be much faster than pretraining (fine-tuning on NLSY takes us less than 13 minutes on a single GPU). You can fine-tune even after the model begins to overfit; fairseq will always save the model with the best validation loss. Notice that here we've included covariates that weren't available for the resumes dataset (ethnicity and gender). 

### Evaluate CAREER
After the model has been fine-tuned, you can evaluate the test perplexity with the following command:
```{bash}
cd ../analysis
python compute_survey_data_perplexity.py --model-name career \
  --binary-data-dir $BINARY_DATA_DIR \
  --save-dir $SAVE_DIR \
  --dataset-name $SURVEY_DATASET_NAME
```
This command will print the test set perplexity.

### Train baseline models
We also include code for training two baseline models described in the paper: a bag-of-jobs model, and a multiclass logistic regression with covariates and hand-crafted features. To train the bag of jobs model, run the following command from the `fairseq` directory:
```{bash}
cd ~/career-code/fairseq
fairseq-train --task occupation_modeling \
  $BINARY_DATA_DIR/nlsy \
  --arch bag_of_jobs \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 \
  --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 5000 \
  --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode eos \
  --max-tokens 16000 --update-freq 1 \
  --max-update 10000 \
  --save-dir $SAVE_DIR/nlsy/bag-of-jobs \
  --tensorboard-logdir $LOG_DIR/nlsy/bag-of-jobs \
  --fp16 --two-stage \
  --no-epoch-checkpoints \
  --include-year --include-education --include-location \
  --include-ethnicity --include-gender \
  --embed-dim 1024
```
To fit the regression model, run the following command (also from the `fairseq` directory):
```{bash}
cd ~/career-code/fairseq
fairseq-train --task occupation_modeling \
  $BINARY_DATA_DIR/nlsy \
  --arch regression \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 \
  --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode eos \
  --max-tokens 16000 --update-freq 1 \
  --max-update 25000  \
  --save-dir $SAVE_DIR/nlsy/regression \
  --tensorboard-logdir $LOG_DIR/nlsy/regression \
  --fp16 --two-stage \
  --no-epoch-checkpoints \
  --second-order-markov --include-years-in-current-job \
  --include-total-years --include-year \
  --non-consecutive-year-effect --include-education \
  --education-difference --include-ethnicity \
  --include-gender --include-location 
```
To evaluate these models, you can run the same script as for evaluating CAREER, replacing `--model-name career` with `bag-of-jobs` or `regression`, e.g.
```{bash}
cd ../analysis
python compute_survey_data_perplexity.py --model-name bag-of-jobs \
  --binary-data-dir $BINARY_DATA_DIR \
  --save-dir $SAVE_DIR \
  --dataset-name $SURVEY_DATASET_NAME
```

## Forecasting with CAREER
CAREER can also be used to forecast jobs. Unlike the code above, which uses CAREER to predict jobs, forecasting only trains on job sequences up to a specific year. The trained model is then used to predict future jobs for an individual, by sampling possible career paths. Data should be processed in [the same format as above](#data_formatting) and should follow [the same variable names](#data_location).

### Preprocess and binarize datasets for forecasting
The command below will take raw resume and survey dataset data and remove all jobs from the training and validation sets that occur after the cutoff year (in our code, we use a cutoff year of 2014). It will then binarize them in a format that fairseq understands:
```{bash}
cd ../fairseq
sh preprocess/preprocess_forecasting_datasets.sh \
  -r $RESUME_DATA_DIR -s $SURVEY_DATA_DIR \
  -b $BINARY_DATA_DIR -n $SURVEY_DATASET_NAME
```

### Pretrain CAREER to forecast on resumes
To pretrain CAREER for forecasting, run the following command from the `fairseq` directory:
```{bash}
fairseq-train --task occupation_modeling \
  $BINARY_DATA_DIR/forecast-resume-pretraining \
  --arch career \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 \
  --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode eos \
  --max-tokens 16000 --update-freq 1 \
  --max-update 50000 --save-interval-updates 1000 \
  --save-dir $SAVE_DIR/forecast-resume-pretraining/career \
  --tensorboard-logdir $LOG_DIR/forecast-resume-pretraining/career \
  --fp16 --two-stage \
  --include-year --include-education --include-location
```
This will pretrain CAREER on job sequences without training or validating on any of the examples that occured after the cutoff year.

### Fine-tune CAREER for forecasting on a survey dataset
After CAREER has been pretrained on the resume data, we can fine-tune it to make forecasts on the survey dataset with the following command:
```{bash}
fairseq-train --task occupation_modeling \
  $BINARY_DATA_DIR/forecast-$SURVEY_DATASET_NAME \
  --finetune-from-model $SAVE_DIR/forecast-resume-pretraining/career/checkpoint_best.pt \
  --arch career \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 \
  --clip-norm 0.0 \
  --lr 0.0001 --lr-scheduler inverse_sqrt --warmup-updates 500 \
  --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode eos \
  --max-tokens 16000 --update-freq 1 \
  --max-update 5000 \
  --save-dir $SAVE_DIR/forecast-$SURVEY_DATASET_NAME/career-transferred \
  --tensorboard-logdir $LOG_DIR/forecast-$SURVEY_DATASET_NAME/career-transferred \
  --fp16 --two-stage \
  --include-year --include-education --include-location \
  --include-ethnicity --include-gender \
  --no-epoch-checkpoints 
```

### Evaluate CAREER's forecasts
To evaluate CAREER's forecasts, run the follwing command:
```{bash}
cd ../analysis
python evaluate_survey_data_forecast.py --model-name career \
  --binary-data-dir $BINARY_DATA_DIR \
  --save-dir $SAVE_DIR \
  --dataset-name $SURVEY_DATASET_NAME
```

### Make forecasts with baselines
We also include code for making forecasts with the bag-of-jobs and regression baselines. For bag-of-jobs:
```{bash}
cd ../fairseq
fairseq-train --task occupation_modeling \
  $BINARY_DATA_DIR/forecast-$SURVEY_DATASET_NAME \
  --arch bag_of_jobs \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 \
  --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 5000 \
  --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode eos \
  --max-tokens 16000 --update-freq 1 \
  --max-update 10000 \
  --save-dir $SAVE_DIR/forecast-$SURVEY_DATASET_NAME/bag-of-jobs \
  --tensorboard-logdir $LOG_DIR/forecast-$SURVEY_DATASET_NAME/bag-of-jobs \
  --fp16 --two-stage \
  --no-epoch-checkpoints \
  --include-year --include-education --include-location \
  --include-ethnicity --include-gender \
  --embed-dim 1024
```
For regression:
```{bash}
cd ~/career-code/fairseq
fairseq-train --task occupation_modeling \
  $BINARY_DATA_DIR/forecast-$SURVEY_DATASET_NAME \
  --arch regression \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 \
  --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode eos \
  --max-tokens 16000 --update-freq 1 \
  --max-update 25000  \
  --save-dir $SAVE_DIR/forecast-$SURVEY_DATASET_NAME/regression \
  --tensorboard-logdir $LOG_DIR/forecast-$SURVEY_DATASET_NAME/regression \
  --two-stage \
  --no-epoch-checkpoints \
  --second-order-markov --include-years-in-current-job \
  --include-total-years --include-year \
  --non-consecutive-year-effect --include-education \
  --education-difference --include-ethnicity \
  --include-gender --include-location 
```

## Rationalize CAREER's predictions
You can also use [greedy rationalization](https://arxiv.org/abs/2109.06387) to help interpret CAREER's predictions (refer to [Appendix E in the CAREER paper](https://arxiv.org/abs/2202.08370) for more details). We first need to fine-tune CAREER with word dropout on the survey dataset, to enable the model to predict jobs from incomplete histories. Run the following command from the `fairseq` directory:
```{bash}
fairseq-train --task occupation_modeling \
  $BINARY_DATA_DIR/$SURVEY_DATASET_NAME \
  --finetune-from-model $SAVE_DIR/resume-pretraining/career/checkpoint_best.pt \
  --arch career \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 \
  --clip-norm 0.0 \
  --lr 0.0001 --lr-scheduler inverse_sqrt --warmup-updates 500 \
  --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode eos \
  --max-tokens 16000 --update-freq 1 \
  --max-update 5000 \
  --save-dir $SAVE_DIR/$SURVEY_DATASET_NAME/career-transferred-word-dropout \
  --tensorboard-logdir $LOG_DIR/$SURVEY_DATASET_NAME/career-transferred-word-dropout \
  --fp16 --two-stage \
  --include-year --include-education --include-location \
  --include-ethnicity --include-gender \
  --no-epoch-checkpoints \
  --word-dropout-mixture 0.5
```
Then, to rationalize an example in the test set, run the following code:
```{bash}
cd ../analysis
python rationalize_example.py \
  --binary-data-dir $BINARY_DATA_DIR \
  --save-dir $SAVE_DIR \
  --dataset-name $SURVEY_DATASET_NAME
```
You can rationalize another example in the test set, or input your own example, by modifying the code in `rationalize_example.py`.

## Train CAREER on Resumes Without Fine-Tuning
Finally, you may want to train CAREER to make predictions directly on the resumes dataset, without fine-tuning on survey datasets. Run the following code to train CAREER on resumes without fine-tuning (data should be processed in [the same format as above](#data_formatting) and should follow [the same variable names](#data_location)).

### Preprocess and binarize data
Run the following command from the fairseq directory:
```{bash}
sh preprocess/preprocess_resumes.sh $RESUME_DATA_DIR $BINARY_DATA_DIR
```

### Train CAREER on resumes
```{bash}
fairseq-train --task occupation_modeling \
  $BINARY_DATA_DIR/resumes \
  --arch career \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 \
  --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode eos \
  --max-tokens 16000 --update-freq 1 \
  --max-update 50000 --save-interval-updates 1000 \
  --save-dir $SAVE_DIR/resumes/career \
  --tensorboard-logdir $LOG_DIR/resumes/career \
  --fp16 --two-stage \
  --include-year --include-education --include-location
```

### Evaluate perplexity on resumes
```{bash}
cd ../analysis
python compute_resume_perplexity.py --model-name career \
  --binary-data-dir $BINARY_DATA_DIR \
  --save-dir $SAVE_DIR
```
