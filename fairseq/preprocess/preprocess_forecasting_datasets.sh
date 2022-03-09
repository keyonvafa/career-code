RESUME_DATA_DIR=''
SURVEY_DATA_DIR=''
BINARY_DATA_DIR=''
SURVEY_DATASET_NAME=''

print_usage() {
  printf "Usage: ..."
}

while getopts 'r:s:b:n:' flag; do
  case "${flag}" in
    r) RESUME_DATA_DIR="${OPTARG}" ;;
    s) SURVEY_DATA_DIR="${OPTARG}" ;;
    b) BINARY_DATA_DIR="${OPTARG}" ;;
    n) SURVEY_DATASET_NAME="${OPTARG}" ;;
    *) print_usage
       exit 1 ;;
  esac
done

  echo "Input: "
  echo "   1) Resume data from $RESUME_DATA_DIR for pretraining"
  echo "   2) Survey data from $SURVEY_DATA_DIR for fine-tuning"
  echo "Output: "
  echo "   1) Binary resume data for forecasting to $BINARY_DATA_DIR/forecast-resume-pretraining"
  echo "   2) Binary survey data for forecasting to $BINARY_DATA_DIR/forecast-$SURVEY_DATASET_NAME"
  echo "... "

echo "Thresholding resume data to cutoff year (2014)..."
python preprocess/convert_sequences_for_forecasting.py \
  --data-dir $RESUME_DATA_DIR \
  --dataset-name resumes
echo "...done."

# The test set contains the original test set.
cp $RESUME_DATA_DIR/test.job $RESUME_DATA_DIR/forecast_resumes/test.job   
cp $RESUME_DATA_DIR/test.year $RESUME_DATA_DIR/forecast_resumes/test.year   
cp $RESUME_DATA_DIR/test.education $RESUME_DATA_DIR/forecast_resumes/test.education   
cp $RESUME_DATA_DIR/test.location $RESUME_DATA_DIR/forecast_resumes/test.location

echo "Thresholding survey data to cutoff year (2014)..."
python preprocess/convert_sequences_for_forecasting.py \
  --data-dir $SURVEY_DATA_DIR \
  --dataset-name $SURVEY_DATASET_NAME
echo "...done."

# The test set combines the original train, valid, and test sets.
cat $SURVEY_DATA_DIR/train.job $SURVEY_DATA_DIR/valid.job $SURVEY_DATA_DIR/test.job > $SURVEY_DATA_DIR/forecast_$SURVEY_DATASET_NAME/test.job   
cat $SURVEY_DATA_DIR/train.year $SURVEY_DATA_DIR/valid.year $SURVEY_DATA_DIR/test.year > $SURVEY_DATA_DIR/forecast_$SURVEY_DATASET_NAME/test.year   
cat $SURVEY_DATA_DIR/train.education $SURVEY_DATA_DIR/valid.education $SURVEY_DATA_DIR/test.education > $SURVEY_DATA_DIR/forecast_$SURVEY_DATASET_NAME/test.education
cat $SURVEY_DATA_DIR/train.ethnicity $SURVEY_DATA_DIR/valid.ethnicity $SURVEY_DATA_DIR/test.ethnicity > $SURVEY_DATA_DIR/forecast_$SURVEY_DATASET_NAME/test.ethnicity
cat $SURVEY_DATA_DIR/train.gender $SURVEY_DATA_DIR/valid.gender $SURVEY_DATA_DIR/test.gender > $SURVEY_DATA_DIR/forecast_$SURVEY_DATASET_NAME/test.gender
cat $SURVEY_DATA_DIR/train.location $SURVEY_DATA_DIR/valid.location $SURVEY_DATA_DIR/test.location > $SURVEY_DATA_DIR/forecast_$SURVEY_DATASET_NAME/test.location

echo "Binarizing data..."
sh preprocess/preprocess_transfer_learning_datasets.sh \
  -r $RESUME_DATA_DIR/forecast_resumes \
  -s $SURVEY_DATA_DIR/forecast_$SURVEY_DATASET_NAME \
  -b $BINARY_DATA_DIR -n $SURVEY_DATASET_NAME \
  -f