RESUME_DATA_DIR=''
SURVEY_DATA_DIR=''
BINARY_DATA_DIR=''
SURVEY_DATASET_NAME=''
FORECAST=false

print_usage() {
  printf "Usage: ..."
}

while getopts 'r:s:b:n:f' flag; do
  case "${flag}" in
    r) RESUME_DATA_DIR="${OPTARG}" ;;
    s) SURVEY_DATA_DIR="${OPTARG}" ;;
    b) BINARY_DATA_DIR="${OPTARG}" ;;
    n) SURVEY_DATASET_NAME="${OPTARG}" ;;
    f) FORECAST=true ;;
    *) print_usage
       exit 1 ;;
  esac
done


if $FORECAST; then
  RESUME_SUFFIX="forecast-resume-pretraining"
  SURVEY_SUFFIX="forecast-$SURVEY_DATASET_NAME"
else
  RESUME_SUFFIX="resume-pretraining"
  SURVEY_SUFFIX="$SURVEY_DATASET_NAME"
  echo "Input: "
  echo "   1) Resume data from $RESUME_DATA_DIR for pretraining"
  echo "   2) Survey data from $SURVEY_DATA_DIR for fine-tuning"
  echo "Output: "
  echo "   1) Binary resume data to $BINARY_DATA_DIR/resume-pretraining"
  echo "   2) Binary survey data to $BINARY_DATA_DIR/$SURVEY_DATASET_NAME"
  echo "... "
fi

echo "Combining resume and survey data..."
cat $RESUME_DATA_DIR/train.job $RESUME_DATA_DIR/valid.job \
    $RESUME_DATA_DIR/test.job $SURVEY_DATA_DIR/train.job \
    $SURVEY_DATA_DIR/valid.job $SURVEY_DATA_DIR/test.job \
    > $RESUME_DATA_DIR/all.job

cat $RESUME_DATA_DIR/train.year $RESUME_DATA_DIR/valid.year \
    $RESUME_DATA_DIR/test.year $SURVEY_DATA_DIR/train.year \
    $SURVEY_DATA_DIR/valid.year $SURVEY_DATA_DIR/test.year \
    > $RESUME_DATA_DIR/all.year

cat $RESUME_DATA_DIR/train.education $RESUME_DATA_DIR/valid.education \
    $RESUME_DATA_DIR/test.education $SURVEY_DATA_DIR/train.education \
    $SURVEY_DATA_DIR/valid.education $SURVEY_DATA_DIR/test.education \
    > $RESUME_DATA_DIR/all.education

cat $RESUME_DATA_DIR/train.location $RESUME_DATA_DIR/valid.location \
    $RESUME_DATA_DIR/test.location $SURVEY_DATA_DIR/train.location \
    $SURVEY_DATA_DIR/valid.location $SURVEY_DATA_DIR/test.location \
    > $RESUME_DATA_DIR/all.location
echo "...done."

## Create dictionaries from combined data.
echo "Creating dictionary for jobs..."
fairseq-preprocess \
    --only-source \
    --trainpref $RESUME_DATA_DIR/all.job \
    --validpref $RESUME_DATA_DIR/valid.job \
    --testpref $RESUME_DATA_DIR/test.job \
    --destdir $BINARY_DATA_DIR/$RESUME_SUFFIX/job \
    --dict-only \
    --workers 60
echo "...done."

echo "Creating dictionary for years..."
fairseq-preprocess \
    --only-source \
    --trainpref $RESUME_DATA_DIR/all.year \
    --validpref $RESUME_DATA_DIR/valid.year \
    --testpref $RESUME_DATA_DIR/test.year \
    --destdir $BINARY_DATA_DIR/$RESUME_SUFFIX/year \
    --dict-only \
    --workers 60
echo "...done."

echo "Creating dictionary for educations..."
fairseq-preprocess \
    --only-source \
    --trainpref $RESUME_DATA_DIR/all.education \
    --validpref $RESUME_DATA_DIR/valid.education \
    --testpref $RESUME_DATA_DIR/test.education \
    --destdir $BINARY_DATA_DIR/$RESUME_SUFFIX/education \
    --dict-only \
    --workers 60
echo "...done."

echo "Creating dictionary for locations..."
fairseq-preprocess \
    --only-source \
    --trainpref $RESUME_DATA_DIR/all.location \
    --validpref $RESUME_DATA_DIR/valid.location \
    --testpref $RESUME_DATA_DIR/test.location \
    --destdir $BINARY_DATA_DIR/$RESUME_SUFFIX/location \
    --dict-only \
    --workers 60
echo "...done."

## Modify year dictionary so years are ordered (this is helpful for
## forecasting).
python preprocess/create_ordered_year_dictionary.py \
    --data-dir $BINARY_DATA_DIR/$RESUME_SUFFIX/year

## Preprocess resume data using the combined dictionaries.
echo "Preprocessing resume data (jobs)..."
fairseq-preprocess \
    --only-source \
    --trainpref $RESUME_DATA_DIR/train.job \
    --validpref $RESUME_DATA_DIR/valid.job \
    --testpref $RESUME_DATA_DIR/test.job \
    --destdir $BINARY_DATA_DIR/$RESUME_SUFFIX/job \
    --srcdict $BINARY_DATA_DIR/$RESUME_SUFFIX/job/dict.txt \
    --workers 60
echo "...done."

echo "Preprocessing resume data (years)..."
fairseq-preprocess \
    --only-source \
    --trainpref $RESUME_DATA_DIR/train.year \
    --validpref $RESUME_DATA_DIR/valid.year \
    --testpref $RESUME_DATA_DIR/test.year \
    --destdir $BINARY_DATA_DIR/$RESUME_SUFFIX/year \
    --srcdict $BINARY_DATA_DIR/$RESUME_SUFFIX/year/dict.txt \
    --workers 60
echo "...done."

echo "Preprocessing resume data (educations)..."
fairseq-preprocess \
    --only-source \
    --trainpref $RESUME_DATA_DIR/train.education \
    --validpref $RESUME_DATA_DIR/valid.education \
    --testpref $RESUME_DATA_DIR/test.education \
    --destdir $BINARY_DATA_DIR/$RESUME_SUFFIX/education \
    --srcdict $BINARY_DATA_DIR/$RESUME_SUFFIX/education/dict.txt \
    --workers 60
echo "...done."

echo "Preprocessing resume data (locations)..."
fairseq-preprocess \
    --only-source \
    --trainpref $RESUME_DATA_DIR/train.location \
    --validpref $RESUME_DATA_DIR/valid.location \
    --testpref $RESUME_DATA_DIR/test.location \
    --destdir $BINARY_DATA_DIR/$RESUME_SUFFIX/location \
    --srcdict $BINARY_DATA_DIR/$RESUME_SUFFIX/location/dict.txt \
    --workers 60
echo "...done."

## Now repeat this preprocessing for the survey dataset
echo "Preprocessing survey data (jobs)..."
fairseq-preprocess \
    --only-source \
    --trainpref $SURVEY_DATA_DIR/train.job \
    --validpref $SURVEY_DATA_DIR/valid.job \
    --testpref $SURVEY_DATA_DIR/test.job \
    --destdir $BINARY_DATA_DIR/$SURVEY_SUFFIX/job \
    --srcdict $BINARY_DATA_DIR/$RESUME_SUFFIX/job/dict.txt \
    --workers 60
echo "...done."

echo "Preprocessing survey data (years)..."
fairseq-preprocess \
    --only-source \
    --trainpref $SURVEY_DATA_DIR/train.year \
    --validpref $SURVEY_DATA_DIR/valid.year \
    --testpref $SURVEY_DATA_DIR/test.year \
    --destdir $BINARY_DATA_DIR/$SURVEY_SUFFIX/year \
    --srcdict $BINARY_DATA_DIR/$RESUME_SUFFIX/year/dict.txt \
    --workers 60
echo "...done."

echo "Preprocessing survey data (educations)..."
fairseq-preprocess \
    --only-source \
    --trainpref $SURVEY_DATA_DIR/train.education \
    --validpref $SURVEY_DATA_DIR/valid.education \
    --testpref $SURVEY_DATA_DIR/test.education \
    --destdir $BINARY_DATA_DIR/$SURVEY_SUFFIX/education \
    --srcdict $BINARY_DATA_DIR/$RESUME_SUFFIX/education/dict.txt \
    --workers 60
echo "...done."

echo "Preprocessing survey data (locations)..."
fairseq-preprocess \
    --only-source \
    --trainpref $SURVEY_DATA_DIR/train.location \
    --validpref $SURVEY_DATA_DIR/valid.location \
    --testpref $SURVEY_DATA_DIR/test.location \
    --destdir $BINARY_DATA_DIR/$SURVEY_SUFFIX/location \
    --srcdict $BINARY_DATA_DIR/$RESUME_SUFFIX/location/dict.txt \
    --workers 60
echo "...done."

## The last two covariates -- race and ethnicity -- aren't used for the resume
## data, so we create the dictionary specifically for the survey data.
echo "Preprocessing survey data (race/ethnicities)..."
fairseq-preprocess \
    --only-source \
    --trainpref $SURVEY_DATA_DIR/train.ethnicity \
    --validpref $SURVEY_DATA_DIR/valid.ethnicity \
    --testpref $SURVEY_DATA_DIR/test.ethnicity \
    --destdir $BINARY_DATA_DIR/$SURVEY_SUFFIX/ethnicity \
    --workers 60
echo "...done."

echo "Preprocessing survey data (genders)..."
fairseq-preprocess \
    --only-source \
    --trainpref $SURVEY_DATA_DIR/train.gender \
    --validpref $SURVEY_DATA_DIR/valid.gender \
    --testpref $SURVEY_DATA_DIR/test.gender \
    --destdir $BINARY_DATA_DIR/$SURVEY_SUFFIX/gender \
    --workers 60
echo "...done."

## Finally, copy the dictionaries created for ethnicity and gender to the
## resume pretraining directory, since, although the model does not pretrain
## on these variables, for software reasons, it needs to create the embedding
## parameters before fine-tuning.
mkdir $BINARY_DATA_DIR/$RESUME_SUFFIX/ethnicity
mkdir $BINARY_DATA_DIR/$RESUME_SUFFIX/gender
cp $BINARY_DATA_DIR/$SURVEY_SUFFIX/ethnicity/dict.txt \
   $BINARY_DATA_DIR/$RESUME_SUFFIX/ethnicity/dict.txt
cp $BINARY_DATA_DIR/$SURVEY_SUFFIX/gender/dict.txt \
   $BINARY_DATA_DIR/$RESUME_SUFFIX/gender/dict.txt