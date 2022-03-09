# Get passed in arguments
#
# Usage: preprocess_resumes.sh <resume_data_dir> <resume_binary_dir>
#
#
RESUME_DATA_DIR=$1
BINARY_DATA_DIR=$2

echo "Input: resume data from $RESUME_DATA_DIR"
echo "Output: binary resume data to $BINARY_DATA_DIR/resumes"
echo "... "

echo "Preprocessing resumes..."
fairseq-preprocess \
  --only-source \
  --trainpref $RESUME_DATA_DIR/train.job \
  --validpref $RESUME_DATA_DIR/valid.job \
  --testpref $RESUME_DATA_DIR/test.job \
  --destdir $BINARY_DATA_DIR/resumes/job \
  --workers 60
echo "...done."
 
echo "Preprocessing years..."
fairseq-preprocess \
    --only-source \
    --trainpref $RESUME_DATA_DIR/train.year \
    --validpref $RESUME_DATA_DIR/valid.year \
    --testpref $RESUME_DATA_DIR/test.year \
    --destdir $BINARY_DATA_DIR/resumes/year \
    --workers 60
echo "...done."

echo "Preprocessing gender..."
fairseq-preprocess \
    --only-source \
    --trainpref $RESUME_DATA_DIR/train.gender \
    --validpref $RESUME_DATA_DIR/valid.gender \
    --testpref $RESUME_DATA_DIR/test.gender \
    --destdir $BINARY_DATA_DIR/resumes/gender \
    --workers 60
echo "...done."

echo "Preprocessing location..."
fairseq-preprocess \
    --only-source \
    --trainpref $RESUME_DATA_DIR/train.location \
    --validpref $RESUME_DATA_DIR/valid.location \
    --testpref $RESUME_DATA_DIR/test.location \
    --destdir $BINARY_DATA_DIR/resumes/location \
    --workers 60
echo "...done."
