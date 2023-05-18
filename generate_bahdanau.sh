
# arg format: name version best_or_last batch_size
bs=100 # using V100

# using the last or best checkpoint
# best_or_last="last"
best_or_last="best"

# generate improvement
for v in {0..5} 
do
    python generate_bahdanau.py improvement $v $best_or_last $bs
    echo "-------------------version $v done-------------------------"
done

# generate dime
for v in {0..4} 
do
    python generate_bahdanau.py dim $v $best_or_last $bs
    echo "-------------------version $v done-------------------------"
done