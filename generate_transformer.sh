
# arg format: name version best_or_last batch_size
bs=100 # using V100

# using the last or best checkpoint
# best_or_last="last"
best_or_last="best"

for v in {2..17} 
do
    python generate_transformer.py depth $v $best_or_last $bs
    echo "-------------------version $v done-------------------------"
done