
#!/bin/bash


#---------------------improvement---------------------
experiment_name="improvement"
for version in {0..5} 
do 
    echo "BLEU of $experiment_name v$version is:"
    pred_dir=output/$experiment_name/version_$version
    ans_pth=data/answer.txt # 39923
    pred_pth=$pred_dir/$experiment_name\_v$version\_prediction.txt

    if [[ ! -f $pred_pth ]]; 
    then
        pred_json_pth=$pred_dir/$experiment_name\_v$version\_prediction.jsonl

        if [[ ! -f $pred_json_pth ]];
        then
            echo "no jsonl file is found. Step to next file."
            # exit 1
            continue
        else # make txt file
            python utils/make_pred_txt.py $experiment_name $version
        fi
    fi

    if [[ -f $pred_pth ]]
    then
        # sacrebleu $ans_pth -i $pred_pth -m bleu chrf -tok zh -b -w 4 --chrf-word-order 2
        sacrebleu $ans_pth -i $pred_pth -m bleu -tok zh -b -w 4 
    fi

    echo "----------------------"
done


#---------------------dim---------------------
experiment_name="dim"
for version in {0..4} 
do 
    echo "BLEU of $experiment_name v$version is:"
    pred_dir=output/$experiment_name/version_$version
    ans_pth=data/answer.txt # 39923
    pred_pth=$pred_dir/$experiment_name\_v$version\_prediction.txt

    if [[ ! -f $pred_pth ]]; 
    then
        pred_json_pth=$pred_dir/$experiment_name\_v$version\_prediction.jsonl

        if [[ ! -f $pred_json_pth ]];
        then
            echo "no jsonl file is found. Step to next file."
            # exit 1
            continue
        else # make txt file
            python utils/make_pred_txt.py $experiment_name $version
        fi
    fi

    if [[ -f $pred_pth ]]
    then
        # sacrebleu $ans_pth -i $pred_pth -m bleu chrf -tok zh -b -w 4 --chrf-word-order 2
        sacrebleu $ans_pth -i $pred_pth -m bleu -tok zh -b -w 4 
    fi

    echo "----------------------"
done



#---------------------depth---------------------
experiment_name="depth"
for version in {2..17} 
do 
    echo "BLEU of $experiment_name v$version is:"
    pred_dir=output/$experiment_name/version_$version
    ans_pth=data/answer.txt # 39923
    pred_pth=$pred_dir/$experiment_name\_v$version\_prediction.txt

    if [[ ! -f $pred_pth ]]; 
    then
        pred_json_pth=$pred_dir/$experiment_name\_v$version\_prediction.jsonl

        if [[ ! -f $pred_json_pth ]];
        then
            echo "no jsonl file is found. Step to next file."
            # exit 1
            continue
        else # make txt file
            python utils/make_pred_txt.py $experiment_name $version
        fi
    fi

    if [[ -f $pred_pth ]]
    then
        # sacrebleu $ans_pth -i $pred_pth -m bleu chrf -tok zh -b -w 4 --chrf-word-order 2
        sacrebleu $ans_pth -i $pred_pth -m bleu -tok zh -b -w 4 
    fi

    echo "----------------------"
done



#---------------------llm---------------------
experiment_name="llm"
for version in chimera-inst-chat-13b chimera-inst-chat-7b chimera-chat-13b phoenix-chat-7b chimera-chat-7b phoenix-inst-chat-7b 
do 
    echo "BLEU of $experiment_name v$version is:"
    pred_dir=output/$experiment_name/version_$version
    ans_pth=data/answer.txt # 39923
    pred_pth=$pred_dir/$experiment_name\_v$version\_prediction.txt

    if [[ ! -f $pred_pth ]]; 
    then
        pred_json_pth=$pred_dir/$experiment_name\_v$version\_prediction.jsonl

        if [[ ! -f $pred_json_pth ]];
        then
            echo "no jsonl file is found. Step to next file."
            # exit 1
            continue
        else # make txt file
            python utils/make_pred_txt.py $experiment_name $version
        fi
    fi

    if [[ -f $pred_pth ]]
    then
        # sacrebleu $ans_pth -i $pred_pth -m bleu chrf -tok zh -b -w 4 --chrf-word-order 2
        sacrebleu $ans_pth -i $pred_pth -m bleu -tok zh -b -w 4 
    fi

    echo "----------------------"
done


# sacrebleu data/answer.txt -i "cached/version_3/depth_v3_prediction.txt" -m bleu -tok zh -b -w 4
# sacrebleu ref.detok.txt -i output.detok.txt -m bleu -b -w 4
