f = open("grid_search_commands_to_source", "w")
lines = []

for arch in ["tied", "basic"]:
    for inp_len in ["25", "50"]:
        for out_len in ["10", "5"]:
            for action in [None, "eating", "purchases"]:
                action_string = ""
                if action == None:
                    action_string = "--learning_rate 0.005 --iterations 30000"
                else:
                    action_string = "--learning_rate 0.05 --iterations 10000 --action "+action
                lines.append("python src/translate.py --residual_velocities --omit_one_hot --batch_size 32 --architecture "+arch+" --seq_length_in "+inp_len+" --seq_length_out "+out_len+" "+action_string)

f.writelines(lines)
f.close()