f = open("grid_search_commands_to_source", "w")
lines = []

for arch in ["basic"]:
    for inp_len in ["25", "50"]:
        for out_len in ["10"]:
            for size in ["512", "1024", "2048"]:
                for num_layers in ["1", "2"]:
                    for action in [None, "eating", "purchases", "directions", "walking"]:
                        action_string = ""
                        if action == None:
                            action_string = "--learning_rate 0.0000001 --iterations 15000 --test_every 200"
                        else:
                            action_string = "--learning_rate 0.000001 --iterations 3000 --test_every 100 --action "+action
                        lines.append("python src/translate.py --residual_velocities --omit_one_hot --batch_size 32 --architecture "+arch+" --seq_length_in "+inp_len+" --seq_length_out "+out_len+" "+action_string+" --distribution_output_direct True\n")

f.writelines(lines)
f.close()
