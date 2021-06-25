f = open("grid_search_commands_to_source", "w")
lines = []

for arch in ["basic"]:
    for inp_len in ["25"]:
        for out_len in ["14"]:
            for size in ["2048"]:
                for num_layers in ["1"]:
                    for action in ["walking", "eating", "smoking", "discussion", "waiting", "purchases"]:
                        action_string = ""
                        if action == None:
                            action_string = "--learning_rate 0.00000005 --iterations 100000 --test_every 2000"
                        else:
                            action_string = "--learning_rate 0.0000007 --iterations 3500 --test_every 100 --action "+action
                        lines.append("python src/translate.py --residual_velocities --omit_one_hot --batch_size 32 --size "+size+" --num_layers "+num_layers+" --architecture "+arch+" --seq_length_in "+inp_len+" --seq_length_out "+out_len+" "+action_string+" --distribution_output_direct True\n")

f.writelines(lines)
f.close()
