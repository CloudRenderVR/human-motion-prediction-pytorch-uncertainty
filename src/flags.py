#These are pretty much entirely for training new networks. The get_both() function only cares about what
#parameters the network was saved with, so shouldn't matter for that end of things.
fk_taylor = False
fk_display_uncertainty = True
fk_use_sampling = False
fk_show_samples = True
fk_show_history = False
fk_show_future = False
fk_show_truth = True

translate_loss_func = "mae"#"mle" #mse, me, mle (max likelihood)
convert_to_euler_first = False
evaluate_do_SMSE = True




