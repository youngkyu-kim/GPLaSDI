import yaml
import os
template_file = 'burgers1d-flasdi.yml'

with open(template_file, 'r') as file:
    template = yaml.safe_load(file)

# Modify variables lasdi: gplasdi: max_iter and latent space: ae: hidden_units:
# max_iters = [5000, 10000, 15000, 20000]
max_iters = [2000, 5000, 10000]
pointwise_lifts = [5]
layer_widths = [25, 50]
for max_iter in max_iters:
    for pointwise_lift in pointwise_lifts:
        for layer_width in layer_widths:
            new_template = template.copy()
            new_template['lasdi']['gplasdi']['max_iter'] = max_iter
            new_template['lasdi']['gplasdi']['n_iter'] = max_iter
            new_template['latent_space']['fae1d']['pointwise_lift_dim'] = pointwise_lift
            new_template['latent_space']['fae1d']['layer_widths'] = [layer_width]
            exp_key = f"burgers1d-FAE-MI{max_iter}-PLD{pointwise_lift}-LW{layer_width}"
            new_template['exp_key'] = exp_key
            new_template['lasdi']['gplasdi']['path_checkpoint'] = f"{exp_key}/checkpoint"
            new_template['lasdi']['gplasdi']['results'] = f"{exp_key}"
            folder_name = exp_key
            # Create folder if it does not exist
            os.makedirs(folder_name, exist_ok=True)
            os.makedirs(folder_name + '/checkpoint/', exist_ok=True)
            with open(folder_name + '/config.yaml', 'w') as outfile:
                yaml.dump(new_template, outfile, default_flow_style=False)
