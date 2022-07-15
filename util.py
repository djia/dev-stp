import math
import numpy as np
import scipy.stats

# given an array of spike times, calcualte the average rate across time
def calc_avg_rate(spike_times, sim_time, bin_width=5e-3):
	num_bins = int(sim_time / bin_width)
	spike_bins = [0 for i in range(num_bins)]
	for t in spike_times:
		index = int(math.floor(t / bin_width))
		if index >= len(spike_bins):
			index = len(spike_bins) - 1
		spike_bins[index] = spike_bins[index] + 1
	pop_act = [float(i) / float(bin_width) for i in spike_bins]
	return pop_act


# recursively converts integer keys that are encoded as strings back into actual integers
def convert_int_str_keys_to_int(d):
	if isinstance(d, dict):
		new_dict = {}
		keys = d.keys()
		for k in keys:
			v = d[k]
			if is_str_int(k):
				new_k = int(k)
			else:
				new_k = k

			new_dict[new_k] = convert_int_str_keys_to_int(v)
		return new_dict
	elif isinstance(d, list):
		new_list = []
		for i in range(len(d)):
			v = d[i]
			new_list.append(convert_int_str_keys_to_int(v))
		return new_list
	else:
		return d


# check if a string is an integer
def is_str_int(s):
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()






