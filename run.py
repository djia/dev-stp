from dev_stp import *
from brian2 import *
from util import *
from superjson import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# define the STP parameters
ex_std_params = {
	"A": [1.0 / 0.3917],
	"taud": [0.3134] * second,
	"tauf": [0.0798] * second,
	"U": [0.3917],
	"f_syn": [0.0625],
}

ex_stf_params = {
	"A": [1.0 / 0.1973],
	"taud": [0.0845] * second,
	"tauf": [0.2959] * second,
	"U": [0.1973],
	"f_syn": [0.1168],
}


n_ex_stp_changes = 3600
stp_change_dt = 0.5 * second

ex_stp_params = {
	"A": [(1.0 / x) for x in (np.logspace(np.log10(ex_std_params["U"][0]), np.log10(ex_stf_params["U"][0]), n_ex_stp_changes))],
	"taud": np.logspace(np.log10(float(ex_std_params["taud"][0])), np.log10(float(ex_stf_params["taud"][0])), n_ex_stp_changes) * second,
	"tauf": np.logspace(np.log10(float(ex_std_params["tauf"][0])), np.log10(float(ex_stf_params["tauf"][0])), n_ex_stp_changes) * second,
	"U": np.logspace(np.log10(ex_std_params["U"][0]), np.log10(ex_stf_params["U"][0]), n_ex_stp_changes),
	"f_syn": np.logspace(np.log10(ex_std_params["f_syn"][0]), np.log10(ex_stf_params["f_syn"][0]), n_ex_stp_changes),
}

in_stp_params = {
	"A": [1.0 / 0.3917],
	"taud": [0.3134] * second,
	"tauf": [0.0798] * second,
	"U": [0.3917],
	"f_syn": [0.0625],
}

isp_params = {
	"isp_eta": [1e-3],
	"isp_gmax": [1e8],
}
isp_target_rate = 5.0


# define the network parameters
n_channels = 8
receptive_field_weights_filename = "./receptive_field_weights/weights.mat"

mat = scipy.io.loadmat(receptive_field_weights_filename)
weights = mat['weights'].flatten().flatten()
ex_to_receiver_weight_by_channel = {}
for i_channel in range(1, n_channels + 1):
	ex_to_receiver_weight_by_channel[i_channel] = weights[i_channel - 1]

# input time varying rates
time_varying_rates_dir = "./time_varying_rates"
rates = {}
for i_channel in xrange(1, n_channels + 1):
	time_varying_rates_filename = time_varying_rates_dir + "/" + "channel" + str(i_channel) + "_rates.mat"
	mat = scipy.io.loadmat(time_varying_rates_filename)
	channel_rates = mat['rates'].flatten()
	rates[i_channel] = channel_rates

# bg gaussian current
gaussian_current_mu = 0.0 * pA
gaussian_current_sigma = 45 * pA
gaussian_current_tau = 15 * ms

# how long to simulate
sim_time = 100.0 * second
# how often to save data for the synapses
monitor_dt = 5e-3
# how often to save data for the receiver neuron
receiver_monitor_dt = 5e-3

save_file_basename = "dev_stp_results_" + str(uuid.uuid4())
save_file_name = save_file_basename + ".json"

params = {
	"sim_time": sim_time,
	"stp_change_dt": stp_change_dt,
	"isp_target_rate": isp_target_rate,
	"is_change_stp_when_balanced": True,
	"n_channels": n_channels,
	"ex_stp_params": ex_stp_params,
	"in_stp_params": in_stp_params,
	"ex_to_receiver_weight_by_channel": ex_to_receiver_weight_by_channel,
	"ex_rates_by_channel": rates,
	"in_rates_by_channel": rates,
	"isp_params": isp_params,
	"monitor_dt" : monitor_dt,
	"receiver_monitor_dt" : receiver_monitor_dt,
	"bg_gaussian_current": True,
	"gaussian_current_mu": gaussian_current_mu,
	"gaussian_current_sigma": gaussian_current_sigma,
	"gaussian_current_tau": gaussian_current_tau,
	"save_file_name": save_file_name,
	"save_ex_rates_by_channel": False,
	"save_in_rates_by_channel": False,
	"save_stp_weights": False,
	"save_stp_component_values": False,
}

simulate_dev_stp(**params)


#####################################
# plot the output and save it to file
# plot the firing rates over time
# plot the inhibitory weights
# plot the rate of change of dev-STP
#####################################

# pick a name for saving the figure
save_figure_name = save_file_basename + ".pdf"

# load data
data = convert_int_str_keys_to_int(json.load(save_file_name))


fig = plt.figure(figsize=(5, 8))

grid = gridspec.GridSpec(3, 1)
grid.update(wspace = 0.7, hspace = 0.2)

sim_time_ceil = np.ceil(float(sim_time))

# plot the average receiver firing rate across time
ax = fig.add_subplot(grid[0, 0])
bin_width = 1.0
avg_rates = calc_avg_rate(data["spike_times"], sim_time_ceil, bin_width = bin_width)
times = np.linspace(bin_width, sim_time_ceil, int(sim_time_ceil / bin_width))
ax.plot(times, avg_rates, "k")
ax.set_xlim([0, sim_time_ceil])
ax.set_ylabel("Receiver firing rate (Hz)")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# plot the average number of STP changes across time
ax = fig.add_subplot(grid[1, 0])
bin_width = 5.0
avg_stp_changes = calc_avg_rate(data["ex_stp_change_times_array"], sim_time_ceil, bin_width = bin_width)
times = np.linspace(bin_width, sim_time_ceil, int(sim_time_ceil / bin_width))
ax.plot(times, avg_stp_changes, "green")
ax.set_xlim([0, sim_time_ceil])
ax.set_ylabel("dev-STP changes / second")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# plot the average ISP weights over time for each channel
ax = fig.add_subplot(grid[2, 0])
bin_width = 1.0
times = np.linspace(bin_width, sim_time_ceil, int(sim_time_ceil / bin_width))
for i_channel in range(1, n_channels + 1):
    w_isps = np.array(data["w_isps_by_channel"][i_channel])
    n_neurons = len(data["w_isps_by_channel"][i_channel])
    # average across all neurons in the group
    avg_w_isps = np.sum(w_isps, axis = 0)
    # average across time (bin_width)
    bin_size = int(bin_width / monitor_dt)
    avg_w_isps = np.sum(np.reshape(avg_w_isps, (-1, bin_size)), axis = 1) / (n_neurons * bin_size)
    ax.plot(times, avg_w_isps)
ax.legend(["Chanel " + str(x) for x in range(1, n_channels + 1)], bbox_to_anchor=(1.0, 1.0), frameon=False)
ax.set_xlim([0, sim_time_ceil])
ax.set_ylabel("$W_{ISP}$ (nS)")
ax.set_xlabel("Time (s)")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# save figure
fig.savefig(save_figure_name, bbox_inches='tight')