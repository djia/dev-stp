from dev_stp import *
from brian2 import *


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

save_file_name = "dev_stp_results_" + str(uuid.uuid4()) + ".json"

params = {
	"sim_time": 100.0 * second,
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
