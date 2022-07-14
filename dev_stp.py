from brian2 import *
import time
import math
import numpy as np
import os
import scipy
import scipy.io
import scipy.sparse
import uuid
import sys
import gc
import pickle
from superjson import json
import copy
# import random
import subprocess
import io

# ignore warnings about deprecation
import warnings

# IGNORE ALL WARNINGS, uncomment if you want to ignore
# warnings.filterwarnings("ignore")

# runs neural simulation of feed-forward circuit with dev-STP, excitatory and inhibitory STP, inhibitory synaptic plasticity
def simulate_dev_stp(sim_time=1.0 * second, n_ex_sender_neurons_per_channel=100, n_in_sender_neurons_per_channel=25, n_channels=8,
					ex_rates_by_channel={}, in_rates_by_channel={}, ex_rates_dt=0.005, in_rates_dt=0.005, bg_current=0.0,
					monitor_dt=5e-3, receiver_monitor_dt=5e-3,
					isp_target_rate = 5.0,
					ex_stp_params={}, in_stp_params={}, ex_stp_change_times=[], in_stp_change_times=[], stp_change_dt=1.0 * second,
					is_change_stp_when_balanced=True, change_stp_when_balanced_dt=0.5 * second,
					ex_to_receiver_weight_by_channel={}, in_to_receiver_weight=.35,
					isp_params={},
					ex_synaptic_delay = 0*ms, in_synaptic_delay=0*ms,
					bg_gaussian_current=False,
					gaussian_current_mu=0.0 * pA, gaussian_current_sigma=0.0 * pA, gaussian_current_tau=15.0 * ms,
					save_file_name = "dev_stp_results.json",
					save_ex_rates_by_channel=False, save_in_rates_by_channel=False, save_spikes_by_channel=False, save_stp_weights=False, save_stp_component_values=False):
	
	prefs.logging.console_log_level = 'ERROR'

	BrianLogger.suppress_name('clock_driven')
	BrianLogger.suppress_name('resolution_conflict')
	BrianLogger.suppress_name('method_choice')
	BrianLogger.suppress_name('base')

	n_receiver_neurons = 1
	n_neurons = (n_ex_sender_neurons_per_channel + n_in_sender_neurons_per_channel) * 8 + n_receiver_neurons


	# neuron parameters
	tau_stdp = 20*ms    # STDP time constant

	gl = 10.0 * nsiemens   # Leak conductance
	el = -60 * mV          # Resting potential
	er = -80 * mV          # Inhibitory reversal potential
	vt = -50. * mV         # Spiking threshold
	memc = 200.0 * pfarad  # Membrane capacitance
	bg_current = bg_current * pA   # External current

	tau_ampa = 5.0 * ms		# Glutamatergic synaptic time constant
	tau_gaba = 10.0 * ms	# GABAergic synaptic time constant

	#  neuron equation
	if bg_gaussian_current:
		eqs_neurons='''
		dv/dt = (-gl * (v - el) - (g_ampa * v + g_gaba * (v - er)) + (gaussian_current_sigma * xi) / gaussian_current_tau**-0.5 + gaussian_current_mu + bg_current) / memc : volt
		dg_ampa/dt = -g_ampa / tau_ampa : siemens
		dg_gaba/dt = -g_gaba / tau_gaba : siemens
		'''
	else:
		eqs_neurons='''
		dv/dt = (-gl * (v - el) - (g_ampa * v + g_gaba * (v - er)) + bg_current) / memc : volt
		dg_ampa/dt = -g_ampa / tau_ampa : siemens
		dg_gaba/dt = -g_gaba / tau_gaba : siemens
		'''

	for i_channel in xrange(1, n_channels + 1):
		eqs_neurons = eqs_neurons + '''
		dg_ampa''' + str(i_channel) + '''/dt = -g_ampa''' + str(i_channel) + ''' / tau_ampa : siemens
		dg_gaba''' + str(i_channel) + '''/dt = -g_gaba''' + str(i_channel) + ''' / tau_gaba : siemens
		'''

	# connection parameters
	ex_to_receiver_sparseness = 1.0
	in_to_receiver_sparseness = 1.0

	# stdp parameters
	target_rate = isp_target_rate * Hz
	alpha = target_rate * tau_stdp * 2	# Target rate parameter
	if "isp_gmax" in isp_params:
		isp_gmax = isp_params["isp_gmax"][0]
	else:
		isp_gmax = 1e8               		# Maximum inhibitory weight (used to be 100)
	if "isp_gmin" in isp_params:
		isp_gmin = isp_params["isp_gmin"][0]
	else:
		isp_gmin = 0.0
	if "isp_eta" in isp_params:
		isp_eta = isp_params["isp_eta"][0]
	else:
		isp_eta = 1e-3

	eqs_general_ex = '''
	w : 1
	w_esp : 1
	w_stp : 1
	'''

	eqs_general_in = '''
	w : 1
	w_isp : 1
	w_stp : 1
	'''

	##########################
	eqs_stdp_inhib = '''
	isp_eta : 1
	isp_gmax : 1
	isp_gmin : 1
	dA_pre_in/dt = -A_pre_in / tau_stdp : 1 (event-driven)
	dA_post_in/dt = -A_post_in / tau_stdp : 1 (event-driven)
	'''

	eqs_on_pre_stdp_inhib = '''
	A_pre_in += 1.
	w_isp = clip(w_isp + (A_post_in - alpha) * isp_eta * 1.0, isp_gmin, isp_gmax)
	'''

	eqs_on_post_stdp_inhib = '''
	A_post_in += 1.
	w_isp = clip(w_isp + A_pre_in * isp_eta, isp_gmin, isp_gmax)
	'''

	##########################
	# stp parameters
	eqs_stp_ex = '''
	taud : second
	tauf : second
	U : 1
	f_syn : 1
	A : 1

	dR/dt = (1 - R) / taud : 1
	du/dt = (U - u) / tauf : 1
	'''

	eqs_stp_on_pre_ex = '''
	w_stp = A * u * R
	R += -u * R
	u += f_syn * (1 - u)
	'''

	##########################
	eqs_stp_in = '''
	taud : second
	tauf : second
	U : 1
	f_syn : 1
	A : 1

	dR/dt = (1 - R) / taud : 1
	du/dt = (U - u) / tauf : 1
	'''

	eqs_stp_on_pre_in = '''
	w_stp = A * u * R
	R += -u * R
	u += f_syn * (1 - u)
	'''

	##########################

	in_w_isps_by_channel = {}


	n_ex_stp_changes = len(ex_stp_change_times) + 1
	n_in_stp_changes = len(in_stp_change_times) + 1

	# rate is set by the input script

	# receiver neuron group
	receiver_neuron_group = NeuronGroup(n_receiver_neurons, model = eqs_neurons, 
									threshold = "v > vt", reset = "v = el", refractory = 4*ms)
	globals()["receiver_neuron_group"] = receiver_neuron_group

	# neuron groups per channel
	ex_sender_neuron_group_by_channel = {}
	in_sender_neuron_group_by_channel = {}
	ex_timed_array_rates_by_channel = {}
	in_timed_array_rates_by_channel = {}



	# connections
	con_er_by_channel = {}
	con_ir_by_channel = {}

	
	if is_change_stp_when_balanced:
		globals()["isp_balance_time"]= -1
		isp_balance_time = globals()["isp_balance_time"]


	for i_channel in xrange(1, n_channels + 1):
		# calculate the number of rates blocks needed for the entire simulation, add an additional second just for padding

		ex_rates = ex_rates_by_channel[i_channel]
		ex_timed_array_rates = TimedArray(ex_rates, dt = ex_rates_dt * second)
		ex_n_seconds_in_rates = len(ex_rates) * ex_rates_dt

		globals()["ex_timed_array_rates" + str(i_channel)] = ex_timed_array_rates
		globals()["ex_n_seconds_in_rates" + str(i_channel)] = ex_n_seconds_in_rates

		globals()["ex_sender_neuron_group" + str(i_channel)] = NeuronGroup(n_ex_sender_neurons_per_channel, 'rates : Hz', threshold = 'rand() < ex_timed_array_rates' + str(i_channel) + '(((t / second) % ex_n_seconds_in_rates' + str(i_channel) + ') * second) * (dt / second)')
		ex_sender_neuron_group = globals()["ex_sender_neuron_group" + str(i_channel)]
		
		in_rates = in_rates_by_channel[i_channel]
		in_timed_array_rates = TimedArray(in_rates, dt = in_rates_dt * second)
		in_n_seconds_in_rates = len(in_rates) * in_rates_dt

		globals()["in_timed_array_rates" + str(i_channel)] = in_timed_array_rates
		globals()["in_n_seconds_in_rates" + str(i_channel)] = in_n_seconds_in_rates
		
		globals()["in_sender_neuron_group" + str(i_channel)] = NeuronGroup(n_in_sender_neurons_per_channel, 'rates : Hz', threshold = 'rand() < in_timed_array_rates' + str(i_channel) + '(((t / second) % in_n_seconds_in_rates' + str(i_channel) + ') * second) * (dt / second)')
		in_sender_neuron_group = globals()["in_sender_neuron_group" + str(i_channel)]

		ex_sender_neuron_group_by_channel[i_channel] = ex_sender_neuron_group
		in_sender_neuron_group_by_channel[i_channel] = in_sender_neuron_group
		ex_timed_array_rates_by_channel[i_channel] = ex_timed_array_rates
		in_timed_array_rates_by_channel[i_channel] = in_timed_array_rates

		# excitatory connections
		globals()["con_er" + str(i_channel)] = Synapses(ex_sender_neuron_group, receiver_neuron_group, model = eqs_general_ex + eqs_stp_ex,
						on_pre = eqs_stp_on_pre_ex + '''
							w = w_stp * ''' +  str(ex_to_receiver_weight_by_channel[i_channel]) + '''
							g_ampa += w * nS
							g_ampa''' + str(i_channel) + ''' += w * nS
						''',
						delay = ex_synaptic_delay)
		con_er = globals()["con_er" + str(i_channel)]
		con_er.connect("rand() <= ex_to_receiver_sparseness")

		con_er.R = 1.0
		con_er.u = ex_stp_params["U"][0]
		con_er.A = ex_stp_params["A"][0]
		con_er.taud = ex_stp_params["taud"][0]
		con_er.tauf = ex_stp_params["tauf"][0]
		con_er.U = ex_stp_params["U"][0]
		con_er.f_syn = ex_stp_params["f_syn"][0]
		con_er_by_channel[i_channel] = con_er

		# inhibitory connections
		globals()["con_ir" + str(i_channel)] = Synapses(in_sender_neuron_group, receiver_neuron_group, model = eqs_general_in + eqs_stdp_inhib + eqs_stp_in,
						on_pre = eqs_stp_on_pre_in + '''
							w = w_stp * w_isp * ''' +  str(in_to_receiver_weight) + '''
							g_gaba += w * nS
							g_gaba''' + str(i_channel) + ''' += w * nS
						''' + eqs_on_pre_stdp_inhib,
						on_post = eqs_on_post_stdp_inhib,
						delay = in_synaptic_delay)
		con_ir = globals()["con_ir" + str(i_channel)]
		con_ir.connect("rand() <= in_to_receiver_sparseness")

		con_ir.R = 1.0
		con_ir.u = in_stp_params["U"][0]
		con_ir.A = in_stp_params["A"][0]
		con_ir.taud = in_stp_params["taud"][0]
		con_ir.tauf = in_stp_params["tauf"][0]
		con_ir.U = in_stp_params["U"][0]
		con_ir.f_syn = in_stp_params["f_syn"][0]
		if in_w_isps_by_channel:
			con_ir.w_isp = in_w_isps_by_channel[i_channel]
		else:
			con_ir.w_isp = 1e-10
		con_ir.isp_eta = isp_eta
		con_ir.isp_gmax = isp_gmax
		con_ir.isp_gmin = isp_gmin
		con_ir_by_channel[i_channel] = con_ir


	# define network operation to change the stp throughout the simulation
	globals()["next_ex_stp_change_index"] = 0
	globals()["next_in_stp_change_index"] = 0
	globals()["ex_stp_change_times"] = ex_stp_change_times
	globals()["in_stp_change_times"] = in_stp_change_times

	globals()["ex_to_receiver_weight_by_channel"] = ex_to_receiver_weight_by_channel
	globals()["in_to_receiver_weight"] = in_to_receiver_weight
	

	# changing STP when balanced
	globals()["n_times_over_threshold_rate"] = 0
	globals()["ex_to_receiver_weight_by_channel"] = ex_to_receiver_weight_by_channel
	globals()["in_to_receiver_weight"] = in_to_receiver_weight


	threshold_rate = 5.0
	globals()["change_stp_when_balanced_dt"] = change_stp_when_balanced_dt
	globals()["threshold_rate"] = threshold_rate

	if is_change_stp_when_balanced:
		ex_stp_change_times = []
		in_stp_change_times = []
		globals()["ex_stp_change_times"] = ex_stp_change_times
		globals()["in_stp_change_times"] = in_stp_change_times

		@network_operation(dt = change_stp_when_balanced_dt, when = "end")
		def change_stp_when_balanced(t):
			if len(receiver_rate_monitor) == 0:
				return

			global receiver_rate_monitor, change_stp_when_balanced_dt, threshold_rate, n_times_over_threshold_rate
			receiver_firing_rates = receiver_rate_monitor.smooth_rate(window = 'flat', width = change_stp_when_balanced_dt) / Hz
			# print receiver_firing_rates, "\n"
			# print len(receiver_firing_rates), "\n"
			# print change_stp_when_balanced_dt, "\n"
			receiver_firing_rate_times = receiver_rate_monitor.t / second

			start_index = (len(receiver_firing_rates) - int((change_stp_when_balanced_dt / (0.0001 * second))))
			# print starting_index
			last_rate = np.mean(receiver_firing_rates[start_index:])
			print "Average rate for last ", float(change_stp_when_balanced_dt), " second is ", bcolors.BOLD, str(last_rate), "Hz", bcolors.ENDC, "\n"

			if last_rate < threshold_rate:
				if n_times_over_threshold_rate > 0:
					n_times_over_threshold_rate = n_times_over_threshold_rate - 1
			else:
				n_times_over_threshold_rate = n_times_over_threshold_rate + int(round(last_rate / threshold_rate))

			print "n_times_over_threshold_rate is ", bcolors.BOLD, n_times_over_threshold_rate, bcolors.ENDC, "\n"

			if n_times_over_threshold_rate == 0:
				# change STP to next value
				# stp
				global next_ex_stp_change_index, ex_to_receiver_weight_by_channel
				if next_ex_stp_change_index < (len(ex_stp_params["taud"]) - 1):
					print "Changing ex STP, next index", (next_ex_stp_change_index + 2), "out of", len(ex_stp_params["taud"]), "at time", float(t / second), " second\n"
					# change ex stp
					for i_channel in xrange(1, n_channels + 1):
						con_er = con_er_by_channel[i_channel]
						
						con_er.A = ex_stp_params["A"][next_ex_stp_change_index + 1]
						con_er.taud = ex_stp_params["taud"][next_ex_stp_change_index + 1]
						# print  ex_stp_params["taud"][next_ex_stp_change_index + 1]
						# print con_er.A
						con_er.tauf = ex_stp_params["tauf"][next_ex_stp_change_index + 1]
						con_er.U = ex_stp_params["U"][next_ex_stp_change_index + 1]
						con_er.f_syn = ex_stp_params["f_syn"][next_ex_stp_change_index + 1]
						
					next_ex_stp_change_index = next_ex_stp_change_index + 1

					ex_stp_change_times.append(float(t / second))
				else:
					print "Finished changing to final ex STP index\n"

	net = Network(collect())

	# add monitors and run the network
	in_synapses_monitor_by_channel = {}
	ex_synapses_monitor_by_channel = {}
	in_synapses_stp_monitor_by_channel = {}
	ex_synapses_stp_monitor_by_channel = {}
	ex_group_spike_monitor_by_channel = {}
	in_group_spike_monitor_by_channel = {}

	for i_channel in xrange(1, n_channels + 1):
		ex_sender_neuron_group = globals()["ex_sender_neuron_group" + str(i_channel)]
		in_sender_neuron_group = globals()["in_sender_neuron_group" + str(i_channel)]

		con_er = globals()["con_er" + str(i_channel)]
		con_ir = globals()["con_ir" + str(i_channel)]

		globals()["ex_synapses_monitor" + str(i_channel)] = StateMonitor(con_er, ["w", "w_esp"], record = True, dt = monitor_dt * second)
		ex_synapses_monitor = globals()["ex_synapses_monitor" + str(i_channel)]
		ex_synapses_monitor_by_channel[i_channel] = ex_synapses_monitor
		net.add(globals()["ex_synapses_monitor" + str(i_channel)])

		globals()["in_synapses_monitor" + str(i_channel)] = StateMonitor(con_ir, ["w", "w_isp"], record = True, dt = monitor_dt * second)
		in_synapses_monitor = globals()["in_synapses_monitor" + str(i_channel)]
		in_synapses_monitor_by_channel[i_channel] = in_synapses_monitor
		net.add(globals()["in_synapses_monitor" + str(i_channel)])

		globals()["ex_synapses_stp_monitor" + str(i_channel)] = StateMonitor(con_er, ["R", "u", "w_stp"], record = True, dt = monitor_dt * second)
		ex_synapses_stp_monitor = globals()["ex_synapses_stp_monitor" + str(i_channel)]
		ex_synapses_stp_monitor_by_channel[i_channel] = ex_synapses_stp_monitor
		net.add(globals()["ex_synapses_stp_monitor" + str(i_channel)])

		globals()["in_synapses_stp_monitor" + str(i_channel)] = StateMonitor(con_ir, ["R", "u", "w_stp"], record = True, dt = monitor_dt * second)
		in_synapses_stp_monitor = globals()["in_synapses_stp_monitor" + str(i_channel)]
		in_synapses_stp_monitor_by_channel[i_channel] = in_synapses_stp_monitor
		net.add(globals()["in_synapses_stp_monitor" + str(i_channel)])

		if save_spikes_by_channel:
			globals()["ex_group_spike_monitor" + str(i_channel)] = SpikeMonitor(ex_sender_neuron_group)
			ex_group_spike_monitor = globals()["ex_group_spike_monitor" + str(i_channel)]
			ex_group_spike_monitor_by_channel[i_channel] = ex_group_spike_monitor
			net.add(globals()["ex_group_spike_monitor" + str(i_channel)])

			globals()["in_group_spike_monitor" + str(i_channel)] = SpikeMonitor(in_sender_neuron_group)
			in_group_spike_monitor = globals()["in_group_spike_monitor" + str(i_channel)]
			in_group_spike_monitor_by_channel[i_channel] = in_group_spike_monitor
			net.add(globals()["in_group_spike_monitor" + str(i_channel)])


	spike_monitor = SpikeMonitor(receiver_neuron_group)
	net.add(spike_monitor)

	receiver_state_monitor_params = ["v", "g_gaba", "g_ampa"]
	for i_channel in xrange(1, n_channels + 1):
		receiver_state_monitor_params.append("g_gaba" + str(i_channel))
		receiver_state_monitor_params.append("g_ampa" + str(i_channel))
	receiver_state_monitor = StateMonitor(receiver_neuron_group, receiver_state_monitor_params, record = True, dt = receiver_monitor_dt * second)
	net.add(receiver_state_monitor)

	if is_change_stp_when_balanced:
		receiver_rate_monitor = PopulationRateMonitor(receiver_neuron_group)
		net.add(receiver_rate_monitor)
		globals()["receiver_rate_monitor"] = receiver_rate_monitor

	# run the simulation
	net.run(sim_time, report = "text", profile = True)


	###############################
	# post processing after running
	###############################

	# extract the currents
	receiver_ex_conds = receiver_state_monitor[0].g_ampa / siemens
	receiver_in_conds = receiver_state_monitor[0].g_gaba / siemens
	voltages = receiver_state_monitor[0].v / volt
	
	receiver_ex_conds = receiver_ex_conds.tolist()
	receiver_in_conds = receiver_in_conds.tolist()
	voltages = voltages.tolist()

	# extract the spike times
	indices, spike_times_raw = spike_monitor.it
	spike_times = []
	for i in xrange(len(spike_times_raw)):
		spike_times.append(float(spike_times_raw[i]))

	# convert the parameters and change times to matlab readable format
	ex_stp_params_array = {}
	in_stp_params_array = {}
	ex_stp_change_times_array = [float(i) for i in ex_stp_change_times]
	for k, v in ex_stp_params.iteritems():
		if hasattr(v, "__iter__"):
			ex_stp_params_array[k] = [float(i) for i in v]
		else:
			ex_stp_params_array[k] = v
	for k, v in in_stp_params.iteritems():
		if hasattr(v, "__iter__"):
			in_stp_params_array[k] = [float(i) for i in v]
		else:
			in_stp_params_array[k] = v

	# convert the parameters to matlab readable format
	isp_params_array = {}
	for k, v in isp_params.iteritems():
		if hasattr(v, "__iter__"):
			isp_params_array[k] = [float(i) for i in v]
		else:
			isp_params_array[k] = v	
	
	###########################
	# save .json data
	###########################

	json_data = {
		"n_ex_sender_neurons_per_channel": n_ex_sender_neurons_per_channel,
		"n_in_sender_neurons_per_channel": n_in_sender_neurons_per_channel,
		"n_receiver_neurons": n_receiver_neurons,
		"n_neurons": n_neurons,

		"isp_target_rate": isp_target_rate,
		"n_channels": n_channels,
		"monitor_dt": monitor_dt,
		"receiver_monitor_dt": receiver_monitor_dt,
		"ex_rates_dt": ex_rates_dt,
		"in_rates_dt": in_rates_dt,
		"ex_to_receiver_weight_by_channel": ex_to_receiver_weight_by_channel,
		"isp_eta": isp_eta,
		"sim_time": float(sim_time),
		"gl": float(gl),
		"el": float(el),
		"er": float(er),
		"vt": float(vt),
		"isp_alpha": float(alpha),
		"target_rate": float(target_rate),
		"voltages": voltages,

		"ex_synaptic_delay": float(ex_synaptic_delay),
		"in_synaptic_delay": float(in_synaptic_delay),
		"ex_conds": receiver_ex_conds,
		"in_conds": receiver_in_conds,
		"spike_times": spike_times,

		"ex_stp_params_array": ex_stp_params_array,
		"in_stp_params_array": in_stp_params_array,
		"ex_stp_change_times_array": ex_stp_change_times_array,

		"isp_params_array": isp_params_array,

		"n_ex_stp_changes": n_ex_stp_changes,
		"n_in_stp_changes": n_in_stp_changes,

		"bg_gaussian_current": bg_gaussian_current,
		"gaussian_current_mu": float(gaussian_current_mu),
		"gaussian_current_sigma": float(gaussian_current_sigma),
		"gaussian_current_tau": float(gaussian_current_tau),
	}
	
	ex_conds_by_channel_mat = {}
	in_conds_by_channel_mat = {}
	for i_channel in xrange(1, n_channels + 1):
		ex_conds = getattr(receiver_state_monitor[0], "g_ampa" + str(i_channel)) / siemens
		in_conds = getattr(receiver_state_monitor[0], "g_gaba" + str(i_channel)) / siemens
		ex_conds_by_channel_mat[i_channel] = ex_conds.tolist()
		in_conds_by_channel_mat[i_channel] = in_conds.tolist()

	# extract the in synaptic weights
	in_weights_by_channel_mat = {}
	w_isps_by_channel_mat = {}
	for i_channel in xrange(1, n_channels + 1):
		weights = []
		w_isps = []
		for i in xrange(n_in_sender_neurons_per_channel):
			weights.append(in_synapses_monitor_by_channel[i_channel][i].w.tolist())
			w_isps.append(in_synapses_monitor_by_channel[i_channel][i].w_isp.tolist())
		in_weights_by_channel_mat[i_channel] = weights
		w_isps_by_channel_mat[i_channel] = w_isps

	# extract the ex synaptic weights
	ex_weights_by_channel_mat = {}
	w_esps_by_channel_mat = {}
	for i_channel in xrange(1, n_channels + 1):
		weights = []
		w_esps = []
		for i in xrange(n_ex_sender_neurons_per_channel):
			weights.append(ex_synapses_monitor_by_channel[i_channel][i].w.tolist())
			w_esps.append(ex_synapses_monitor_by_channel[i_channel][i].w_esp.tolist())
		ex_weights_by_channel_mat[i_channel] = weights
		w_esps_by_channel_mat[i_channel] = w_esps
	
	# the presynaptic rate by channel
	if save_ex_rates_by_channel:
		ex_rates_by_channel_mat = {}
		for i_channel in xrange(1, n_channels + 1):
			ex_rates_by_channel_mat[i_channel] = ex_rates_by_channel[i_channel]
	
	if save_in_rates_by_channel:
		in_rates_by_channel_mat = {}
		for i_channel in xrange(1, n_channels + 1):
			in_rates_by_channel_mat[i_channel] = in_rates_by_channel[i_channel]
	
	json_data.update({
		"ex_conds_by_channel": ex_conds_by_channel_mat,
		"in_conds_by_channel": in_conds_by_channel_mat,
		
		"in_weights_by_channel": in_weights_by_channel_mat,
		"w_isps_by_channel": w_isps_by_channel_mat,
		"ex_weights_by_channel": ex_weights_by_channel_mat,
		"w_esps_by_channel": w_esps_by_channel_mat,
	})

	if save_ex_rates_by_channel:
		json_data.update({
			"ex_rates_by_channel": ex_rates_by_channel_mat,
		})
	
	if save_in_rates_by_channel:
		json_data.update({
			"in_rates_by_channel": in_rates_by_channel_mat,
		})

	if save_spikes_by_channel:
		ex_spike_time_indices_by_channel_mat = {}
		ex_spike_times_by_channel_mat = {}
		in_spike_time_indices_by_channel_mat = {}
		in_spike_times_by_channel_mat = {}
		for i_channel in xrange(1, n_channels + 1):
			# extract the spike times for the excitatory group
			indices_raw, spike_times_raw = ex_group_spike_monitor_by_channel[i_channel].it

			ex_spike_time_indices = []
			ex_spike_times = []
			for i in xrange(len(indices_raw)):
				ex_spike_time_indices.append(indices_raw[i])
				ex_spike_times.append(float(spike_times_raw[i]))

			# extract the spike times for the inhibitory group
			indices_raw, spike_times_raw = in_group_spike_monitor_by_channel[i_channel].it
			in_spike_time_indices = []
			in_spike_times = []
			for i in xrange(len(indices_raw)):
				in_spike_time_indices.append(indices_raw[i])
				in_spike_times.append(float(spike_times_raw[i]))

			ex_spike_time_indices_by_channel_mat[i_channel] = ex_spike_time_indices
			ex_spike_times_by_channel_mat[i_channel] = ex_spike_times
			in_spike_time_indices_by_channel_mat[i_channel] = in_spike_time_indices
			in_spike_times_by_channel_mat[i_channel] = in_spike_times

		json_data.update({
			"ex_spike_time_indices_by_channel": ex_spike_time_indices_by_channel_mat,
			"ex_spike_times_by_channel": ex_spike_times_by_channel_mat,
			"in_spike_time_indices_by_channel": in_spike_time_indices_by_channel_mat,
			"in_spike_times_by_channel": in_spike_times_by_channel_mat,
		})
		
	if save_stp_weights:
		w_ex_stps_by_channel = {}
		for i_channel in xrange(1, n_channels + 1):
			ex_synapses_stp_monitor = ex_synapses_stp_monitor_by_channel[i_channel]
			w_ex_stps = []
			for i in xrange(n_ex_sender_neurons_per_channel):
				w_ex_stps.append(ex_synapses_stp_monitor[i].w_stp)
			w_ex_stps_by_channel[i_channel] = w_ex_stps
		json_data["w_ex_stps_by_channel"] = w_ex_stps_by_channel
		
		w_in_stps_by_channel = {}
		for i_channel in xrange(1, n_channels + 1):
			in_synapses_stp_monitor = in_synapses_stp_monitor_by_channel[i_channel]
			w_in_stps = []
			for i in xrange(n_in_sender_neurons_per_channel):
				w_in_stps.append(in_synapses_stp_monitor[i].w_stp)
			w_in_stps_by_channel[i_channel] = w_in_stps
		json_data["w_in_stps_by_channel"] = w_in_stps_by_channel

	if save_stp_component_values:
		ex_stp_rs_by_channel_mat = {}
		ex_stp_us_by_channel_mat = {}
		for i_channel in xrange(1, n_channels + 1):
			ex_synapses_stp_monitor = ex_synapses_stp_monitor_by_channel[i_channel]
			ex_stp_rs = []
			ex_stp_us = []
			for i in xrange(n_ex_sender_neurons_per_channel):
				ex_stp_rs.append(ex_synapses_stp_monitor[i].R)
				ex_stp_us.append(ex_synapses_stp_monitor[i].u)
			ex_stp_rs_by_channel_mat[i_channel] = ex_stp_rs
			ex_stp_us_by_channel_mat[i_channel] = ex_stp_us
		json_data["ex_stp_rs_by_channel"] = ex_stp_rs_by_channel_mat
		json_data["ex_stp_us_by_channel"] = ex_stp_us_by_channel_mat

		in_stp_rs_by_channel_mat = {}
		in_stp_us_by_channel_mat = {}
		for i_channel in xrange(1, n_channels + 1):
			in_synapses_stp_monitor = in_synapses_stp_monitor_by_channel[i_channel]
			in_stp_rs = []
			in_stp_us = []
			for i in xrange(n_in_sender_neurons_per_channel):
				in_stp_rs.append(in_synapses_stp_monitor[i].R)
				in_stp_us.append(in_synapses_stp_monitor[i].u)
			in_stp_rs_by_channel_mat[i_channel] = in_stp_rs
			in_stp_us_by_channel_mat[i_channel] = in_stp_us
		json_data["in_stp_rs_by_channel"] = in_stp_rs_by_channel_mat
		json_data["in_stp_us_by_channel"] = in_stp_us_by_channel_mat
	
	if is_change_stp_when_balanced:
		json_data.update({
			"isp_balance_time": isp_balance_time,
		})
	
		
	json.dump(json_data, save_file_name)


# define terminal output colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'