using DiffEqFlux, Flux, Optim, DifferentialEquations, PyPlot, LinearAlgebra, OrdinaryDiffEq, DelimitedFiles
using DataInterpolations
using BSON: @save, @load
using FFTW

function tbnn(σ,γd,model_weights_1,model_weights_2)
    # Tensor basis neural network (TBNN)
    # Unpack the inputs
    σ11,σ22,σ33,σ12,σ13,σ23 = σ
    γd11,γd22,γd33,γd12,γd13,γd23 = γd

    # Compute elements of the tensor basis
    # T1 = I, T2 = σ, T3 = γd
    # T4 = σ⋅σ
    T4_11 = σ11^2 + σ12^2 + σ13^2
    T4_22 = σ12^2 + σ22^2 + σ23^2
    T4_33 = σ13^2 + σ23^2 + σ33^2
    T4_12 = σ11*σ12 + σ12*σ22 + σ13*σ23
    T4_13 = σ11*σ13 + σ12*σ23 + σ13*σ33
    T4_23 = σ12*σ13 + σ22*σ23 + σ23*σ33

    # T5 = γd⋅γd
    T5_11 = γd11^2 + γd12^2 + γd13^2
    T5_22 = γd12^2 + γd22^2 + γd23^2
    T5_33 = γd13^2 + γd23^2 + γd33^2
    T5_12 = γd11*γd12 + γd12*γd22 + γd13*γd23
    T5_13 = γd11*γd13 + γd12*γd23 + γd13*γd33
    T5_23 = γd12*γd13 + γd22*γd23 + γd23*γd33

    # T6 = σ⋅γd + γd⋅σ
    T6_11 = 2*(σ11*γd11 + σ12*γd12 + σ13*γd13)
    T6_22 = 2*(σ12*γd12 + σ22*γd22 + σ23*γd23)
    T6_33 = 2*(σ13*γd13 + σ23*γd23 + σ33*γd33)
    T6_12 = σ11*γd12 + σ12*γd22 + σ13*γd23 + γd11*σ12 + γd12*σ22 + γd13*σ23
    T6_13 = σ11*γd13 + σ12*γd23 + σ13*γd33 + γd11*σ13 + γd12*σ23 + γd13*σ33
    T6_23 = σ12*γd13 + σ22*γd23 + σ23*γd33 + γd12*σ13 + γd22*σ23 + γd23*σ33

    # T7 = σ⋅σ⋅γd + γd⋅σ⋅σ
    T7_11 = 2*(T4_11*γd11 + T4_12*γd12 + T4_13*γd13)
    T7_22 = 2*(T4_12*γd12 + T4_22*γd22 + T4_23*γd23)
    T7_33 = 2*(T4_13*γd13 + T4_23*γd23 + T4_33*γd33)
    T7_12 = T4_11*γd12 + T4_12*γd22 + T4_13*γd23 + γd11*T4_12 + γd12*T4_22 + γd13*T4_23
    T7_13 = T4_11*γd13 + T4_12*γd23 + T4_13*γd33 + γd11*T4_13 + γd12*T4_23 + γd13*T4_33
    T7_23 = T4_12*γd13 + T4_22*γd23 + T4_23*γd33 + γd12*T4_13 + γd22*T4_23 + γd23*T4_33

    # T8 = σ⋅γd⋅γd + γd⋅γd⋅σ
    T8_11 = 2*(σ11*T5_11 + σ12*T5_12 + σ13*T5_13)
    T8_22 = 2*(σ12*T5_12 + σ22*T5_22 + σ23*T5_23)
    T8_33 = 2*(σ13*T5_13 + σ23*T5_23 + σ33*T5_33)
    T8_12 = σ11*T5_12 + σ12*T5_22 + σ13*T5_23 + T5_11*σ12 + T5_12*σ22 + T5_13*σ23
    T8_13 = σ11*T5_13 + σ12*T5_23 + σ13*T5_33 + T5_11*σ13 + T5_12*σ23 + T5_13*σ33
    T8_23 = σ12*T5_13 + σ22*T5_23 + σ23*T5_33 + T5_12*σ13 + T5_22*σ23 + T5_23*σ33

    # T9 = σ⋅σ⋅γd⋅γd + γd⋅γd⋅σ⋅σ
    T9_11 = 2*(T4_11*T5_11 + T4_12*T5_12 + T4_13*T5_13)
    T9_22 = 2*(T4_12*T5_12 + T4_22*T5_22 + T4_23*T5_23)
    T9_33 = 2*(T4_13*T5_13 + T4_23*T5_23 + T4_33*T5_33)
    T9_12 = T4_11*T5_12 + T4_12*T5_22 + T4_13*T5_23 + T5_11*T4_12 + T5_12*T4_22 + T5_13*T4_23
    T9_13 = T4_11*T5_13 + T4_12*T5_23 + T4_13*T5_33 + T5_11*T4_13 + T5_12*T4_23 + T5_13*T4_33
    T9_23 = T4_12*T5_13 + T4_22*T5_23 + T4_23*T5_33 + T5_12*T4_13 + T5_22*T4_23 + T5_23*T4_33

    # Compute the integrity basis from scalar invariants
    # λ1 = tr(σ)
    λ1 = σ11 + σ22 + σ33

    # λ2 = tr(σ^2)
    λ2 = T4_11 + T4_22 + T4_33

    # λ3 = tr(γd^2)
    λ3 = T5_11 + T5_22 + T5_33

    # λ4 = tr(σ^3)
    λ4 = σ11*T4_11 + σ22*T4_22 + σ33*T4_33 + 2*(σ12*T4_12 + σ13*T4_13 + σ23*T4_23)

    # λ5 = tr(γd^3)
    λ5 = γd11*T5_11 + γd22*T5_22 + γd33*T5_33 + 2*(γd12*T5_12 + γd13*T5_13 + γd23*T5_23)

    # λ6 = tr(σ^2⋅γd^2)
    λ6 = T4_11*T5_11 + T4_22*T5_22 + T4_33*T5_33 + 2*(T4_12*T5_12 + T4_13*T5_13 + T4_23*T5_23)

    # λ7 = tr(σ^2⋅γd)
    λ7 = (T7_11 + T7_22 + T7_33)/2

    # λ8 = tr(σ⋅γd^2)
    λ8 = (T8_11 + T8_22 + T8_33)/2

    # λ9 = tr(σ⋅γd)
    λ9 = (T6_11 + T6_22 + T6_33)/2

    # Run the integrity basis through a neural network
    model_inputs = [λ1;λ2;λ3;λ4;λ5;λ6;λ7;λ8;λ9]
    g1,g2,g3 = model_univ_1(model_inputs, model_weights_1)
    g4,g5,g6,g7,g8,g9 = model_univ_2(model_inputs, model_weights_2)
    
    # Tensor combining layer
    F11 = g1 + g2*σ11 + g3*γd11 + g4*T4_11 + g5*T5_11 + g6*T6_11 + g7*T7_11 + g8*T8_11 + g9*T9_11
    F22 = g1 + g2*σ22 + g3*γd22 + g4*T4_22 + g5*T5_22 + g6*T6_22 + g7*T7_22 + g8*T8_22 + g9*T9_22
    F33 = g1 + g2*σ33 + g3*γd33 + g4*T4_33 + g5*T5_33 + g6*T6_33 + g7*T7_33 + g8*T8_33 + g9*T9_33
    F12 = g2*σ12 + g3*γd12 + g4*T4_12 + g5*T5_12 + g6*T6_12 + g7*T7_12 + g8*T8_12 + g9*T9_12
    F13 = g2*σ13 + g3*γd13 + g4*T4_13 + g5*T5_13 + g6*T6_13 + g7*T7_13 + g8*T8_13 + g9*T9_13
    F23 = g2*σ23 + g3*γd23 + g4*T4_23 + g5*T5_23 + g6*T6_23 + g7*T7_23 + g8*T8_23 + g9*T9_23

    return F11,F22,F33,F12,F13,F23
end

function dudt_univ!(du, u, p, t, gradv)
    # Destructure the parameters
    model_weights_1 = p[1:n_weights_1]
    model_weights_2 = p[n_weights_1+1:n_weights_1+n_weights_2]
    η0 = p[end - 1]
    τ = p[end]

    # Governing equations are for components of the stress tensor
    σ11,σ22,σ33,σ12,σ13,σ23 = u

    # Specify the velocity gradient tensor
    v11,v12,v13,v21,v22,v23,v31,v32,v33 = gradv

    # Compute the rate-of-strain (symmetric) and vorticity (antisymmetric) tensors
    γd11 = 2*v11(t)
    γd22 = 2*v22(t)
    γd33 = 2*v33(t) 
    γd12 = v12(t) + v21(t)
    γd13 = v13(t) + v31(t)
    γd23 = v23(t) + v32(t)

    # Run stress/strain through a TBNN
    γd = [γd11,γd22,γd33,γd12,γd13,γd23]
    F11,F22,F33,F12,F13,F23 = tbnn(u,γd,model_weights_1,model_weights_2)

    # The model differential equations
    dσ11 = η0*γd11/τ - σ11/τ + 2*v11(t)*σ11 + v21(t)*σ12 + v31(t)*σ13 + σ12*v21(t) + σ13*v31(t) - F11/τ
    dσ22 = η0*γd22/τ - σ22/τ + 2*v22(t)*σ22 + v12(t)*σ12 + v32(t)*σ23 + σ12*v12(t) + σ23*v32(t) - F22/τ
    dσ33 = η0*γd33/τ - σ33/τ + 2*v33(t)*σ33 + v13(t)*σ13 + v23(t)*σ23 + σ13*v13(t) + σ23*v23(t) - F33/τ
    dσ12 = η0*γd12/τ - σ12/τ + v11(t)*σ12 + v21(t)*σ22 + v31(t)*σ23 + σ11*v12(t) + σ12*v22(t) + σ13*v32(t) - F12/τ
    dσ13 = η0*γd13/τ - σ13/τ + v11(t)*σ13 + v21(t)*σ23 + v31(t)*σ33 + σ11*v13(t) + σ12*v23(t) + σ13*v33(t) - F13/τ
    dσ23 = η0*γd23/τ - σ23/τ + v12(t)*σ13 + v22(t)*σ23 + v32(t)*σ33 + σ12*v13(t) + σ22*v23(t) + σ23*v33(t) - F23/τ

    # Update in place
    du[1] = dσ11
    du[2] = dσ22
    du[3] = dσ33
    du[4] = dσ12
    du[5] = dσ13
    du[6] = dσ23
end

function ensemble_solve(θ,η0s,τs,ensemble,protocols,tspans,σ0,saveat)
	# Define the (default) ODEProblem
	dudt_protocol!(du,u,p,t) = dudt_univ!(du,u,p,t,protocols[1])
	prob = ODEProblem(dudt_protocol!, σ0, tspans[1], [θ; η0s[1]; τs[1]])
	n_protocols = length(protocols)
	n_modes =  length(η0s)

	# Remake the problem for different protocols
	function prob_func(prob, i, repeat)
		j = cld(i,n_modes)
		k = mod(i,n_modes) + 1
		dudt_remade!(du,u,p,t) = dudt_univ!(du,u,p,t,protocols[j])
		remake(prob, f=dudt_remade!, tspan=tspans[j], p=[θ; η0s[k]; τs[k]])
	end

	ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
	sim = solve(ensemble_prob, Tsit5(), ensemble, trajectories=n_protocols*n_modes, saveat=saveat, dtmin=1E-4)
end

function loss_univ(θ,p_system,n_modes,protocols,tspans,σ0,σ12_all,saveat)
	# Compute the multimode solution
	loss = 0
	trajectories = length(protocols)
	η0s = p_system[1:n_modes]
	τs = p_system[n_modes+1:end]
	results = ensemble_solve(θ, η0s, τs, EnsembleThreads(), protocols, tspans, σ0, saveat)
	
	# Iterate over all flow protocols
	for k = range(1,trajectories,step=1)
		σ12_pred = Zygote.Buffer(σ12_all[k], length(σ12_all[k]))
		σ12_pred[:] = zeros(length(σ12_all[k]))

		# Iterate over all modes
		for n = range(1,n_modes,step=1)
			if size(results[(k-1)*n_modes + n][4,2:end]) == size(σ12_all[k])
				σ12_pred[:] += results[(k-1)*n_modes + n][4,2:end]
			else
				loss += Inf
			end
		end

		# Add to loss
		σ12_pred = copy(σ12_pred)
		loss += sum(abs2, σ12_all[k] - σ12_pred)/sum(abs2, σ12_all[k])
	end

	return loss
end

function solve_ude_mm(protocol, tspan, θ, p, n_modes, saveat)
	# Solve the multimode Giesekus model for a particular protocol
	# Get parameters
	η0s = p[1:n_modes]
	τs = p[n_modes+1:end]

	# Define problem
	dudt!(du,u,p,t) = dudt_univ!(du,u,p,t,protocol)
	σ_data = zeros(6,Int((tspan[2] - tspan[1])/saveat) + 1)
	t = LinRange(tspan[1],tspan[2],Int((tspan[2] - tspan[1])/saveat) + 1)

	# Add solution for each mode
	for n = range(1,length(η0),step=1)
		prob_mode = ODEProblem(dudt!, σ0, tspan, [θ; η0s[n]; τs[n]])
		sol_mode = solve(prob_mode, Rodas4(), saveat=saveat)
		σ_data += sol_mode[:,:]
	end
	return (t, σ_data)
end

# Define the simple shear deformation protocol
v11(t) = 0
v12(t) = 0
v13(t) = 0
v22(t) = 0
v23(t) = 0
v31(t) = 0
v32(t) = 0
v33(t) = 0

# Iniitial conditions
σ0 = [0f0,0f0,0f0,0f0,0f0,0f0]

# Solve for the multimode Giesekus model
Gm = 37585f0
τm = 0.557f0
σ12_all = Any[]
γ12_all = Any[]
N1_all = Any[]
t_all = Any[]
freqs = [1]
num_freqs = length(freqs)
num_amps = [4]
protocols = Array{Vector{Function}, 1}(undef, sum(num_amps))
for l = range(1,num_freqs,step=1)
	w = freqs[l]
	for k = range(1,num_amps[l],step=1)
		# Read and store data
		data = readdlm("data/gel_"*string(w)*"rads_"*string(k)*".csv", ',', Float32)
		push!(t_all, data[5:5:end,4]/τm)
		σ12_data = data[5:5:end,6]
		N1_data = data[5:5:end,3]
		push!(σ12_all, σ12_data/Gm)
		push!(γ12_all, data[5:5:end,5])
		push!(N1_all, (N1_data .- N1_data[1])/Gm)

		# Build strain rate function
		γdot_fd = ((data[2:end,5]) .- (data[1:end-1,5]))./(data[2:end,4]/τm .- data[1:end-1,4]/τm)
		t_mid = (data[2:end,4] + data[1:end-1,4])/(2*τm)
		interp = LinearInterpolation(γdot_fd,t_mid)
		γdot_fun(t) = interp(t)
		gradv = [v11,v12,v13,γdot_fun,v22,v23,v31,v32,v33]
		protocols[sum(num_amps[1:l-1]) + k] = gradv
	end
end

tspan = (0f0, t_all[end][end])
saveat = t_all[end][end]/length(t_all[end])
tspans = [tspan, tspan, tspan, tspan, tspan, tspan, tspan, tspan, tspan, tspan, tspan]

# NN model for the nonlinear function F(σ,γ̇)
model_univ_1 = FastChain(FastDense(9, 32, tanh, bias=false),
                       FastDense(32, 32, tanh, bias=false),
                       FastDense(32, 3, bias=false))
model_univ_2 = FastChain(FastDense(9, 32, tanh),
                       FastDense(32, 32, tanh),
                       FastDense(32, 6))

# The protocol at which we'll start continuation training
# (choose start_at > length(protocols) to skip training)
start_at = 5

if start_at > 1
	# Load the partially-trained model (if not starting from scratch)
	@load "weights_603.bson" θ
	p_model = θ
	n_weights_1 = length(initial_params(model_univ_1))
	n_weights_2 = length(initial_params(model_univ_2))
else
	# The model weights are destructured into a vector of parameters
	p_model_1 = initial_params(model_univ_1)
	p_model_2 = initial_params(model_univ_2)
	n_weights_1 = length(p_model_1)
	n_weights_2 = length(p_model_2)
	p_model = zeros(n_weights_1 + n_weights_2)
end

# Parameters of the linear response (η0,τ)
η0 = [1]
τ = [1]
n_modes = length(η0)
p_system = Float32[η0; τ]

θ0 = zeros(length(p_model))
θi = p_model

iter = 0
losses = []
callback = function (θ, l, protocols, tspans, σ0, σ12_all)
  global iter, losses
  iter += 1
  push!(losses, l)

  # Uncomment the following two lines to save the model weights at each interation
  #filename = "weights/weights_" * string(iter) * ".bson"
  #@save filename θ
  
  println(l)
  println(iter)
  return false
end

# Continuation training loop
for k = range(start_at,length(protocols),step=1)
	loss_fn(θ) = loss_univ(θ, p_system, n_modes, protocols[1:k], tspans[1:k], σ0, σ12_all, saveat)
	cb_fun(θ, l) = callback(θ, l, protocols[1:k], tspans[1:k], σ0, σ12_all)
	@time result_univ = DiffEqFlux.sciml_train(loss_fn, θi,
					     AMSGrad(),
					     cb = cb_fun,
					     allow_f_increases = false,
					     maxiters = 200)

	global θi = result_univ.minimizer
	
	# Save the weights
	@save "tbnn.bson" θi	
end

# Uncomment to plot the training curve
fig, ax = subplots()
ax.semilogy(losses)

# Test the UDE on a new condition
target = ["σ12","σ12","σ12","σ12"]

Gp_data = Any[]
Gpp_data = Any[]
γ0_data = Any[]
N1_zero_data = Any[]
N1_range_data = Any[]
N1_phase_data = Any[]
for k = range(1,length(protocols),step=1)
	# Get data
	t_data = τm*t_all[k]
	local σ12_data = Gm*σ12_all[k]
	γ12_data = γ12_all[k]
	N1_data = Gm*N1_all[k]
	
	# Compute the Fourier transform
	σ12_data_fft = rfft(σ12_data)
	γ12_data_fft = rfft(γ12_data)
	N1_data_fft = rfft(N1_data)
	ω_fft = 2*π*rfftfreq(length(σ12_data), (length(t_data)-1)/(t_data[end]-t_data[1]))
	
	# Get values at the first harmonic
	I_min = argmin(abs.(ω_fft .- 1))
	I_min_2 = argmin(abs.(ω_fft .- 2))
	G1 = σ12_data_fft[I_min]/γ12_data_fft[I_min]
	N1_2 = N1_data_fft[I_min_2]/γ12_data_fft[I_min]
	
	# Store in arrays
	push!(Gp_data, real(G1))
	push!(Gpp_data, imag(G1))
	push!(γ0_data, abs(γ12_data_fft[I_min])/length(γ12_data_fft))
	push!(N1_zero_data, sum(N1_data)/length(N1_data))
	push!(N1_range_data, maximum(N1_data[end-600:end]) - minimum(N1_data[end-600:end]))
	push!(N1_phase_data, angle(N1_2))

	# Solve the UDE pre-training
	res = solve_ude_mm(protocols[k], tspan, θ0, p_system, n_modes, saveat)
	t_pre = τm*res[1]
	sol_pre = Gm*res[2]
	σ12_ude_pre = sol_pre[4,:]
	N1_ude_pre = sol_pre[1,:] - sol_pre[2,:]
	N2_ude_pre = sol_pre[2,:] - sol_pre[3,:]
	
	# Solve the UDE post-training
	res = solve_ude_mm(protocols[k], tspan, θi, p_system, n_modes, saveat)
	t_post = τm*res[1]
	sol_univ = Gm*res[2]
	σ12_ude_post = sol_univ[4,:]
	N1_ude_post = sol_univ[1,:] - sol_univ[2,:]
	N2_ude_post = sol_univ[2,:] - sol_univ[3,:]

	# Plot
	if target[k] == "σ12"
		fig, ax = subplots()
		ax.plot(γ12_data,σ12_ude_pre[2:end],"b--",lw=2)
		ax.plot(γ12_data,σ12_ude_post[2:end],"r-",lw=2)
		ax.plot(γ12_data[1:20:end],σ12_data[1:20:end],"ko")
	elseif target[k] == "N1"
		fig, ax = subplots()
		ax.plot(γ12_data,N1_ude_pre[2:end],"b--",lw=2)
		ax.plot(γ12_data,N1_ude_post[2:end],"r-",lw=2)
		N1_data_new = copy(N1_data)

		# Smooth the normal stress data over a sliding window
		for i in range(11,length(N1_data)-10,step=1)
			N1_data_new[i] = N1_data[i]/21
			for j in range(1,10,step=1)
				N1_data_new[i] += N1_data[i-j]/21 + N1_data[i+j]/21
			end
		end
		ax.plot(γ12_data[1:5:end],N1_data_new[1:5:end],"ko")
	elseif target[k] == "N2"
		fig, ax = subplots()
		ax.plot(t_pre,N2_ude_pre,"b--",lw=2)
		ax.plot(t_post,N2_ude_post,"r-",lw=2)
	elseif target[k] == "ηE"
		fig, ax = subplots()
		ax.plot(t_pre,-N2_ude_pre-N1_ude_pre,"b--",lw=2)
		ax.plot(t_post,-N2_ude_post-N1_ude_post,"r-",lw=2)
	end
	ax[:tick_params](axis="both", direction="in", which="both", right="true", top="true", labelsize=12)
	savefig("test"*string(k)*".pdf")
end

Gp_pre = Number[]
Gpp_pre = Number[]
N1_pre = Number[]
N1_phase_pre = Number[]
N1_range_pre = Number[]
Gp_post = Number[]
Gpp_post = Number[]
N1_post = Number[]
N1_phase_post = Number[]
N1_range_post = Number[]
γ0 = Number[]
ω = 1
for γ0i = 10 .^ (range(-2,log10(3),length=50))
	push!(γ0, γ0i)
	γdot_fun(t) = γ0i*τm*ω*cos(τm*ω*t)
	gradv = [v11,v12,v13,γdot_fun,v22,v23,v31,v32,v33]
	res_pre = solve_ude_mm(gradv, tspan, θ0, p_system, n_modes, saveat)
	res_post = solve_ude_mm(gradv, tspan, θi, p_system, n_modes, saveat)

	# Compute FFTs
	t_pre = τm*res_pre[1]
	γ12 = γ0i*sin.(ω*t_pre) 
	sol_pre = Gm*res_pre[2]
	σ12_ude_pre = sol_pre[4,:]
	σ12_pre_fft = rfft(σ12_ude_pre)
	γ12_fft = rfft(γ12)
	ω_fft = 2*π*rfftfreq(length(σ12_ude_pre), (length(t_pre)-1)/(t_pre[end]-t_pre[1]))
	I_min = argmin(abs.(ω_fft .- 1))

	sol_post = Gm*res_post[2]
	σ12_ude_post = sol_post[4,:]
	σ12_post_fft = rfft(σ12_ude_post)

	N1_ude_pre = sol_pre[1,:] - sol_pre[2,:]
	N1_pre_fft = rfft(N1_ude_pre)
	N1_ude_post = sol_post[1,:] - sol_post[2,:]
	N1_post_fft = rfft(N1_ude_post)
	I_min_2 = argmin(abs.(ω_fft .- 2))
	N1_2_pre = N1_pre_fft[I_min_2]/γ12_fft[I_min]
	N1_2_post = N1_post_fft[I_min_2]/γ12_fft[I_min]

	# Compute moduli
	G1_pre = σ12_pre_fft[I_min]/γ12_fft[I_min]
	G1_post = σ12_post_fft[I_min]/γ12_fft[I_min]
	push!(Gp_pre, real(G1_pre))
	push!(Gpp_pre, imag(G1_pre))
	push!(N1_pre, sum(sol_pre[1,:] - sol_pre[2,:])/length(sol_pre[1,:]))
	push!(N1_range_pre, maximum(sol_pre[1,:] - sol_pre[2,:]) - minimum(sol_pre[1,:] - sol_pre[2,:]))
	push!(Gp_post, real(G1_post))
	push!(Gpp_post, imag(G1_post))
	push!(N1_post, sum(sol_post[1,:] - sol_post[2,:])/length(sol_post[1,:]))
	push!(N1_range_post, maximum(sol_post[1,:] - sol_post[2,:]) - minimum(sol_post[1,:] - sol_post[2,:]))
	push!(N1_phase_pre, angle(N1_2_pre))
	push!(N1_phase_post, angle(N1_2_post))
end

fig, ax = subplots(2,1)
ax[1].loglog(γ0[1:10], Gp_pre[1:10], "r--")
ax[1].loglog(γ0[1:10], Gpp_pre[1:10], "b--")
ax[1].loglog(γ0, Gp_post, "r-")
ax[1].loglog(γ0, Gpp_post, "b-")
ax[1].loglog(γ0_data[1:3], Gp_data[1:3], "ro", fillstyle="none")
ax[1].loglog(γ0_data[1:3], Gpp_data[1:3], "bo", fillstyle="none")
ax[1].loglog(γ0_data[4:end], Gp_data[4:end], "ro")
ax[1].loglog(γ0_data[4:end], Gpp_data[4:end], "bo")
ax[1][:tick_params](axis="both", direction="in", which="both", right="true", top="true", labelsize=12)
ax[1].set_xticklabels([])

#fig, ax = subplots()
ax[2].loglog(γ0[1:10], N1_pre[1:10]./γ0[1:10].^2, ls="--", c="purple")
ax[2].loglog(γ0, N1_post./γ0.^2, ls="-", c="purple")
ax[2].loglog(γ0_data, N1_zero_data./γ0_data.^2, ls="", marker="o", c="purple")
ax[2][:tick_params](axis="both", direction="in", which="both", right="true", top="true", labelsize=12)
subplots_adjust(wspace=0, hspace=0)

