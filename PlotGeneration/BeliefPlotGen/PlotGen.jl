using CairoMakie
using CSV
using Colors

color_lst = [colorant"#332288" colorant"#555555" colorant"#117733" colorant"#882255"]
# Blue, Grey, Green, Red, Yellow
# Colorblind friendly color palette

f = CSV.File("HopfSampling/HopfSampling.csv")

Hopf_random_sampling_trace  = f.random_sampling_trace 
Hopf_optimal_sampling_trace = f.optimal_sampling_trace
Hopf_optimal_sampling_trace_ekf = f.optimal_sampling_trace_ekf
Hopf_approx_sampling_trace  = f.approx_sampling_trace
Hopf_approx_sampling_trace_ekf = f.approx_sampling_trace_ekf
Hopf_samples = f.samples


f = CSV.File("LorenzSampling/LorenzSampling.csv")

Lorenz_random_sampling_trace = f.random_sampling_trace 
Lorenz_optimal_sampling_trace = f.optimal_sampling_trace
Lorenz_optimal_sampling_trace_ekf = f.optimal_sampling_trace_ekf
Lorenz_approx_sampling_trace = f.approx_sampling_trace
Lorenz_approx_sampling_trace_ekf = f.approx_sampling_trace_ekf
Lorenz_samples = f.samples

max_steps = length(Lorenz_samples)


noto_sans = "./resources/NotoSans-Regular.ttf"
noto_sans_bold = "./resources/NotoSans-Bold.ttf"

tickfontsize    = 32
labelfontsize   = 38
basewidth = 2


f = Figure(font=noto_sans, resolution=(1200,700), figure_padding=40)

ax1 = Axis( f[1,2],
            xticklabelsize=tickfontsize, 
            yticklabelsize=tickfontsize, 
            yticklabelpad=2,
            xlabel="Sample", ylabel = "Tr(Σ)",
            xlabelsize=labelfontsize,
            ylabelsize=labelfontsize,
            yscale = log10,
            yminorticksvisible = true,
            yminorgridvisible = true,
            yminorticks = IntervalsBetween(10),
            xminorticksvisible = true,
            xminorgridvisible = true,
            xminorticks = IntervalsBetween(10),
            title="Lorenz System",
            titlesize=labelfontsize)

ylims!(ax1,(1e-1,1e1))
xlims!(ax1,(0,1000))

l2 = lines!(ax1, 1:max_steps, Lorenz_optimal_sampling_trace ./ Lorenz_optimal_sampling_trace,     color=color_lst[1], linewidth=basewidth)
l3 = lines!(ax1, 1:max_steps, Lorenz_optimal_sampling_trace_ekf ./ Lorenz_optimal_sampling_trace, color=color_lst[2], linewidth=basewidth)
l1 = lines!(ax1, 1:max_steps, Lorenz_approx_sampling_trace ./ Lorenz_optimal_sampling_trace,      color=color_lst[3], linewidth=basewidth)
l4 = lines!(ax1, 1:max_steps, Lorenz_approx_sampling_trace_ekf./ Lorenz_optimal_sampling_trace,  color=color_lst[4], linewidth=basewidth)
#l5 = lines!(ax1, 1:max_steps, Lorenz_random_sampling_trace./ Lorenz_optimal_sampling_trace, color=:black)


ax2= Axis(  f[1,1],
            xticklabelsize=tickfontsize, 
            yticklabelsize=tickfontsize, 
            yticklabelpad=2,
            xlabel="Sample", ylabel = "Tr(Σ)",
            xlabelsize=labelfontsize,
            ylabelsize=labelfontsize,
            yscale = log10,
            yminorticksvisible = true,
            yminorgridvisible = true,
            yminorticks = IntervalsBetween(10),
            xminorticksvisible = true,
            xminorgridvisible = true,
            xminorticks = IntervalsBetween(10),
            title="Hopf Bifurcation",
            titlesize=labelfontsize)


ylims!(ax2,(1e-1,1e1))
xlims!(ax2,(0,1000))

l2 = lines!(ax2, 1:max_steps, Hopf_optimal_sampling_trace ./ Hopf_optimal_sampling_trace,     color=color_lst[1], linewidth=basewidth)
l3 = lines!(ax2, 1:max_steps, Hopf_optimal_sampling_trace_ekf./ Hopf_optimal_sampling_trace, color=color_lst[2], linewidth=basewidth)
l1 = lines!(ax2, 1:max_steps, Hopf_approx_sampling_trace./ Hopf_optimal_sampling_trace,      color=color_lst[3], linewidth=basewidth)
l4 = lines!(ax2, 1:max_steps, Hopf_approx_sampling_trace_ekf./ Hopf_optimal_sampling_trace,  color=color_lst[4], linewidth=basewidth)



l1elem = LineElement(color=color_lst[3], linewidth=10)
l2elem = LineElement(color=color_lst[1], linewidth=10)
l3elem = LineElement(color=color_lst[2], linewidth=10)
l4elem = LineElement(color=color_lst[4], linewidth=10)


Legend(f[2,:], 
       [l1elem, l4elem, l2elem, l3elem], 
       ["1D Limit Set Oracle", "1D Limit Set EKF", "Dynamic Programming Oracle", "Dynamic Programming EKF"],
       orientation=:horizontal,
       nbanks=2,
       labelsize=36)


colgap!(f.layout,60)

Label(f[1,1,TopLeft()], "A", font=noto_sans_bold, textsize = 40, halign=:left, valign=:bottom, padding=(25,0,10,0))
Label(f[1,2,TopLeft()], "B", font=noto_sans_bold, textsize = 40, halign=:left, valign=:bottom, padding=(25,0,10,0))

save("out/BeliefSim.pdf",f)
