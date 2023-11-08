## These are highly specific functions used for the paper
## Some are overriden before analysis is performed- this is not analysis ready at this time. 
## TODO:
##     1) Remove any unused functions (left over from other analysis)
##     2) Update any functions that are later overriden in analysis to avoid confusion (prior values etc)  


using JSON
using Glob
using OrdinaryDiffEq, ForwardDiff

using Plots
using Distributions 
using PyPlot
using LinearAlgebra

using ParameterizedFunctions,ODEInterfaceDiffEq, StaticArrays

using NLopt

using ThreadPools
using Base.Threads: nthreads
using Base.Threads

using DiffEqSensitivity
using LatinHypercubeSampling

using CPUTime




using StatsBase: sample
using Random


function lognorm_ll(x,mu,sigma)
    return log(1.0/(x*sigma*sqrt(2*3.141592)) *  exp(- (log(x) - mu)^2/(2*sigma^2)))

end


function getPriorSO(par_,model,ALL_RUNS)
    
    t_LL=0
    
    means,lower,upper=[par_umap(x,model,ALL_RUNS) for x in get_priors(ALL_RUNS)]
    sigmas = par_umap(get_priors_sigma(ALL_RUNS),model,ALL_RUNS)

    par = par_umap(par_,model,ALL_RUNS)
    
    let global_prior
        global_prior=0
        N_LOCAL=4 #in this case we know we have 4 local varaiables, otherwise this gets messy. 
        for (p_idx,(m,l,u,p,sig)) in enumerate(zip(means[1+N_LOCAL*length(ALL_RUNS):end], 
                            lower[1+N_LOCAL*length(ALL_RUNS):end], 
                            upper[1+N_LOCAL*length(ALL_RUNS):end],
                    par[1+N_LOCAL*length(ALL_RUNS):end],
                    sigmas[1+N_LOCAL*length(ALL_RUNS):end] ))

                global_prior += -norm_ll(m,sig,p)/length(ALL_RUNS) #/4 #minimize neg LL! so negative 
        end
            
        for e in enumerate(ALL_RUNS)
            for (m,l,u,p,sig) in zip(means[1+N_LOCAL*(e[1]-1):N_LOCAL*(e[1])], 
                                lower[1+N_LOCAL*(e[1]-1):N_LOCAL*(e[1])], 
                                upper[1+N_LOCAL*(e[1]-1):N_LOCAL*(e[1])],
                                par[1+N_LOCAL*(e[1]-1):N_LOCAL*(e[1])],
                                sigmas[1+N_LOCAL*(e[1]-1):N_LOCAL*(e[1])])
                    
                t_LL +=  -norm_ll(m,sig,p) #minimize neg LL! so negative 
            end
            t_LL  += global_prior  #divided into the 4 sperate LLs
        end
        return t_LL
    
    end
end

 

function getR(x,y)
        sq(z) = -1*sqrt(z)
        return [[(x ^ 2 - y ^ 2 * sq(1 - x ^ 2 - y ^ 2)) / (x ^ 2 + y ^ 2),
                       -((x * y * (1 + sq(1 - x ^ 2 - y ^ 2))) / (x ^ 2 + y ^ 2)), y],
                      [-((x * y * (1 + sq(1 - x ^ 2 - y ^ 2))) / (x ^ 2 + y ^ 2)), (y ^ 2 -
                                                                                      x ^ 2 * sq(
                                  1 - x ^ 2 - y ^ 2)) / (x ^ 2 + y ^ 2),
                       x], [-y, -x, -sq(1 - x ^ 2 - y ^ 2)]]
end
    
    
    

function get_q_from_a(ϕIC, θIC, ψIC)
        # Given inital conditions in 313 euler angles, get quaternions
        qIC = [cos(ϕIC / 2) * cos(θIC / 2) * cos(ψIC / 2) -
                        sin(ϕIC / 2) * cos(θIC / 2) * sin(ψIC / 2),
                        cos(ϕIC / 2) * sin(θIC / 2) * cos(ψIC / 2) +
                        sin(ϕIC / 2) * sin(θIC / 2) * sin(ψIC / 2),
                        cos(ϕIC / 2) * sin(θIC / 2) * sin(ψIC / 2) -
                        sin(ϕIC / 2) * sin(θIC / 2) * cos(ψIC / 2),
                        cos(ϕIC / 2) * cos(θIC / 2) * sin(ψIC / 2) +
                        sin(ϕIC / 2) * cos(θIC / 2) * cos(ψIC / 2)]
return qIC
end


function xp_fn(q)
        # from Rq to position
        return [2 * q[2] * q[4] + 2 * q[1] * q[3], 2 * q[3] * q[4] - 2 * q[1] * q[2]]
end
    

    
function eval_xyz_init(ϕIC, θIC, ψIC)
        # Given inital conditions in 313 euler angles, get corresponding x,y position of particle
        qIC = get_q_from_a(ϕIC, θIC, ψIC)
        return xp_fn(qIC)
end

function simple_norm(A)
         x = zero(eltype(A))
         for v in A
           x += v * v
         end
         x
end

function norm_ll(mu,sig,x)
    return -log(sig*sqrt(2*pi)) - 0.5*(((mu-x)/sig)^2)
end

function r_ll(r,sig) #where r is residuals. 
    return -log(sig*sqrt(2*pi)) .- 0.5*(((r)./sig).^2)
end

function get_EA(x,y; ϕ0=0.0)
    let Θ,Ψ
        pi=3.141592653589793

        if abs(x) < 1e-4 
            if abs(y) < 1e-4
                Θ=pi
                Ψ=-pi/2
            elseif abs(y-1) < 1e-4 
                Θ=pi/2
                Ψ=pi
            elseif abs(y+1) < 1e-4 
                Θ=0.0
                Ψ=pi/2
            elseif y>0
                Θ=2*atan((1+y*sqrt((1-y^2)/y^2))/y)
                Ψ=pi
            elseif y<0
                Θ=2*atan((-1+y*sqrt((1-y^2)/y^2))/y)
                Ψ=0.0
            end 

        elseif abs(y) < 1e-4 
            if abs(x-1) < 1e-4 
                Θ=pi/2
                Ψ=pi/2
            elseif abs(x+1) < 1e-4
                Θ=-pi/2
                Ψ=pi/2
            elseif x > 0
                Θ=2*atan((1+x*sqrt((1-x^2)/x^2))/x)
                Ψ=pi/2
            elseif x < 0
                Θ=-2*atan((-1+x*sqrt((1-x^2)/x^2))/x)
                Ψ=pi/2

            end  

        elseif x < 0 
            if y < 0 
                Ψ=2*atan((y+x*sqrt((x^2+y^2)/x^2))/x)
                Θ=2*atan((y+x*sqrt((x^2+y^2)/x^2)-x^2*sqrt(-(
                    ((-1+x^2+y^2)*(x^2+2*y^2+2*x*y*sqrt((x^2+y^2)/x^2)))/(x^2+y^2+x*y*sqrt((x^2+y^2)/x^2))^2))-y^2*sqrt(-(
                    ((-1+x^2+y^2)*(x^2+2*y^2+2*x*y*sqrt((x^2+y^2)/x^2)))/(x^2+y^2+x*y*sqrt((x^2+y^2)/x^2))^2))-x*y*sqrt((x^2+y^2)/x^2)*sqrt(-(
                    ((-1+x^2+y^2)*(x^2+2*y^2+2*x*y*sqrt((x^2+y^2)/x^2)))/(x^2+y^2+x*y*sqrt((x^2+y^2)/x^2))^2)))/(x^2+y^2+x*y*sqrt((x^2+y^2)/x^2)))
            else 
                 Ψ=2*atan((y+x*sqrt((x^2+y^2)/x^2))/x)
                 Θ=2*atan((y+x*sqrt((x^2+y^2)/x^2)-x^2*sqrt(-(
                    ((-1+x^2+y^2)*(x^2+2*y^2+2*x*y*sqrt((x^2+y^2)/x^2)))/(x^2+y^2+x*y*sqrt((x^2+y^2)/x^2))^2))-y^2*sqrt(-(
                    ((-1+x^2+y^2)*(x^2+2*y^2+2*x*y*sqrt((x^2+y^2)/x^2)))/(x^2+y^2+x*y*sqrt((x^2+y^2)/x^2))^2))-x*y*sqrt((x^2+y^2)/x^2)*sqrt(-(
                    ((-1+x^2+y^2)*(x^2+2*y^2+2*x*y*sqrt((x^2+y^2)/x^2)))/(x^2+y^2+x*y*sqrt((x^2+y^2)/x^2))^2)))/(x^2+y^2+x*y*sqrt((x^2+y^2)/x^2)))
            end
        elseif x > 0 
            Ψ=2*atan((y+x*sqrt((x^2+y^2)/x^2))/x)
            Θ=2*atan((y+x*sqrt((x^2+y^2)/x^2)+x^2*sqrt(-(
                ((-1+x^2+y^2)*(x^2+2*y^2+2*x*y*sqrt((x^2+y^2)/x^2)))/(x^2+y^2+x*y*sqrt((x^2+y^2)/x^2))^2))+y^2*sqrt(-(
                ((-1+x^2+y^2)*(x^2+2*y^2+2*x*y*sqrt((x^2+y^2)/x^2)))/(x^2+y^2+x*y*sqrt((x^2+y^2)/x^2))^2))+x*y*sqrt((x^2+y^2)/x^2)*sqrt(-(
                ((-1+x^2+y^2)*(x^2+2*y^2+2*x*y*sqrt((x^2+y^2)/x^2)))/(x^2+y^2+x*y*sqrt((x^2+y^2)/x^2))^2)))/(x^2+y^2+x*y*sqrt((x^2+y^2)/x^2)))

        end

        return(ϕ0,Θ,Ψ)
    end
end

function xp_ua_fn(q)
    return [sin(q[2])*sin(q[3]),-cos(q[3])*sin(q[2])]#, cos(q[2])]
end    

function xp_ua_fn_tilt(v, Φ, Θ)
    e1 = v[1]
    e2 = v[2]
    e3 = v[3]
    return [
        (-cos(e2)) * sin(Φ) + cos(Φ) * sin(e2) * sin(e3),
        (-cos(Θ)) * cos(e3) * sin(e2) + sin(Θ) * (cos(e2) * cos(Φ) + sin(e2) * sin(Φ) * sin(e3))
    ]
end

function corr_ll(r, ϵ, σ)
    #8.34 Sivia 
    
    N= length(r)
    
    r2 = r.^2
    ϕ = r2[1] + r2[end] 
    χ2= sum(r2)
    ψ= sum(r[1:end-1] .* r[2:end]) 
    
    Q = χ2 + ϵ * (ϵ * (χ2-ϕ) - 2*ψ)

    L = -1/2.0 * (( N - 1) * log( 1 - ϵ^2) + 2*N*log(σ) + Q/(σ^2 *(1 - ϵ^2)))
    
    return L
end


#     >-###'>           g         <'###-<
model_g = @ode_def begin
  de1 = (1/(4*a^2))*(-3*cos(e3 - (t+phase)*ω)*sin(e1)* sin(φ) + cos(e1)*(cos(φ)*(3 - 4*a^2 + 4*a^2*csc(e2)^2)*sin(e2) + (-3 + 4*a^2)*cos(e2)*sin(φ)* sin(e3 - (t+phase)*ω)))
  de2 =  cos(e2)*cos(φ)*sin(e1) + sin(e2)*(mg + sin(e1)*sin(φ)*sin(e3 - (t+phase)*ω))
  de3 =  (-cos(e1))*(cos(φ)*cot(e2) + sin(φ)*sin(e3 - (t+phase)*ω))
end φ ω a μ mg α1 Φg Θg phase
#     >-###'>           p         <'###-<
model_p = @ode_def begin
  de1 = (1/(4*a^2))*(3*cos(e1)* cos(φ)*(1 - a^2 + a^2*csc(e2)^2)* sin(e2) - 3*cos(e3 - (t+phase)*ω)*sin(e1)*sin(φ) + cos(e2)*(cos(e1)*(a^2*cos(φ)* cot(e2) + (-3 + 4*a^2)*sin(φ)* sin(e3 - (t+phase)*ω)) + 2*a^2*α1*(cos(e3 - (t+phase)*ω)*cot(e2)* sin(2*φ) + sin(φ)^2*sin(2*(e3 - (t+phase)*ω)))))
  de2 =  (1/8)*α1*(2*cos(e3 - (t+phase)*ω)^2 - cos(2*φ)*(-3 + cos(2*(e3 - (t+phase)*ω))))* sin(2*e2) + cos(e2)*cos(φ)* sin(e1) + (1/2)*(2*sin(e2)*sin(e1)*sin(φ) - α1*cos(2*e2)*sin(2*φ))* sin(e3 - (t+phase)*ω)
  de3 =  -((cos(e1) + α1*cos(e3 - (t+phase)*ω)* sin(φ))*(cos(φ)*cot(e2) + sin(φ)*sin(e3 - (t+phase)*ω)))
end φ ω a μ mg α1 Φg Θg phase
#     >-###'>           a         <'###-<
model_a = @ode_def begin
  de1 = (1/(4*a^2))*(cos(μ)*cos(e1)* cos(φ)*(3 - 4*a^2 + 4*a^2*csc(e2)^2)* sin(e2) - cos(e3 - (t+phase)*ω)*(4*a^2*cot(e2)*sin(μ) + 3*cos(μ)*sin(e1))*sin(φ) + (-3 + 4*a^2)*cos(e2)*cos(μ)*cos(e1)* sin(φ)*sin(e3 - (t+phase)*ω))
  de2 =  cos(φ)*((-sin(e2))*sin(μ) + cos(e2)*cos(μ)*sin(e1)) + (cos(e2)*sin(μ) + cos(μ)*sin(e2)*sin(e1))*sin(φ)* sin(e3 - (t+phase)*ω)
  de3 =  cos(e3 - (t+phase)*ω)*csc(e2)*sin(μ)* sin(φ) - cos(μ)*cos(e1)* (cos(φ)*cot(e2) + sin(φ)*sin(e3 - (t+phase)*ω))
end φ ω a μ mg α1 Φg Θg phase
#     >-###'>           g12         <'###-<
model_g12 = @ode_def begin
  de1 = (1/(4*a^2))*(3*cos(e1)* cos(φ)*(1 - a^2 + a^2*csc(e2)^2)* sin(e2) - 3*cos(e3 - (t+phase)*ω)*sin(e1)*sin(φ) + 4*a^2*mg*cot(e2)* (cos(Θg)*cos(e3)*sin(Φg) - sin(Θg)*sin(e3)) + cos(e2)* cos(e1)*(a^2*cos(φ)*cot(e2) + (-3 + 4*a^2)* sin(φ)*sin(e3 - (t+phase)*ω)))
  de2 =  cos(e2)*((-mg)*cos(e3)*sin(Θg) + cos(φ)*sin(e1)) + mg*cos(Θg)*(cos(Φg)*sin(e2) - cos(e2)*sin(Φg)*sin(e3)) + sin(e2)*sin(e1)*sin(φ)* sin(e3 - (t+phase)*ω)
  de3 =  mg*csc(e2)*((-cos(Θg))*cos(e3)* sin(Φg) + sin(Θg)*sin(e3)) - cos(e1)*(cos(φ)*cot(e2) + sin(φ)*sin(e3 - (t+phase)*ω))
end φ ω a μ mg α1 Φg Θg phase
#     >-###'>           pg         <'###-<
model_pg = @ode_def begin
  de1 = (1/(4*a^2))*(3*cos(e1)* cos(φ)*(1 - a^2 + a^2*csc(e2)^2)* sin(e2) - 3*cos(e3 - (t+phase)*ω)*sin(e1)*sin(φ) + cos(e2)*(cos(e1)*(a^2*cos(φ)* cot(e2) + (-3 + 4*a^2)*sin(φ)* sin(e3 - (t+phase)*ω)) + 2*a^2*α1*(cos(e3 - (t+phase)*ω)*cot(e2)* sin(2*φ) + sin(φ)^2*sin(2*(e3 - (t+phase)*ω)))))
  de2 =  mg*sin(e2) + (1/8)*α1*(2*cos(e3 - (t+phase)*ω)^2 - cos(2*φ)*(-3 + cos(2*(e3 - (t+phase)*ω))))* sin(2*e2) + cos(e2)*cos(φ)*sin(e1) + (1/2)*(2*sin(e2)*sin(e1)*sin(φ) - α1* cos(2*e2)*sin(2*φ))*sin(e3 - (t+phase)*ω)
  de3 =  -((cos(e1) + α1*cos(e3 - (t+phase)*ω)* sin(φ))*(cos(φ)*cot(e2) + sin(φ)*sin(e3 - (t+phase)*ω)))
end φ ω a μ mg α1 Φg Θg phase
#     >-###'>           ag         <'###-<
model_ag = @ode_def begin
  de1 = (1/(4*a^2))*(cos(μ)*cos(e1)* cos(φ)*(3 - 4*a^2 + 4*a^2*csc(e2)^2)* sin(e2) - cos(e3 - (t+phase)*ω)*(4*a^2*cot(e2)*sin(μ) + 3*cos(μ)*sin(e1))*sin(φ) + (-3 + 4*a^2)*cos(e2)*cos(μ)*cos(e1)* sin(φ)*sin(e3 - (t+phase)*ω))
  de2 =  mg*sin(e2) - cos(φ)*sin(e2)*sin(μ) + cos(e2)*cos(μ)*cos(φ)*sin(e1) + (cos(e2)*sin(μ) + cos(μ)*sin(e2)*sin(e1))*sin(φ)* sin(e3 - (t+phase)*ω)
  de3 =  cos(e3 - (t+phase)*ω)*csc(e2)*sin(μ)* sin(φ) - cos(μ)*cos(e1)* (cos(φ)*cot(e2) + sin(φ)*sin(e3 - (t+phase)*ω))
end φ ω a μ mg α1 Φg Θg phase
#     >-###'>           ap         <'###-<
model_ap = @ode_def begin
  de1 = (1/(32*a^2))*(-6*α1*cos(e1)*cos(φ)^2* sin(2*e2)*sin(2*μ) - 9*α1*cos(φ)^2*sin(e2)^2* sin(2*e1) + 3*α1*cos(2*μ)*cos(φ)^2*sin(e2)^2* sin(2*e1) - 16*a^2*cos(e1)^2*cos(e3)*cos((t+phase)*ω)* sin(2*e2)*sin(μ)*sin(φ) - 16*a^2*cos(e3)*cos((t+phase)*ω)*sin(2*e2)*sin(μ)* sin(e1)^2*sin(φ) + 8*a^2*α1*cos(e1)^3*cos(e3)^2*cos((t+phase)*ω)^2* sin(2*e2)*sin(2*μ)*sin(φ)^2 + 8*a^2*α1*cos(e1)*cos(e3)^2*cos((t+phase)*ω)^2* sin(2*e2)*sin(2*μ)*sin(e1)^2*sin(φ)^2 + 9*α1*cos(e3 - (t+phase)*ω)^2*sin(2*e1)* sin(φ)^2 - 3*α1*cos(2*μ)*cos(e3 - (t+phase)*ω)^2* sin(2*e1)*sin(φ)^2 + 24*cos(μ)*(cos(e1)*cos(φ)*sin(e2) - cos(e3 - (t+phase)*ω)*sin(e1)*sin(φ)) - 9*α1*cos(e1)^2*cos(e3 - (t+phase)*ω)*sin(e2)* sin(2*φ) + 3*α1*cos(2*μ)*cos(e1)^2* cos(e3 - (t+phase)*ω)*sin(e2)*sin(2*φ) + 9*α1*cos(e3 - (t+phase)*ω)*sin(e2)* sin(e1)^2*sin(2*φ) - 3*α1*cos(2*μ)*cos(e3 - (t+phase)*ω)* sin(e2)*sin(e1)^2*sin(2*φ) + 6*α1*cos(e3 - 2*(t+phase)*ω)*sin(e2)*sin(2*μ)* sin(e1)* sin(φ)^2*sin(e3) - 6*α1*cos(e1)*cos((t+phase)*ω)*sin(e2)^2* sin(2*μ)*sin(2*φ)*sin(e3) - 8*a^2*α1*cos(e1)^3*cos((t+phase)*ω)*sin(2*e2)* sin(e1)*sin(2*φ)*sin(e3) - 8*a^2*α1*cos(e1)*cos((t+phase)*ω)*sin(2*e2)* sin(e1)^3*sin(2*φ)*sin(e3) - 8*a^2*α1*cos(e1)^3*cos((t+phase)*ω)^2* sin(2*e2)*sin(2*μ)*sin(φ)^2*sin(e3)^2 - 8*a^2*α1*cos(e1)*cos((t+phase)*ω)^2*sin(2*e2)* sin(2*μ)*sin(e1)^2*sin(φ)^2*sin(e3)^2 + 3*α1*sin(e2)*sin(2*μ)*sin(e1)* sin(φ)^2*sin(2*e3) - 12*α1*cos(e3)*cos(e3 - (t+phase)*ω)* sin(e2)*sin(2*μ)*sin(e1)*sin(φ)^2* sin((t+phase)*ω) + 6*α1*cos(e1)*cos(e3)*sin(e2)^2* sin(2*μ)*sin(2*φ)*sin((t+phase)*ω) + 8*a^2*α1*cos(e1)^3*cos(e3)*sin(2*e2)* sin(e1)* sin(2*φ)*sin((t+phase)*ω) + 8*a^2*α1*cos(e1)*cos(e3)*sin(2*e2)* sin(e1)^3*sin(2*φ)* sin((t+phase)*ω) - 16*a^2*cos(e1)^2*sin(2*e2)*sin(μ)*sin(φ)* sin(e3)*sin((t+phase)*ω) - 16*a^2*sin(2*e2)*sin(μ)*sin(e1)^2* sin(φ)*sin(e3)*sin((t+phase)*ω) - 8*a^2*α1*cos(e1)^3*cos(e3)^2*sin(2*e2)* sin(2*μ)*sin(φ)^2*sin((t+phase)*ω)^2 - 8*a^2*α1*cos(e1)*cos(e3)^2*sin(2*e2)* sin(2*μ)*sin(e1)^2*sin(φ)^2*sin((t+phase)*ω)^2 + 8*a^2*α1*cos(e1)^3*sin(2*e2)*sin(2*μ)* sin(φ)^2*sin(e3)^2*sin((t+phase)*ω)^2 + 8*a^2*α1*cos(e1)*sin(2*e2)*sin(2*μ)* sin(e1)^2*sin(φ)^2*sin(e3)^2* sin((t+phase)*ω)^2 + 8*a^2*α1*sin(e2)*sin(2*e2)*sin(μ)^2* sin(e1)^2*sin(φ)^2*sin(2*e3)* sin((t+phase)*ω)^2 + 8*a^2*α1*sin(2*e2)* sin(φ)^2*((-sin(e2))*sin(μ)^2*sin(e1)^2* sin(e3)^2 + cos(e1)*sin(2*μ)*sin(2*e3))* sin(2*(t+phase)*ω) + α1* cos(μ)^2*(6* sin(2*e1)*(cos(φ)^2*sin(e2)^2 - cos(e3 - (t+phase)*ω)^2*sin(φ)^2) + 6*cos(2*e1)*cos(e3 - (t+phase)*ω)*sin(e2)* sin(2*φ) + (-3 + 4*a^2)*sin(2*e2)* sin(2*e1)* sin(2*φ)*sin(e3 - (t+phase)*ω)) - cos(e2)^2*(32*a^2*cos(e3)*cos((t+phase)*ω)*cot(e2)* sin(μ)* sin(e1)^2*sin(φ) - 8*a^2*α1*cos(e3)^2*cos((t+phase)*ω)*cot(e2)* sin(2*e1)*sin(2*φ)* sin(e3) + 8*a^2*α1*cos(μ)*cos(e1)*sin(μ)* (cos(e1)^2* cot(e2)*(4*cos(φ)^2 - cos(2*(e3 + (t+phase)*ω))*sin(φ)^2) + 2*cos(φ)*cos(2*e3)*cos((t+phase)*ω)* sin(e1)^2*sin(φ)*sin(e3)) + 8*a^2*α1*cos(e3)^3*cot(e2)*sin(2*e1)* sin(2*φ)*sin((t+phase)*ω) + 32*a^2*cot(e2)*sin(μ)*sin(e1)^2*sin(φ)* sin(e3)*sin((t+phase)*ω) + 8*a^2*cos(e1)^2*(2*cos(e3)*cos((t+phase)*ω)* sin(μ)*(2*cot(e2)*sin(φ) + α1*sin(e2)*sin(μ)* sin(2*φ)) + sin(e3)*(4*cot(e2)*sin(μ)*sin(φ)* sin((t+phase)*ω) + α1* sin(e2)*(2*sin(μ)^2*sin(2*φ)* sin((t+phase)*ω) + sin(2*μ)*sin(e1)*sin(φ)^2* sin(e3)*sin(2*(t+phase)*ω)))) + 9*α1*sin(2*e1)*sin(φ)^2* sin(e3 - (t+phase)*ω)^2 - 3*α1*cos(2*μ)*sin(2*e1)*sin(φ)^2* sin(e3 - (t+phase)*ω)^2 + 2*α1* cos(e1)*(a^2* sin(e1)^2*(2* cos(2*φ)*((3 + cos(2*(e3 - (t+phase)*ω)))* cot(e2)* sin(2*μ) + 6*sin(e1)) + sin(e1)*(4 + 8*cos(2*(e3 - (t+phase)*ω))*sin(φ)^2) + sin(2*μ)* sin(2*φ)*((-cos((t+phase)*ω))*(-9* sin(e3) + sin(3*e3)) - 8*cos(e3)*sin((t+phase)*ω))) - sin(2*φ)*(3*sin(2*μ) + 8*a^2*cot(e2)*sin(e1)^3* sin(e3)^2)*sin(e3 - (t+phase)*ω) + 4*a^2*cot(e2)*sin(2*μ)*sin(e1)^2* sin(e3 - (t+phase)*ω)^2) + 2*α1*cos(μ)^2*sin(2*e1)* (2* a^2*(-2*cos(e3 - (t+phase)*ω)^2 + cos(2*φ)*(-3 + cos(2*(e3 - (t+phase)*ω)))) + 4*a^2*cot(e2)*sin(2*φ)* sin(e3 - (t+phase)*ω) - 3*sin(φ)^2*sin(e3 - (t+phase)*ω)^2) + 4*a^2*α1* cos(e1)^3*(8*cos(φ)^2* sin(e1) + (-2 - 2*cos(2*(e3 - (t+phase)*ω)) + cos(2*(e3 + (t+phase)*ω)))*cot(e2)* sin(2*μ)*sin(φ)^2 + 4*sin(2*φ)*(sin(2*μ) - cot(e2)*sin(e1)*sin(e3)^2)* sin(e3 - (t+phase)*ω) - 8*sin(e1)*sin(φ)^2* sin(e3 - (t+phase)*ω)^2) + 8*a^2*α1*sin(e2)*sin(2*μ)*sin(e1)^3* sin(φ)^2*sin(2*(e3 - (t+phase)*ω))) + 8*a^2*α1*cos(e2)^3* (cos(e3 - (t+phase)*ω)*(-2*cot(e2)*sin(μ)^2 + sin(2*μ)*sin(e1))*sin(2*φ) - (2*sin(μ)^2 + cot(e2)*sin(2*μ)*sin(e1))*sin(φ)^2* sin(2*(e3 - (t+phase)*ω))) + cos(e2)*(8*cos(μ)* cos(e1)*(4*a^2*cos(φ)*cot(e2) + sin(φ)*(-3 + 4*a^2 + 3*α1*cos((t+phase)*ω)*sin(e2)*sin(μ)* sin(φ)*sin(e3))* sin(e3 - (t+phase)*ω)) + 2*α1* cos(μ)^2*(8*a^2*cos(e1)^2*cos(e3 - (t+phase)*ω)* cot(e2)* sin(2*φ) + (4*a^2 + (-3 + 4*a^2)*cos(2*e1))* sin(φ)^2*sin(2*(e3 - (t+phase)*ω))) + α1*(-4*a^2*cos(2*(e2 + μ))* cos(φ)*cos(e3 - (t+phase)*ω)*cot(e2)* sin(e1)^2*sin(φ) + 8*a^2*cos(e1)^2*cos(e3)*cos((t+phase)*ω)* sin(e2)^2*sin(2*μ)*sin(e1)* sin(2*φ) + 8*a^2*cos(e3)*cos((t+phase)*ω)*sin(e2)^2* sin(2*μ)*sin(e1)^3*sin(2*φ) + 2*cos(e3 - (t+phase)*ω)* sin(e1)*(3*sin(2*μ) + a^2*(6 + 2*cos(2*e2) - cos(2*(e2 - μ)) + 2*cos(2*μ))*cot(e2)* sin(e1))*sin(2*φ) - 16*a^2*cos(e1)^2*cos((t+phase)*ω)^2* sin(e2)^2*sin(μ)^2*sin(φ)^2* sin(2*e3) - 16*a^2*cos((t+phase)*ω)^2*sin(e2)^2*sin(μ)^2* sin(e1)^2*sin(φ)^2*sin(2*e3) + 16*a^2*cos((t+phase)*ω)^2*sin(e1)^4*sin(φ)^2* sin(2*e3) + 4*a^2*cos((t+phase)*ω)^2*sin(2*e1)^2*sin(φ)^2* sin(2*e3) + 8*a^2*cos(e1)^2*sin(e2)^2*sin(2*μ)* sin(e1)*sin(2*φ)*sin(e3)* sin((t+phase)*ω) + 8*a^2*sin(e2)^2*sin(2*μ)*sin(e1)^3* sin(2*φ)*sin(e3)*sin((t+phase)*ω) + 16*a^2*cos(e1)^2*sin(e2)^2*sin(μ)^2* sin(φ)^2*sin(2*e3)*sin((t+phase)*ω)^2 - 16*a^2*sin(e1)^4*sin(φ)^2*sin(2*e3)* sin((t+phase)*ω)^2 - 4*a^2*sin(2*e1)^2*sin(φ)^2* sin(2*e3)*sin((t+phase)*ω)^2 + 16*a^2*cos(e1)^2*cos(e3)^2*sin(e2)^2* sin(μ)^2*sin(φ)^2* sin(2*(t+phase)*ω) + 16*a^2*cos(e3)^2*sin(e2)^2*sin(μ)^2* sin(e1)^2*sin(φ)^2* sin(2*(t+phase)*ω) - 16*a^2*cos(e3)^2*sin(e1)^4*sin(φ)^2* sin(2*(t+phase)*ω) - 4*a^2*cos(e3)^2*sin(2*e1)^2*sin(φ)^2* sin(2*(t+phase)*ω) - 16*a^2*cos(e1)^2*sin(e2)^2* sin(μ)^2*sin(φ)^2*sin(e3)^2* sin(2*(t+phase)*ω) - 8*a^2*cos(e1)^2*cot(e2)*sin(2*μ)* sin(e1)*sin(φ)^2*sin(e3)^2* sin(2*(t+phase)*ω) + 16*a^2*sin(e1)^4*sin(φ)^2*sin(e3)^2* sin(2*(t+phase)*ω) + 4*a^2*sin(2*e1)^2*sin(φ)^2*sin(e3)^2* sin(2*(t+phase)*ω) - 18*a^2*cos(e1)^2*cos(e3)*cot(e2)*sin(2*μ)* sin(e1)*sin(φ)^2*sin(e3 - 2*(t+phase)*ω) + 6*a^2*cos(2*e2)*cos(e1)^2*cos(e3)* cot(e2)*sin(2*μ)*sin(e1)*sin(φ)^2* sin(e3 - 2*(t+phase)*ω) + 9*sin(e2)*sin(2*e1)*sin(2*φ)* sin(e3 - (t+phase)*ω) - 3*cos(2*μ)*sin(e2)*sin(2*e1)* sin(2*φ)*sin(e3 - (t+phase)*ω) - 12*cos(e1)*cos(e3)*sin(e2)*sin(2*μ)* sin(φ)^2*sin((t+phase)*ω)* sin(e3 - (t+phase)*ω) + 9*cos(e1)^2*sin(φ)^2* sin(2*(e3 - (t+phase)*ω)) - 3*cos(2*μ)*cos(e1)^2*sin(φ)^2* sin(2*(e3 - (t+phase)*ω)) + 8*a^2*cos(e1)^2*cot(e2)*sin(2*μ)*sin(e1)* sin(φ)^2* sin(2*(e3 - (t+phase)*ω)) - 9*sin(e1)^2*sin(φ)^2* sin(2*(e3 - (t+phase)*ω)) + 3*cos(2*μ)*sin(e1)^2*sin(φ)^2* sin(2*(e3 - (t+phase)*ω)) + 2*a^2*(-3 + cos(2*e2))* cos(e1)^2*cos(e3)*cot(e2)* sin(2*μ)*sin(e1)*sin(φ)^2* sin(e3 + 2*(t+phase)*ω))))
  de2 =  (1/32)*(-4*α1*cos(e2)^3*cos(e3 - (t+phase)*ω)* sin(2*e1)*sin(2*φ) + 4*α1* cos(e2)^2*((-2*cos(e3 - (t+phase)*ω)^2 + cos(2*φ)*(-3 + cos(2*(e3 - (t+phase)*ω))))* sin(2*μ)* sin(e1) - (1 + 3*cos(2*μ) + 2*cos(2*e1)*sin(μ)^2)*sin(2*φ)* sin(e3 - (t+phase)*ω)) + 2*(α1* cos(e1)^2*(cos(2*μ) - cos(2*e1) + 6*cos(μ)^2*cos(2*φ))*sin(2*e2) + 6*α1*cos(μ)^2*cos(2*φ)*sin(2*e2)* sin(e1)^2 - 16*cos(φ)*cos(e3)^2*sin(e2)*sin(μ)* sin(e1)^2 + 8*α1*cos(φ)^2*cos(e3)^2*sin(e2)^2* sin(2*μ)*sin(e1)^3 + 2*α1*cos(μ)^2*cos(2*(e3 - (t+phase)*ω))* sin(2*e2)*sin(e1)^2*sin(φ)^2 + 2*α1*cos(μ)^2*cos(2*(e3 + (t+phase)*ω))* sin(2*e2)*sin(e1)^2*sin(φ)^2 + 4*α1*cos(e1)^3*cos(e3 - (t+phase)*ω)* sin(e2)*sin(2*μ)*sin(2*φ) + 16*cos(μ)*cos((t+phase)*ω)*sin(e2)*sin(e1)^3* sin(φ)*sin(e3) - 8*α1*cos(e3)^2*cos((t+phase)*ω)*sin(e2)^2* sin(μ)^2*sin(e1)^2*sin(2*φ)*sin(e3) - 8*α1*cos((t+phase)*ω)*sin(2*e2)*sin(2*μ)* sin(e1)^3*sin(2*φ)*sin(e3) + 8*α1*cos(μ)^2*cos((t+phase)*ω)*sin(e2)^2* sin(e1)^4*sin(2*φ)*sin(e3) + 2*α1*cos((t+phase)*ω)*sin(e2)^2*sin(2*e1)^2* sin(2*φ)*sin(e3) + 2*α1*cos(μ)^2*cos((t+phase)*ω)*sin(e2)^2* sin(2*e1)^2*sin(2*φ)*sin(e3) - 16*cos(φ)*sin(e2)*sin(μ)*sin(e1)^2* sin(e3)^2 + 8*α1*cos(φ)^2*sin(e2)^2*sin(2*μ)* sin(e1)^3*sin(e3)^2 - 8*α1*cos((t+phase)*ω)^2*sin(e2)^2*sin(2*μ)* sin(e1)^3*sin(φ)^2* sin(e3)^2 - 8*α1*cos((t+phase)*ω)*sin(e2)^2*sin(μ)^2* sin(e1)^2*sin(2*φ)*sin(e3)^3 - 16*cos(μ)*cos(e3)*sin(e2)*sin(e1)^3* sin(φ)*sin((t+phase)*ω) + 8*α1*cos(e3)^3*sin(e2)^2*sin(μ)^2* sin(e1)^2*sin(2*φ)*sin((t+phase)*ω) + 8*α1*cos(e3)*sin(2*e2)*sin(2*μ)* sin(e1)^3*sin(2*φ)*sin((t+phase)*ω) - 8*α1*cos(μ)^2*cos(e3)*sin(e2)^2* sin(e1)^4*sin(2*φ)*sin((t+phase)*ω) - 2*α1*cos(e3)*sin(e2)^2*sin(2*e1)^2* sin(2*φ)*sin((t+phase)*ω) - 2*α1*cos(μ)^2*cos(e3)*sin(e2)^2* sin(2*e1)^2*sin(2*φ)*sin((t+phase)*ω) + 8*α1*cos(e3)*sin(e2)^2*sin(μ)^2* sin(e1)^2*sin(2*φ)*sin(e3)^2* sin((t+phase)*ω) - 8*α1*cos(e3)^2*sin(e2)^2*sin(2*μ)* sin(e1)^3*sin(φ)^2*sin((t+phase)*ω)^2 + α1*(4* sin(e1)^2*(cos(2*μ)*sin(2*e2) + sin(e2)^2*sin(2*μ)*sin(e1)) + sin(2*e2)*sin(2*e1)^2)*sin(φ)^2* sin(2*e3)*sin(2*(t+phase)*ω) + 4*α1*cos(e1)*cos(e3 - (t+phase)*ω)*sin(e2)* sin(e1)*(sin(2*μ)*sin(e1)*sin(2*φ) - 4*sin(μ)^2*sin(φ)^2* sin(e3 - (t+phase)*ω)) + 4*α1*cos(e1)^4* (cos(2*(e3 - (t+phase)*ω))*sin(2*e2)* sin(φ)^2 + 2*sin(e2)^2*sin(2*φ)* sin(e3 - (t+phase)*ω)) + cos(e1)^2*(2*(-8*cos(φ)*sin(e2)* sin(μ) + 4*α1*cos(φ)^2*sin(e2)^2* sin(2*μ)* sin(e1) + α1*sin(2*e2)* sin(φ)^2*((cos(2*μ) - cos(2*e1))* cos(2*e3)* cos(2*(t+phase)*ω) - 2*sin(μ)^2*sin(2*e3)*sin(2*(t+phase)*ω))) - (16* sin(e2)*(α1*cos(φ)*sin(e2)* sin(μ)^2 - cos(μ)*sin(e1))* sin(φ) + α1*(8 + cos(2*e3))*sin(2*e2)* sin(2*μ)*sin(e1)*sin(2*φ))* sin(e3 - (t+phase)*ω) - 8*α1*sin(e2)^2*sin(2*μ)*sin(e1)* sin(φ)^2*sin(e3 - (t+phase)*ω)^2)) + cos(e2)*(8*α1*cos(e1)^4*sin(e2) - 32*α1*cos(φ)^2*cos(e3)^2*sin(e2)* sin(μ)^2* sin(e1)^2 + 32*cos(μ)*cos(φ)*sin(e1)^3 - 16*α1*cos(e1)^3*cos(e3 - (t+phase)*ω)*sin(e1)* sin(2*φ) - 2*α1*(sin(e2)^2*(cos(2*μ)* cos(e3 - (t+phase)*ω) + (cos(e3 - (t+phase)*ω) + 2*cos(e3 + (t+phase)*ω))*sin(μ)^2)* sin(2*e1) - cos(e3 - (t+phase)*ω)*sin(4*e1))* sin(2*φ) + α1* cos(μ)^2*(8*sin(e2)*sin(e1)^2 + (7 + cos(2*e2))*cos(e3 - (t+phase)*ω)* sin(2*e1)*sin(2*φ)) + 32*cos((t+phase)*ω)*sin(μ)*sin(e1)^2*sin(φ)* sin(e3) - 32*α1*sin(e2)*sin(μ)^2*sin(e1)^2* (cos(φ)^2 - cos((t+phase)*ω)^2*sin(φ)^2)*sin(e3)^2 - 8*sin(μ)*(4*cos(e3)*sin(e1)^2* sin(φ) + α1*sin(e2)^2*sin(μ)* sin(2*e1)*sin(2*φ)* sin(e3))*sin((t+phase)*ω) + 32*α1*cos(e3)^2*sin(e2)*sin(μ)^2* sin(e1)^2*sin(φ)^2* sin((t+phase)*ω)^2 - 4*cos(e1)^2*(-8*cos(μ)*cos(φ)*sin(e1) - cos((t+phase)*ω)*(8*sin(μ)*sin(φ) + α1* cos(2*e3)*sin(e2)*sin(2*μ)*sin(e1)* sin(2*φ))* sin(e3) + cos(e3)*(8*sin(μ)*sin(φ) + α1* cos(2*e3)*sin(e2)*sin(2*μ)*sin(e1)* sin(2*φ))*sin((t+phase)*ω)) - 8*α1*cos(e1)*sin(2*μ)*sin(φ)^2* sin(2*(e3 - (t+phase)*ω))))
  de3 =  (1/8)*(8*cos(e1)^2*cos(e3)*cos((t+phase)*ω)*sin(e2)* sin(μ)*sin(φ) + α1*cos(2*(e2 - μ))*cos(φ)* cos(e3 - (t+phase)*ω)*cot(e2)*sin(e1)^2* sin(φ) + α1*cos(2*(e2 + μ))*cos(φ)* cos(e3 - (t+phase)*ω)*cot(e2)*sin(e1)^2* sin(φ) + 8*cos(e3)*cos((t+phase)*ω)*sin(e2)*sin(μ)* sin(e1)^2*sin(φ) - 4*α1*cos(e1)^3*cos(e3)^2*cos(2*(t+phase)*ω)* sin(e2)*sin(2*μ)*sin(φ)^2 - 4*α1*cos(e1)*cos(e3)^2*cos(2*(t+phase)*ω)* sin(e2)*sin(2*μ)*sin(e1)^2*sin(φ)^2 - 2*α1*cos(e1)^2*cos(e3 - (t+phase)*ω)* sin(e2)^2*sin(2*μ)*sin(e1)*sin(2*φ) - α1*cos(e3 - (t+phase)*ω)*cot(e2)* sin(e1)^2*sin(2*φ) - α1*cos(2*e2)* cos(e3 - (t+phase)*ω)*cot(e2)* sin(e1)^2*sin(2*φ) - α1*cos(2*μ)* cos(e3 - (t+phase)*ω)*cot(e2)*sin(e1)^2* sin(2*φ) - 4*α1*cos(e1)^2*cos(e3 - (t+phase)*ω)* cot(e2)*sin(e1)^2*sin(2*φ) + 2*α1*cos(2*e1)*cos(e3 - (t+phase)*ω)* cot(e2)*sin(e1)^2*sin(2*φ) - 2*α1*cos(e3 - (t+phase)*ω)*sin(2*μ)* sin(e1)^3*sin(2*φ) + 8*α1*cos(e1)^3*cos(φ)*cos((t+phase)*ω)* sin(e2)*sin(e1)*sin(φ)*sin(e3) + 8*α1*cos(e1)*cos(φ)*cos((t+phase)*ω)* sin(e2)*sin(e1)^3* sin(φ)*sin(e3) + 8*α1*cos(e1)^2*cos(e3)*cos((t+phase)*ω)^2* sin(e2)^2*sin(μ)^2*sin(φ)^2* sin(e3) - 8*α1*cos(e1)^2*cos(e3)*cos((t+phase)*ω)^2* sin(e1)^2*sin(φ)^2*sin(e3) + 8*α1*cos(e3)*cos((t+phase)*ω)^2*sin(e2)^2* sin(μ)^2*sin(e1)^2*sin(φ)^2*sin(e3) - 8*α1*cos(e3)*cos((t+phase)*ω)^2*sin(e1)^4* sin(φ)^2*sin(e3) - 8*α1*cos(e1)^3*cos(φ)*cos(e3)* sin(e2)*sin(e1)*sin(φ)*sin((t+phase)*ω) - 8*α1*cos(e1)*cos(φ)*cos(e3)* sin(e2)*sin(e1)^3*sin(φ)*sin((t+phase)*ω) + 8*cos(e1)^2*sin(e2)*sin(μ)*sin(φ)* sin(e3)*sin((t+phase)*ω) + 8*sin(e2)*sin(μ)*sin(e1)^2*sin(φ)* sin(e3)*sin((t+phase)*ω) - 8*α1*cos(e1)^2*cos(e3)*sin(e2)^2* sin(μ)^2*sin(φ)^2*sin(e3)*sin((t+phase)*ω)^2 - 8*α1*cos(e3)*sin(e2)^2*sin(μ)^2* sin(e1)^2*sin(φ)^2*sin(e3)* sin((t+phase)*ω)^2 + 8*α1*cos(e3)*sin(e1)^4*sin(φ)^2* sin(e3)*sin((t+phase)*ω)^2 + α1*sin(2*e1)^2*sin(φ)^2*sin(2*e3)* sin((t+phase)*ω)^2 - 4*α1*cos(e1)^2*cos(e3)^2*sin(e2)^2* sin(μ)^2*sin(φ)^2*sin(2*(t+phase)*ω) + 4*α1*cos(e1)^2*cos(e3)^2*sin(e1)^2* sin(φ)^2* sin(2*(t+phase)*ω) - 4*α1*cos(e3)^2*sin(e2)^2*sin(μ)^2* sin(e1)^2*sin(φ)^2*sin(2*(t+phase)*ω) + 4*α1*cos(e3)^2*sin(e1)^4*sin(φ)^2* sin(2*(t+phase)*ω) + 4*α1*cos(e1)^2*sin(e2)^2*sin(μ)^2* sin(φ)^2*sin(e3)^2* sin(2*(t+phase)*ω) + α1*cos(e1)^2*sin(2*e2)* sin(2*μ)*sin(e1)*sin(φ)^2* sin(e3)^2*sin(2*(t+phase)*ω) + 4*α1*sin(e2)^2*sin(μ)^2*sin(e1)^2* sin(φ)^2*sin(e3)^2* sin(2*(t+phase)*ω) - 4*α1*sin(e1)^4*sin(φ)^2*sin(e3)^2* sin(2*(t+phase)*ω) + α1*cos(e1)*cot(e2)*sin(2*μ)* sin(2*e1)*sin(φ)^2*sin(e3)^2* sin(2*(t+phase)*ω) - α1*sin(2*e1)^2*sin(φ)^2*sin(e3)^2* sin(2*(t+phase)*ω) - 4*α1*cos(e1)^3*sin(e2)*sin(2*μ)* sin(φ)^2*sin(2*e3)*sin(2*(t+phase)*ω) - 4*α1*cos(e1)*sin(e2)*sin(2*μ)*sin(e1)^2* sin(φ)^2* sin(2*e3)*sin(2*(t+phase)*ω) + (1/4)* cos(e2)*(32*cos(e3)*cos((t+phase)*ω)*cot(e2)* sin(μ)*sin(e1)^2* sin(φ) + 4*α1* cos(e1)^3*(2*(1 + 3*cos(2*φ))*sin(e1) + (cos(2*(e3 + (t+phase)*ω))*cot(e2)* sin(2*μ) + cos(2*(e3 - (t+phase)*ω))*(-2*cot(e2)* sin(2*μ) + 4*sin(e1)))*sin(φ)^2) + 4*sin(e3)* (α1* cos((t+phase)*ω)^2*(-2*cos(2*μ)*sin(2*e1) + sin(4*e1))*sin(φ)^2*sin(e3) + 8*cot(e2)*sin(μ)*sin(e1)^2*sin(φ)* sin((t+phase)*ω)) + 32*cos(e1)^2*(cos(e3)*cos((t+phase)*ω)* sin(μ)*(cot(e2) + α1*cos(φ)* sin(e2)*sin(μ))* sin(φ) + sin(μ)*(cot(e2) + α1*cos(φ)* sin(e2)*sin(μ))*sin(φ)*sin(e3)* sin((t+phase)*ω)) + α1*(5*cos(3*e1)* sin(2*μ) - 8*cot(e2)*sin(2*e1))* sin(2*φ)* sin(e3 - (t+phase)*ω) + α1* cos(e1)*(4* sin(e1)*(8*cos(φ)^2*sin(e1)^2 + 5*cos((t+phase)*ω)*sin(2*μ)*sin(e1)* sin(2*φ)*sin(e3) - 8*cos((t+phase)*ω)^2*sin(μ)^2* sin(φ)^2*sin(e3)^2 + sin(e1)*((-cos(e3))* sin((t+phase)*ω)*(5*sin(2*μ)* sin(2*φ) + 8*cos(e3)*sin(e1)*sin(φ)^2* sin((t+phase)*ω)) + 4*((-cot(e2))*sin(2*μ) + sin(e1))* sin(φ)^2*sin(2*e3)* sin(2*(t+phase)*ω))) + 11*sin(2*μ)*sin(2*φ)* sin(e3 - (t+phase)*ω))) + (1/4)*α1*cot(e2)* sin(2*μ)*((5 + cos(2*e2))* sin(e1) + (-3 + cos(2*e2))*sin(3*e1))* sin(φ)^2*sin(2*(e3 - (t+phase)*ω)) + 2*α1*cos(e2)^2* (cos(e3 - (t+phase)*ω)*(2*cot(e2)*sin(μ)^2 - cos(e1)^2*sin(2*μ)*sin(e1))*sin(2*φ) + 2*sin(μ)^2*sin(φ)^2* sin(2*(e3 - (t+phase)*ω))) - 2*α1*cos(μ)^2* (sin(e2)*sin(2*e1)*sin(2*φ)* sin(e3 - (t+phase)*ω) + cos(e2)*(4*cos(e1)*cos(φ)^2*sin(e1) - sin(2*e1)*sin(e3 - (t+phase)*ω)* (cot(e2)*sin(2*φ) + 2*sin(φ)^2*sin(e3 - (t+phase)*ω))) + 2*cos(e1)^2*(cos(e3 - (t+phase)*ω)*cot(e2)* sin(2*φ) + sin(φ)^2*sin(2*(e3 - (t+phase)*ω)))) + cos(μ)* cos(e1)*(2* sin(e1)^2*(-4*cos(φ)*cot(e2) + 4*α1*cos(e2)*cos(φ)^2*cot(e2)* sin(μ) + sin(φ)*(α1*sin(μ)* sin(φ)*(-((-1 + cos(2*e2) + 2*cos(2*e3))* cos(2*(t+phase)*ω)*csc(e2)) + 2*cos(e2)* cot(e2)*(-1 + sin(2*e3)*sin(2*(t+phase)*ω))) - 4*sin(e3 - (t+phase)*ω))) + cos(e1)^2*(-8*cos(φ)*cot(e2) + 8*α1*cos(e2)*cos(φ)^2* cot(e2)*sin(μ) - 2*α1*sin(μ)* sin(φ)^2*(cos(e2)*(2 + cos(2*(e3 + (t+phase)*ω)))* cot(e2) - 4*cos(2*(t+phase)*ω)*sin(e2)*sin(e3)^2) - 8*sin(φ)*sin(e3 - (t+phase)*ω)) - α1*(-3 + cos(2*e2))*cos(e1)*cos(e3)* cot(e2)*sin(μ)*sin(e1)*sin(φ)^2* (3*sin(e3 - 2*(t+phase)*ω) + sin(e3 + 2*(t+phase)*ω))))
end φ ω a μ mg α1 Φg Θg phase
#     >-###'>           pg12         <'###-<
model_pg12 = @ode_def begin
  de1 = (1/(4*a^2))*(3*cos(e1)* cos(φ)*(1 - a^2 + a^2*csc(e2)^2)* sin(e2) - 3*cos(e3 - (t+phase)*ω)*sin(e1)*sin(φ) + 4*a^2*mg*cot(e2)* (cos(Θg)*cos(e3)*sin(Φg) - sin(Θg)*sin(e3)) + cos(e2)*(cos(e1)*(a^2*cos(φ)* cot(e2) + (-3 + 4*a^2)*sin(φ)* sin(e3 - (t+phase)*ω)) + 2*a^2*α1*(cos(e3 - (t+phase)*ω)*cot(e2)* sin(2*φ) + sin(φ)^2*sin(2*(e3 - (t+phase)*ω)))))
  de2 =  (1/8)*α1*(2*cos(e3 - (t+phase)*ω)^2 - cos(2*φ)*(-3 + cos(2*(e3 - (t+phase)*ω))))* sin(2*e2) + cos(e2)*((-mg)*cos(e3)*sin(Θg) + cos(φ)*sin(e1)) + mg*cos(Θg)*(cos(Φg)*sin(e2) - cos(e2)*sin(Φg)*sin(e3)) + (1/2)*(2*sin(e2)*sin(e1)*sin(φ) - α1* cos(2*e2)*sin(2*φ))*sin(e3 - (t+phase)*ω)
  de3 =  (1/2)*(2*mg*csc(e2)*sin(Θg)*sin(e3) - cos(e3)*(2*mg*cos(Θg)*csc(e2)* sin(Φg) + α1* cos((t+phase)*ω)*(cot(e2)*sin(2*φ) + 2*cos((t+phase)*ω)*sin(φ)^2*sin(e3))) - α1*cot(e2)*sin(2*φ)*sin(e3)* sin((t+phase)*ω) + α1*sin(φ)^2*sin(2*e3)* sin((t+phase)*ω)^2 + α1*cos(2*e3)*sin(φ)^2*sin(2*(t+phase)*ω) - 2* cos(e1)*(cos(φ)*cot(e2) + sin(φ)*sin(e3 - (t+phase)*ω)))
end φ ω a μ mg α1 Φg Θg phase
#     >-###'>           ag12         <'###-<
model_ag12 = @ode_def begin
  de1 = (1/(4*a^2))*(3* cos(μ)*(cos(e1)* cos(φ)*(1 - a^2 + a^2*csc(e2)^2)* sin(e2) - cos(e3 - (t+phase)*ω)*sin(e1)*sin(φ)) - 4*a^2*cot(e2)* (cos(e3)*cos((t+phase)*ω)*sin(μ)*sin(φ) - mg*cos(Θg)*cos(e3)*sin(Φg) + mg*sin(Θg)*sin(e3) + sin(μ)*sin(φ)*sin(e3)*sin((t+phase)*ω)) + cos(e2)*cos(μ)* cos(e1)*(a^2*cos(φ)*cot(e2) + (-3 + 4*a^2)* sin(φ)* sin(e3 - (t+phase)*ω)))
  de2 =  mg*cos(Θg)*(cos(Φg)*sin(e2) - cos(e2)*sin(Φg)*sin(e3)) + cos(e2)*(cos(μ)*cos(φ)*sin(e1) + cos((t+phase)*ω)*sin(μ)*sin(φ)*sin(e3) - cos(e3)*(mg*sin(Θg) + sin(μ)*sin(φ)*sin((t+phase)*ω))) + sin(e2)*((-cos(φ))*sin(μ) + cos(μ)*sin(e1)*sin(φ)* sin(e3 - (t+phase)*ω))
  de3 =  csc(e2)*(cos(e3)*(cos((t+phase)*ω)*sin(μ)* sin(φ) - mg*cos(Θg)*sin(Φg)) + sin(e3)*(mg*sin(Θg) + sin(μ)*sin(φ)*sin((t+phase)*ω))) - cos(μ)* cos(e1)*(cos(φ)*cot(e2) + sin(φ)*sin(e3 - (t+phase)*ω))
end φ ω a μ mg α1 Φg Θg phase
#     >-###'>           apg         <'###-<
model_apg = @ode_def begin
  de1 = (1/(32*a^2))*(-6*α1*cos(e1)*cos(φ)^2* sin(2*e2)*sin(2*μ) - 9*α1*cos(φ)^2*sin(e2)^2* sin(2*e1) + 3*α1*cos(2*μ)*cos(φ)^2*sin(e2)^2* sin(2*e1) - 16*a^2*cos(e1)^2*cos(e3)*cos((t+phase)*ω)* sin(2*e2)*sin(μ)*sin(φ) - 16*a^2*cos(e3)*cos((t+phase)*ω)*sin(2*e2)*sin(μ)* sin(e1)^2*sin(φ) + 8*a^2*α1*cos(e1)^3*cos(e3)^2*cos((t+phase)*ω)^2* sin(2*e2)*sin(2*μ)*sin(φ)^2 + 8*a^2*α1*cos(e1)*cos(e3)^2*cos((t+phase)*ω)^2* sin(2*e2)*sin(2*μ)*sin(e1)^2*sin(φ)^2 + 9*α1*cos(e3 - (t+phase)*ω)^2*sin(2*e1)* sin(φ)^2 - 3*α1*cos(2*μ)*cos(e3 - (t+phase)*ω)^2* sin(2*e1)*sin(φ)^2 + 24*cos(μ)*(cos(e1)*cos(φ)*sin(e2) - cos(e3 - (t+phase)*ω)*sin(e1)*sin(φ)) - 9*α1*cos(e1)^2*cos(e3 - (t+phase)*ω)*sin(e2)* sin(2*φ) + 3*α1*cos(2*μ)*cos(e1)^2* cos(e3 - (t+phase)*ω)*sin(e2)*sin(2*φ) + 9*α1*cos(e3 - (t+phase)*ω)*sin(e2)* sin(e1)^2*sin(2*φ) - 3*α1*cos(2*μ)*cos(e3 - (t+phase)*ω)* sin(e2)*sin(e1)^2*sin(2*φ) + 6*α1*cos(e3 - 2*(t+phase)*ω)*sin(e2)*sin(2*μ)* sin(e1)* sin(φ)^2*sin(e3) - 6*α1*cos(e1)*cos((t+phase)*ω)*sin(e2)^2* sin(2*μ)*sin(2*φ)*sin(e3) - 8*a^2*α1*cos(e1)^3*cos((t+phase)*ω)*sin(2*e2)* sin(e1)*sin(2*φ)*sin(e3) - 8*a^2*α1*cos(e1)*cos((t+phase)*ω)*sin(2*e2)* sin(e1)^3*sin(2*φ)*sin(e3) - 8*a^2*α1*cos(e1)^3*cos((t+phase)*ω)^2* sin(2*e2)*sin(2*μ)*sin(φ)^2*sin(e3)^2 - 8*a^2*α1*cos(e1)*cos((t+phase)*ω)^2*sin(2*e2)* sin(2*μ)*sin(e1)^2*sin(φ)^2*sin(e3)^2 + 3*α1*sin(e2)*sin(2*μ)*sin(e1)* sin(φ)^2*sin(2*e3) - 12*α1*cos(e3)*cos(e3 - (t+phase)*ω)* sin(e2)*sin(2*μ)*sin(e1)*sin(φ)^2* sin((t+phase)*ω) + 6*α1*cos(e1)*cos(e3)*sin(e2)^2* sin(2*μ)*sin(2*φ)*sin((t+phase)*ω) + 8*a^2*α1*cos(e1)^3*cos(e3)*sin(2*e2)* sin(e1)* sin(2*φ)*sin((t+phase)*ω) + 8*a^2*α1*cos(e1)*cos(e3)*sin(2*e2)* sin(e1)^3*sin(2*φ)* sin((t+phase)*ω) - 16*a^2*cos(e1)^2*sin(2*e2)*sin(μ)*sin(φ)* sin(e3)*sin((t+phase)*ω) - 16*a^2*sin(2*e2)*sin(μ)*sin(e1)^2* sin(φ)*sin(e3)*sin((t+phase)*ω) - 8*a^2*α1*cos(e1)^3*cos(e3)^2*sin(2*e2)* sin(2*μ)*sin(φ)^2*sin((t+phase)*ω)^2 - 8*a^2*α1*cos(e1)*cos(e3)^2*sin(2*e2)* sin(2*μ)*sin(e1)^2*sin(φ)^2*sin((t+phase)*ω)^2 + 8*a^2*α1*cos(e1)^3*sin(2*e2)*sin(2*μ)* sin(φ)^2*sin(e3)^2*sin((t+phase)*ω)^2 + 8*a^2*α1*cos(e1)*sin(2*e2)*sin(2*μ)* sin(e1)^2*sin(φ)^2*sin(e3)^2* sin((t+phase)*ω)^2 + 8*a^2*α1*sin(e2)*sin(2*e2)*sin(μ)^2* sin(e1)^2*sin(φ)^2*sin(2*e3)* sin((t+phase)*ω)^2 + 8*a^2*α1*sin(2*e2)* sin(φ)^2*((-sin(e2))*sin(μ)^2*sin(e1)^2* sin(e3)^2 + cos(e1)*sin(2*μ)*sin(2*e3))* sin(2*(t+phase)*ω) + α1* cos(μ)^2*(6* sin(2*e1)*(cos(φ)^2*sin(e2)^2 - cos(e3 - (t+phase)*ω)^2*sin(φ)^2) + 6*cos(2*e1)*cos(e3 - (t+phase)*ω)*sin(e2)* sin(2*φ) + (-3 + 4*a^2)*sin(2*e2)* sin(2*e1)* sin(2*φ)*sin(e3 - (t+phase)*ω)) - cos(e2)^2*(32*a^2*cos(e3)*cos((t+phase)*ω)*cot(e2)* sin(μ)* sin(e1)^2*sin(φ) - 8*a^2*α1*cos(e3)^2*cos((t+phase)*ω)*cot(e2)* sin(2*e1)*sin(2*φ)* sin(e3) + 8*a^2*α1*cos(μ)*cos(e1)*sin(μ)* (cos(e1)^2* cot(e2)*(4*cos(φ)^2 - cos(2*(e3 + (t+phase)*ω))*sin(φ)^2) + 2*cos(φ)*cos(2*e3)*cos((t+phase)*ω)* sin(e1)^2*sin(φ)*sin(e3)) + 8*a^2*α1*cos(e3)^3*cot(e2)*sin(2*e1)* sin(2*φ)*sin((t+phase)*ω) + 32*a^2*cot(e2)*sin(μ)*sin(e1)^2*sin(φ)* sin(e3)*sin((t+phase)*ω) + 8*a^2*cos(e1)^2*(2*cos(e3)*cos((t+phase)*ω)* sin(μ)*(2*cot(e2)*sin(φ) + α1*sin(e2)*sin(μ)* sin(2*φ)) + sin(e3)*(4*cot(e2)*sin(μ)*sin(φ)* sin((t+phase)*ω) + α1* sin(e2)*(2*sin(μ)^2*sin(2*φ)* sin((t+phase)*ω) + sin(2*μ)*sin(e1)*sin(φ)^2* sin(e3)*sin(2*(t+phase)*ω)))) + 9*α1*sin(2*e1)*sin(φ)^2* sin(e3 - (t+phase)*ω)^2 - 3*α1*cos(2*μ)*sin(2*e1)*sin(φ)^2* sin(e3 - (t+phase)*ω)^2 + 2*α1* cos(e1)*(a^2* sin(e1)^2*(2* cos(2*φ)*((3 + cos(2*(e3 - (t+phase)*ω)))* cot(e2)* sin(2*μ) + 6*sin(e1)) + sin(e1)*(4 + 8*cos(2*(e3 - (t+phase)*ω))*sin(φ)^2) + sin(2*μ)* sin(2*φ)*((-cos((t+phase)*ω))*(-9* sin(e3) + sin(3*e3)) - 8*cos(e3)*sin((t+phase)*ω))) - sin(2*φ)*(3*sin(2*μ) + 8*a^2*cot(e2)*sin(e1)^3* sin(e3)^2)*sin(e3 - (t+phase)*ω) + 4*a^2*cot(e2)*sin(2*μ)*sin(e1)^2* sin(e3 - (t+phase)*ω)^2) + 2*α1*cos(μ)^2*sin(2*e1)* (2* a^2*(-2*cos(e3 - (t+phase)*ω)^2 + cos(2*φ)*(-3 + cos(2*(e3 - (t+phase)*ω)))) + 4*a^2*cot(e2)*sin(2*φ)* sin(e3 - (t+phase)*ω) - 3*sin(φ)^2*sin(e3 - (t+phase)*ω)^2) + 4*a^2*α1* cos(e1)^3*(8*cos(φ)^2* sin(e1) + (-2 - 2*cos(2*(e3 - (t+phase)*ω)) + cos(2*(e3 + (t+phase)*ω)))*cot(e2)* sin(2*μ)*sin(φ)^2 + 4*sin(2*φ)*(sin(2*μ) - cot(e2)*sin(e1)*sin(e3)^2)* sin(e3 - (t+phase)*ω) - 8*sin(e1)*sin(φ)^2* sin(e3 - (t+phase)*ω)^2) + 8*a^2*α1*sin(e2)*sin(2*μ)*sin(e1)^3* sin(φ)^2*sin(2*(e3 - (t+phase)*ω))) + 8*a^2*α1*cos(e2)^3* (cos(e3 - (t+phase)*ω)*(-2*cot(e2)*sin(μ)^2 + sin(2*μ)*sin(e1))*sin(2*φ) - (2*sin(μ)^2 + cot(e2)*sin(2*μ)*sin(e1))*sin(φ)^2* sin(2*(e3 - (t+phase)*ω))) + cos(e2)*(8*cos(μ)* cos(e1)*(4*a^2*cos(φ)*cot(e2) + sin(φ)*(-3 + 4*a^2 + 3*α1*cos((t+phase)*ω)*sin(e2)*sin(μ)* sin(φ)*sin(e3))* sin(e3 - (t+phase)*ω)) + 2*α1* cos(μ)^2*(8*a^2*cos(e1)^2*cos(e3 - (t+phase)*ω)* cot(e2)* sin(2*φ) + (4*a^2 + (-3 + 4*a^2)*cos(2*e1))* sin(φ)^2*sin(2*(e3 - (t+phase)*ω))) + α1*(-4*a^2*cos(2*(e2 + μ))* cos(φ)*cos(e3 - (t+phase)*ω)*cot(e2)* sin(e1)^2*sin(φ) + 8*a^2*cos(e1)^2*cos(e3)*cos((t+phase)*ω)* sin(e2)^2*sin(2*μ)*sin(e1)* sin(2*φ) + 8*a^2*cos(e3)*cos((t+phase)*ω)*sin(e2)^2* sin(2*μ)*sin(e1)^3*sin(2*φ) + 2*cos(e3 - (t+phase)*ω)* sin(e1)*(3*sin(2*μ) + a^2*(6 + 2*cos(2*e2) - cos(2*(e2 - μ)) + 2*cos(2*μ))*cot(e2)* sin(e1))*sin(2*φ) - 16*a^2*cos(e1)^2*cos((t+phase)*ω)^2* sin(e2)^2*sin(μ)^2*sin(φ)^2* sin(2*e3) - 16*a^2*cos((t+phase)*ω)^2*sin(e2)^2*sin(μ)^2* sin(e1)^2*sin(φ)^2*sin(2*e3) + 16*a^2*cos((t+phase)*ω)^2*sin(e1)^4*sin(φ)^2* sin(2*e3) + 4*a^2*cos((t+phase)*ω)^2*sin(2*e1)^2*sin(φ)^2* sin(2*e3) + 8*a^2*cos(e1)^2*sin(e2)^2*sin(2*μ)* sin(e1)*sin(2*φ)*sin(e3)* sin((t+phase)*ω) + 8*a^2*sin(e2)^2*sin(2*μ)*sin(e1)^3* sin(2*φ)*sin(e3)*sin((t+phase)*ω) + 16*a^2*cos(e1)^2*sin(e2)^2*sin(μ)^2* sin(φ)^2*sin(2*e3)*sin((t+phase)*ω)^2 - 16*a^2*sin(e1)^4*sin(φ)^2*sin(2*e3)* sin((t+phase)*ω)^2 - 4*a^2*sin(2*e1)^2*sin(φ)^2* sin(2*e3)*sin((t+phase)*ω)^2 + 16*a^2*cos(e1)^2*cos(e3)^2*sin(e2)^2* sin(μ)^2*sin(φ)^2* sin(2*(t+phase)*ω) + 16*a^2*cos(e3)^2*sin(e2)^2*sin(μ)^2* sin(e1)^2*sin(φ)^2* sin(2*(t+phase)*ω) - 16*a^2*cos(e3)^2*sin(e1)^4*sin(φ)^2* sin(2*(t+phase)*ω) - 4*a^2*cos(e3)^2*sin(2*e1)^2*sin(φ)^2* sin(2*(t+phase)*ω) - 16*a^2*cos(e1)^2*sin(e2)^2* sin(μ)^2*sin(φ)^2*sin(e3)^2* sin(2*(t+phase)*ω) - 8*a^2*cos(e1)^2*cot(e2)*sin(2*μ)* sin(e1)*sin(φ)^2*sin(e3)^2* sin(2*(t+phase)*ω) + 16*a^2*sin(e1)^4*sin(φ)^2*sin(e3)^2* sin(2*(t+phase)*ω) + 4*a^2*sin(2*e1)^2*sin(φ)^2*sin(e3)^2* sin(2*(t+phase)*ω) - 18*a^2*cos(e1)^2*cos(e3)*cot(e2)*sin(2*μ)* sin(e1)*sin(φ)^2*sin(e3 - 2*(t+phase)*ω) + 6*a^2*cos(2*e2)*cos(e1)^2*cos(e3)* cot(e2)*sin(2*μ)*sin(e1)*sin(φ)^2* sin(e3 - 2*(t+phase)*ω) + 9*sin(e2)*sin(2*e1)*sin(2*φ)* sin(e3 - (t+phase)*ω) - 3*cos(2*μ)*sin(e2)*sin(2*e1)* sin(2*φ)*sin(e3 - (t+phase)*ω) - 12*cos(e1)*cos(e3)*sin(e2)*sin(2*μ)* sin(φ)^2*sin((t+phase)*ω)* sin(e3 - (t+phase)*ω) + 9*cos(e1)^2*sin(φ)^2* sin(2*(e3 - (t+phase)*ω)) - 3*cos(2*μ)*cos(e1)^2*sin(φ)^2* sin(2*(e3 - (t+phase)*ω)) + 8*a^2*cos(e1)^2*cot(e2)*sin(2*μ)*sin(e1)* sin(φ)^2* sin(2*(e3 - (t+phase)*ω)) - 9*sin(e1)^2*sin(φ)^2* sin(2*(e3 - (t+phase)*ω)) + 3*cos(2*μ)*sin(e1)^2*sin(φ)^2* sin(2*(e3 - (t+phase)*ω)) + 2*a^2*(-3 + cos(2*e2))* cos(e1)^2*cos(e3)*cot(e2)* sin(2*μ)*sin(e1)*sin(φ)^2* sin(e3 + 2*(t+phase)*ω))))
  de2 =  (1/32)*(-4*α1*cos(e2)^3*cos(e3 - (t+phase)*ω)* sin(2*e1)*sin(2*φ) + 4*α1* cos(e2)^2*((-2*cos(e3 - (t+phase)*ω)^2 + cos(2*φ)*(-3 + cos(2*(e3 - (t+phase)*ω))))* sin(2*μ)* sin(e1) - (1 + 3*cos(2*μ) + 2*cos(2*e1)*sin(μ)^2)*sin(2*φ)* sin(e3 - (t+phase)*ω)) + 2*(16*mg*sin(e2) + α1* cos(e1)^2*(cos(2*μ) - cos(2*e1) + 6*cos(μ)^2*cos(2*φ))* sin(2*e2) + 6*α1*cos(μ)^2*cos(2*φ)*sin(2*e2)* sin(e1)^2 - 16*cos(φ)*cos(e3)^2*sin(e2)*sin(μ)* sin(e1)^2 + 8*α1*cos(φ)^2*cos(e3)^2*sin(e2)^2* sin(2*μ)*sin(e1)^3 + 2*α1*cos(μ)^2*cos(2*(e3 - (t+phase)*ω))* sin(2*e2)*sin(e1)^2* sin(φ)^2 + 2*α1*cos(μ)^2*cos(2*(e3 + (t+phase)*ω))* sin(2*e2)*sin(e1)^2*sin(φ)^2 + 4*α1*cos(e1)^3*cos(e3 - (t+phase)*ω)* sin(e2)*sin(2*μ)*sin(2*φ) + 16*cos(μ)*cos((t+phase)*ω)*sin(e2)*sin(e1)^3* sin(φ)*sin(e3) - 8*α1*cos(e3)^2*cos((t+phase)*ω)*sin(e2)^2* sin(μ)^2*sin(e1)^2*sin(2*φ)*sin(e3) - 8*α1*cos((t+phase)*ω)*sin(2*e2)*sin(2*μ)* sin(e1)^3*sin(2*φ)*sin(e3) + 8*α1*cos(μ)^2*cos((t+phase)*ω)*sin(e2)^2* sin(e1)^4*sin(2*φ)*sin(e3) + 2*α1*cos((t+phase)*ω)*sin(e2)^2*sin(2*e1)^2* sin(2*φ)*sin(e3) + 2*α1*cos(μ)^2*cos((t+phase)*ω)*sin(e2)^2* sin(2*e1)^2*sin(2*φ)*sin(e3) - 16*cos(φ)*sin(e2)*sin(μ)*sin(e1)^2* sin(e3)^2 + 8*α1*cos(φ)^2*sin(e2)^2*sin(2*μ)* sin(e1)^3*sin(e3)^2 - 8*α1*cos((t+phase)*ω)^2*sin(e2)^2*sin(2*μ)* sin(e1)^3*sin(φ)^2* sin(e3)^2 - 8*α1*cos((t+phase)*ω)*sin(e2)^2*sin(μ)^2* sin(e1)^2*sin(2*φ)*sin(e3)^3 - 16*cos(μ)*cos(e3)*sin(e2)*sin(e1)^3* sin(φ)*sin((t+phase)*ω) + 8*α1*cos(e3)^3*sin(e2)^2*sin(μ)^2* sin(e1)^2*sin(2*φ)*sin((t+phase)*ω) + 8*α1*cos(e3)*sin(2*e2)*sin(2*μ)* sin(e1)^3*sin(2*φ)*sin((t+phase)*ω) - 8*α1*cos(μ)^2*cos(e3)*sin(e2)^2* sin(e1)^4*sin(2*φ)*sin((t+phase)*ω) - 2*α1*cos(e3)*sin(e2)^2*sin(2*e1)^2* sin(2*φ)*sin((t+phase)*ω) - 2*α1*cos(μ)^2*cos(e3)*sin(e2)^2* sin(2*e1)^2*sin(2*φ)*sin((t+phase)*ω) + 8*α1*cos(e3)*sin(e2)^2*sin(μ)^2* sin(e1)^2*sin(2*φ)*sin(e3)^2* sin((t+phase)*ω) - 8*α1*cos(e3)^2*sin(e2)^2*sin(2*μ)* sin(e1)^3*sin(φ)^2*sin((t+phase)*ω)^2 + α1*(4* sin(e1)^2*(cos(2*μ)*sin(2*e2) + sin(e2)^2*sin(2*μ)*sin(e1)) + sin(2*e2)*sin(2*e1)^2)*sin(φ)^2* sin(2*e3)*sin(2*(t+phase)*ω) + 4*α1*cos(e1)*cos(e3 - (t+phase)*ω)*sin(e2)* sin(e1)*(sin(2*μ)*sin(e1)*sin(2*φ) - 4*sin(μ)^2*sin(φ)^2* sin(e3 - (t+phase)*ω)) + 4*α1*cos(e1)^4* (cos(2*(e3 - (t+phase)*ω))*sin(2*e2)* sin(φ)^2 + 2*sin(e2)^2*sin(2*φ)* sin(e3 - (t+phase)*ω)) + cos(e1)^2*(2*(-8*cos(φ)*sin(e2)* sin(μ) + 4*α1*cos(φ)^2*sin(e2)^2* sin(2*μ)* sin(e1) + α1*sin(2*e2)* sin(φ)^2*((cos(2*μ) - cos(2*e1))* cos(2*e3)* cos(2*(t+phase)*ω) - 2*sin(μ)^2*sin(2*e3)*sin(2*(t+phase)*ω))) - (16* sin(e2)*(α1*cos(φ)*sin(e2)* sin(μ)^2 - cos(μ)*sin(e1))* sin(φ) + α1*(8 + cos(2*e3))*sin(2*e2)* sin(2*μ)*sin(e1)*sin(2*φ))* sin(e3 - (t+phase)*ω) - 8*α1*sin(e2)^2*sin(2*μ)*sin(e1)* sin(φ)^2*sin(e3 - (t+phase)*ω)^2)) + cos(e2)*(8*α1*cos(e1)^4*sin(e2) - 32*α1*cos(φ)^2*cos(e3)^2*sin(e2)* sin(μ)^2* sin(e1)^2 + 32*cos(μ)*cos(φ)*sin(e1)^3 - 16*α1*cos(e1)^3*cos(e3 - (t+phase)*ω)*sin(e1)* sin(2*φ) - 2*α1*(sin(e2)^2*(cos(2*μ)* cos(e3 - (t+phase)*ω) + (cos(e3 - (t+phase)*ω) + 2*cos(e3 + (t+phase)*ω))*sin(μ)^2)* sin(2*e1) - cos(e3 - (t+phase)*ω)*sin(4*e1))* sin(2*φ) + α1* cos(μ)^2*(8*sin(e2)*sin(e1)^2 + (7 + cos(2*e2))*cos(e3 - (t+phase)*ω)* sin(2*e1)*sin(2*φ)) + 32*cos((t+phase)*ω)*sin(μ)*sin(e1)^2*sin(φ)* sin(e3) - 32*α1*sin(e2)*sin(μ)^2*sin(e1)^2* (cos(φ)^2 - cos((t+phase)*ω)^2*sin(φ)^2)*sin(e3)^2 - 8*sin(μ)*(4*cos(e3)*sin(e1)^2* sin(φ) + α1*sin(e2)^2*sin(μ)* sin(2*e1)*sin(2*φ)* sin(e3))*sin((t+phase)*ω) + 32*α1*cos(e3)^2*sin(e2)*sin(μ)^2* sin(e1)^2*sin(φ)^2* sin((t+phase)*ω)^2 - 4*cos(e1)^2*(-8*cos(μ)*cos(φ)*sin(e1) - cos((t+phase)*ω)*(8*sin(μ)*sin(φ) + α1* cos(2*e3)*sin(e2)*sin(2*μ)*sin(e1)* sin(2*φ))* sin(e3) + cos(e3)*(8*sin(μ)*sin(φ) + α1* cos(2*e3)*sin(e2)*sin(2*μ)*sin(e1)* sin(2*φ))*sin((t+phase)*ω)) - 8*α1*cos(e1)*sin(2*μ)*sin(φ)^2* sin(2*(e3 - (t+phase)*ω))))
  de3 =  (1/8)*(8*cos(e1)^2*cos(e3)*cos((t+phase)*ω)*sin(e2)* sin(μ)*sin(φ) + α1*cos(2*(e2 - μ))*cos(φ)* cos(e3 - (t+phase)*ω)*cot(e2)*sin(e1)^2* sin(φ) + α1*cos(2*(e2 + μ))*cos(φ)* cos(e3 - (t+phase)*ω)*cot(e2)*sin(e1)^2* sin(φ) + 8*cos(e3)*cos((t+phase)*ω)*sin(e2)*sin(μ)* sin(e1)^2*sin(φ) - 4*α1*cos(e1)^3*cos(e3)^2*cos(2*(t+phase)*ω)* sin(e2)*sin(2*μ)*sin(φ)^2 - 4*α1*cos(e1)*cos(e3)^2*cos(2*(t+phase)*ω)* sin(e2)*sin(2*μ)*sin(e1)^2*sin(φ)^2 - 2*α1*cos(e1)^2*cos(e3 - (t+phase)*ω)* sin(e2)^2*sin(2*μ)*sin(e1)*sin(2*φ) - α1*cos(e3 - (t+phase)*ω)*cot(e2)* sin(e1)^2*sin(2*φ) - α1*cos(2*e2)* cos(e3 - (t+phase)*ω)*cot(e2)* sin(e1)^2*sin(2*φ) - α1*cos(2*μ)* cos(e3 - (t+phase)*ω)*cot(e2)*sin(e1)^2* sin(2*φ) - 4*α1*cos(e1)^2*cos(e3 - (t+phase)*ω)* cot(e2)*sin(e1)^2*sin(2*φ) + 2*α1*cos(2*e1)*cos(e3 - (t+phase)*ω)* cot(e2)*sin(e1)^2*sin(2*φ) - 2*α1*cos(e3 - (t+phase)*ω)*sin(2*μ)* sin(e1)^3*sin(2*φ) + 8*α1*cos(e1)^3*cos(φ)*cos((t+phase)*ω)* sin(e2)*sin(e1)*sin(φ)*sin(e3) + 8*α1*cos(e1)*cos(φ)*cos((t+phase)*ω)* sin(e2)*sin(e1)^3* sin(φ)*sin(e3) + 8*α1*cos(e1)^2*cos(e3)*cos((t+phase)*ω)^2* sin(e2)^2*sin(μ)^2*sin(φ)^2* sin(e3) - 8*α1*cos(e1)^2*cos(e3)*cos((t+phase)*ω)^2* sin(e1)^2*sin(φ)^2*sin(e3) + 8*α1*cos(e3)*cos((t+phase)*ω)^2*sin(e2)^2* sin(μ)^2*sin(e1)^2*sin(φ)^2*sin(e3) - 8*α1*cos(e3)*cos((t+phase)*ω)^2*sin(e1)^4* sin(φ)^2*sin(e3) - 8*α1*cos(e1)^3*cos(φ)*cos(e3)* sin(e2)*sin(e1)*sin(φ)*sin((t+phase)*ω) - 8*α1*cos(e1)*cos(φ)*cos(e3)* sin(e2)*sin(e1)^3*sin(φ)*sin((t+phase)*ω) + 8*cos(e1)^2*sin(e2)*sin(μ)*sin(φ)* sin(e3)*sin((t+phase)*ω) + 8*sin(e2)*sin(μ)*sin(e1)^2*sin(φ)* sin(e3)*sin((t+phase)*ω) - 8*α1*cos(e1)^2*cos(e3)*sin(e2)^2* sin(μ)^2*sin(φ)^2*sin(e3)*sin((t+phase)*ω)^2 - 8*α1*cos(e3)*sin(e2)^2*sin(μ)^2* sin(e1)^2*sin(φ)^2*sin(e3)* sin((t+phase)*ω)^2 + 8*α1*cos(e3)*sin(e1)^4*sin(φ)^2* sin(e3)*sin((t+phase)*ω)^2 + α1*sin(2*e1)^2*sin(φ)^2*sin(2*e3)* sin((t+phase)*ω)^2 - 4*α1*cos(e1)^2*cos(e3)^2*sin(e2)^2* sin(μ)^2*sin(φ)^2*sin(2*(t+phase)*ω) + 4*α1*cos(e1)^2*cos(e3)^2*sin(e1)^2* sin(φ)^2* sin(2*(t+phase)*ω) - 4*α1*cos(e3)^2*sin(e2)^2*sin(μ)^2* sin(e1)^2*sin(φ)^2*sin(2*(t+phase)*ω) + 4*α1*cos(e3)^2*sin(e1)^4*sin(φ)^2* sin(2*(t+phase)*ω) + 4*α1*cos(e1)^2*sin(e2)^2*sin(μ)^2* sin(φ)^2*sin(e3)^2* sin(2*(t+phase)*ω) + α1*cos(e1)^2*sin(2*e2)* sin(2*μ)*sin(e1)*sin(φ)^2* sin(e3)^2*sin(2*(t+phase)*ω) + 4*α1*sin(e2)^2*sin(μ)^2*sin(e1)^2* sin(φ)^2*sin(e3)^2* sin(2*(t+phase)*ω) - 4*α1*sin(e1)^4*sin(φ)^2*sin(e3)^2* sin(2*(t+phase)*ω) + α1*cos(e1)*cot(e2)*sin(2*μ)* sin(2*e1)*sin(φ)^2*sin(e3)^2* sin(2*(t+phase)*ω) - α1*sin(2*e1)^2*sin(φ)^2*sin(e3)^2* sin(2*(t+phase)*ω) - 4*α1*cos(e1)^3*sin(e2)*sin(2*μ)* sin(φ)^2*sin(2*e3)*sin(2*(t+phase)*ω) - 4*α1*cos(e1)*sin(e2)*sin(2*μ)*sin(e1)^2* sin(φ)^2* sin(2*e3)*sin(2*(t+phase)*ω) + (1/4)* cos(e2)*(32*cos(e3)*cos((t+phase)*ω)*cot(e2)* sin(μ)*sin(e1)^2* sin(φ) + 4*α1* cos(e1)^3*(2*(1 + 3*cos(2*φ))*sin(e1) + (cos(2*(e3 + (t+phase)*ω))*cot(e2)* sin(2*μ) + cos(2*(e3 - (t+phase)*ω))*(-2*cot(e2)* sin(2*μ) + 4*sin(e1)))*sin(φ)^2) + 4*sin(e3)* (α1* cos((t+phase)*ω)^2*(-2*cos(2*μ)*sin(2*e1) + sin(4*e1))*sin(φ)^2*sin(e3) + 8*cot(e2)*sin(μ)*sin(e1)^2*sin(φ)* sin((t+phase)*ω)) + 32*cos(e1)^2*(cos(e3)*cos((t+phase)*ω)* sin(μ)*(cot(e2) + α1*cos(φ)* sin(e2)*sin(μ))* sin(φ) + sin(μ)*(cot(e2) + α1*cos(φ)* sin(e2)*sin(μ))*sin(φ)*sin(e3)* sin((t+phase)*ω)) + α1*(5*cos(3*e1)* sin(2*μ) - 8*cot(e2)*sin(2*e1))* sin(2*φ)* sin(e3 - (t+phase)*ω) + α1* cos(e1)*(4* sin(e1)*(8*cos(φ)^2*sin(e1)^2 + 5*cos((t+phase)*ω)*sin(2*μ)*sin(e1)* sin(2*φ)*sin(e3) - 8*cos((t+phase)*ω)^2*sin(μ)^2* sin(φ)^2*sin(e3)^2 + sin(e1)*((-cos(e3))* sin((t+phase)*ω)*(5*sin(2*μ)* sin(2*φ) + 8*cos(e3)*sin(e1)*sin(φ)^2* sin((t+phase)*ω)) + 4*((-cot(e2))*sin(2*μ) + sin(e1))* sin(φ)^2*sin(2*e3)* sin(2*(t+phase)*ω))) + 11*sin(2*μ)*sin(2*φ)* sin(e3 - (t+phase)*ω))) + (1/4)*α1*cot(e2)* sin(2*μ)*((5 + cos(2*e2))* sin(e1) + (-3 + cos(2*e2))*sin(3*e1))* sin(φ)^2*sin(2*(e3 - (t+phase)*ω)) + 2*α1*cos(e2)^2* (cos(e3 - (t+phase)*ω)*(2*cot(e2)*sin(μ)^2 - cos(e1)^2*sin(2*μ)*sin(e1))*sin(2*φ) + 2*sin(μ)^2*sin(φ)^2* sin(2*(e3 - (t+phase)*ω))) - 2*α1*cos(μ)^2* (sin(e2)*sin(2*e1)*sin(2*φ)* sin(e3 - (t+phase)*ω) + cos(e2)*(4*cos(e1)*cos(φ)^2*sin(e1) - sin(2*e1)*sin(e3 - (t+phase)*ω)* (cot(e2)*sin(2*φ) + 2*sin(φ)^2*sin(e3 - (t+phase)*ω))) + 2*cos(e1)^2*(cos(e3 - (t+phase)*ω)*cot(e2)* sin(2*φ) + sin(φ)^2*sin(2*(e3 - (t+phase)*ω)))) + cos(μ)* cos(e1)*(2* sin(e1)^2*(-4*cos(φ)*cot(e2) + 4*α1*cos(e2)*cos(φ)^2*cot(e2)* sin(μ) + sin(φ)*(α1*sin(μ)* sin(φ)*(-((-1 + cos(2*e2) + 2*cos(2*e3))* cos(2*(t+phase)*ω)*csc(e2)) + 2*cos(e2)* cot(e2)*(-1 + sin(2*e3)*sin(2*(t+phase)*ω))) - 4*sin(e3 - (t+phase)*ω))) + cos(e1)^2*(-8*cos(φ)*cot(e2) + 8*α1*cos(e2)*cos(φ)^2* cot(e2)*sin(μ) - 2*α1*sin(μ)* sin(φ)^2*(cos(e2)*(2 + cos(2*(e3 + (t+phase)*ω)))* cot(e2) - 4*cos(2*(t+phase)*ω)*sin(e2)*sin(e3)^2) - 8*sin(φ)*sin(e3 - (t+phase)*ω)) - α1*(-3 + cos(2*e2))*cos(e1)*cos(e3)* cot(e2)*sin(μ)*sin(e1)*sin(φ)^2* (3*sin(e3 - 2*(t+phase)*ω) + sin(e3 + 2*(t+phase)*ω))))
end φ ω a μ mg α1 Φg Θg phase
#     >-###'>           apg12         <'###-<
model_apg12 = @ode_def begin
  de1 = (1/(32*a^2))*(-6*α1*cos(e1)*cos(φ)^2* sin(2*e2)*sin(2*μ) - 9*α1*cos(φ)^2*sin(e2)^2*sin(2*e1) + 3*α1*cos(2*μ)*cos(φ)^2*sin(e2)^2* sin(2*e1) - 16*a^2*cos(e1)^2*cos(e3)*cos((t+phase)*ω)* sin(2*e2)*sin(μ)*sin(φ) - 16*a^2*cos(e3)*cos((t+phase)*ω)*sin(2*e2)*sin(μ)* sin(e1)^2*sin(φ) + 8*a^2*α1*cos(e1)^3*cos(e3)^2*cos((t+phase)*ω)^2* sin(2*e2)*sin(2*μ)* sin(φ)^2 + 8*a^2*α1*cos(e1)*cos(e3)^2*cos((t+phase)*ω)^2* sin(2*e2)*sin(2*μ)*sin(e1)^2*sin(φ)^2 + 9*α1*cos(e3 - (t+phase)*ω)^2*sin(2*e1)* sin(φ)^2 - 3*α1*cos(2*μ)*cos(e3 - (t+phase)*ω)^2*sin(2*e1)* sin(φ)^2 + 24* cos(μ)*(cos(e1)*cos(φ)*sin(e2) - cos(e3 - (t+phase)*ω)*sin(e1)*sin(φ)) - 9*α1*cos(e1)^2*cos(e3 - (t+phase)*ω)*sin(e2)* sin(2*φ) + 3*α1*cos(2*μ)*cos(e1)^2*cos(e3 - (t+phase)*ω)* sin(e2)*sin(2*φ) + 9*α1*cos(e3 - (t+phase)*ω)*sin(e2)* sin(e1)^2*sin(2*φ) - 3*α1*cos(2*μ)*cos(e3 - (t+phase)*ω)*sin(e2)* sin(e1)^2*sin(2*φ) + 16*a^2*mg*cos(Θg)*cos(e1)^2*cos(e3)* sin(2*e2)*sin(Φg) + 16*a^2*mg*cos(Θg)*cos(e3)*sin(2*e2)* sin(e1)^2*sin(Φg) - 16*a^2*mg*cos(e1)^2*sin(2*e2)*sin(Θg)* sin(e3) - 16*a^2*mg*sin(2*e2)*sin(Θg)*sin(e1)^2* sin(e3) + 6*α1*cos(e3 - 2*(t+phase)*ω)*sin(e2)* sin(2*μ)*sin(e1)*sin(φ)^2*sin(e3) - 6*α1*cos(e1)*cos((t+phase)*ω)*sin(e2)^2* sin(2*μ)*sin(2*φ)*sin(e3) - 8*a^2*α1*cos(e1)^3*cos((t+phase)*ω)*sin(2*e2)* sin(e1)*sin(2*φ)*sin(e3) - 8*a^2*α1*cos(e1)*cos((t+phase)*ω)*sin(2*e2)* sin(e1)^3*sin(2*φ)*sin(e3) - 8*a^2*α1*cos(e1)^3*cos((t+phase)*ω)^2* sin(2*e2)*sin(2*μ)*sin(φ)^2*sin(e3)^2 - 8*a^2*α1*cos(e1)*cos((t+phase)*ω)^2*sin(2*e2)* sin(2*μ)*sin(e1)^2*sin(φ)^2*sin(e3)^2 + 3*α1*sin(e2)*sin(2*μ)*sin(e1)* sin(φ)^2*sin(2*e3) - 12*α1*cos(e3)*cos(e3 - (t+phase)*ω)*sin(e2)* sin(2*μ)*sin(e1)*sin(φ)^2*sin((t+phase)*ω) + 6*α1*cos(e1)*cos(e3)*sin(e2)^2*sin(2*μ)* sin(2*φ)*sin((t+phase)*ω) + 8*a^2*α1*cos(e1)^3*cos(e3)*sin(2*e2)* sin(e1)*sin(2*φ)*sin((t+phase)*ω) + 8*a^2*α1*cos(e1)*cos(e3)*sin(2*e2)* sin(e1)^3*sin(2*φ)*sin((t+phase)*ω) - 16*a^2*cos(e1)^2*sin(2*e2)*sin(μ)*sin(φ)* sin(e3)*sin((t+phase)*ω) - 16*a^2*sin(2*e2)*sin(μ)*sin(e1)^2* sin(φ)*sin(e3)*sin((t+phase)*ω) - 8*a^2*α1*cos(e1)^3*cos(e3)^2*sin(2*e2)* sin(2*μ)*sin(φ)^2*sin((t+phase)*ω)^2 - 8*a^2*α1*cos(e1)*cos(e3)^2*sin(2*e2)* sin(2*μ)*sin(e1)^2*sin(φ)^2* sin((t+phase)*ω)^2 + 8*a^2*α1*cos(e1)^3*sin(2*e2)*sin(2*μ)* sin(φ)^2*sin(e3)^2*sin((t+phase)*ω)^2 + 8*a^2*α1*cos(e1)*sin(2*e2)*sin(2*μ)* sin(e1)^2*sin(φ)^2*sin(e3)^2* sin((t+phase)*ω)^2 + 8*a^2*α1*sin(e2)*sin(2*e2)*sin(μ)^2* sin(e1)^2*sin(φ)^2*sin(2*e3)* sin((t+phase)*ω)^2 + 8*a^2*α1*sin(2*e2)* sin(φ)^2*((-sin(e2))*sin(μ)^2*sin(e1)^2* sin(e3)^2 + cos(e1)*sin(2*μ)*sin(2*e3))* sin(2*(t+phase)*ω) + α1* cos(μ)^2*(6* sin(2*e1)*(cos(φ)^2*sin(e2)^2 - cos(e3 - (t+phase)*ω)^2*sin(φ)^2) + 6*cos(2*e1)*cos(e3 - (t+phase)*ω)*sin(e2)* sin(2*φ) + (-3 + 4*a^2)*sin(2*e2)* sin(2*e1)*sin(2*φ)*sin(e3 - (t+phase)*ω)) - cos(e2)^2*(32*a^2*cos(e3)*cos((t+phase)*ω)* cot(e2)*sin(μ)*sin(e1)^2*sin(φ) - 32*a^2*mg*cos(Θg)*cos(e3)*cot(e2)* sin(e1)^2*sin(Φg) + 32*a^2*mg*cot(e2)*sin(Θg)*sin(e1)^2* sin(e3) - 8*a^2*α1*cos(e3)^2*cos((t+phase)*ω)*cot(e2)* sin(2*e1)*sin(2*φ)*sin(e3) + 8*a^2*α1*cos(μ)*cos(e1)* sin(μ)*(cos(e1)^2* cot(e2)*(4*cos(φ)^2 - cos(2*(e3 + (t+phase)*ω))*sin(φ)^2) + 2*cos(φ)*cos(2*e3)*cos((t+phase)*ω)* sin(e1)^2*sin(φ)*sin(e3)) + 8*a^2*α1*cos(e3)^3*cot(e2)*sin(2*e1)* sin(2*φ)*sin((t+phase)*ω) + 32*a^2*cot(e2)*sin(μ)*sin(e1)^2*sin(φ)* sin(e3)*sin((t+phase)*ω) + 8*a^2*cos(e1)^2*(2* cos(e3)*(cos((t+phase)*ω)* sin(μ)*(2*cot(e2)* sin(φ) + α1*sin(e2)* sin(μ)*sin(2*φ)) - 2*mg*cos(Θg)*cot(e2)* sin(Φg)) + sin(e3)*(4* cot(e2)*(mg*sin(Θg) + sin(μ)*sin(φ)* sin((t+phase)*ω)) + α1* sin(e2)*(2*sin(μ)^2*sin(2*φ)* sin((t+phase)*ω) + sin(2*μ)*sin(e1)*sin(φ)^2* sin(e3)*sin(2*(t+phase)*ω)))) + 9*α1*sin(2*e1)*sin(φ)^2* sin(e3 - (t+phase)*ω)^2 - 3*α1*cos(2*μ)*sin(2*e1)*sin(φ)^2* sin(e3 - (t+phase)*ω)^2 + 2*α1* cos(e1)*(a^2* sin(e1)^2*(2* cos(2*φ)*((3 + cos(2*(e3 - (t+phase)*ω)))* cot(e2)*sin(2*μ) + 6*sin(e1)) + sin(e1)*(4 + 8*cos(2*(e3 - (t+phase)*ω))*sin(φ)^2) + sin(2*μ)* sin(2*φ)*((-cos((t+phase)*ω))*(-9* sin(e3) + sin(3*e3)) - 8*cos(e3)*sin((t+phase)*ω))) - sin(2*φ)*(3*sin(2*μ) + 8*a^2*cot(e2)*sin(e1)^3*sin(e3)^2)* sin(e3 - (t+phase)*ω) + 4*a^2*cot(e2)*sin(2*μ)*sin(e1)^2* sin(e3 - (t+phase)*ω)^2) + 2*α1*cos(μ)^2* sin(2*e1)*(2* a^2*(-2*cos(e3 - (t+phase)*ω)^2 + cos(2*φ)*(-3 + cos(2*(e3 - (t+phase)*ω)))) + 4*a^2*cot(e2)*sin(2*φ)* sin(e3 - (t+phase)*ω) - 3*sin(φ)^2*sin(e3 - (t+phase)*ω)^2) + 4*a^2*α1* cos(e1)^3*(8*cos(φ)^2* sin(e1) + (-2 - 2*cos(2*(e3 - (t+phase)*ω)) + cos(2*(e3 + (t+phase)*ω)))*cot(e2)*sin(2*μ)* sin(φ)^2 + 4*sin(2*φ)*(sin(2*μ) - cot(e2)*sin(e1)*sin(e3)^2)* sin(e3 - (t+phase)*ω) - 8*sin(e1)*sin(φ)^2* sin(e3 - (t+phase)*ω)^2) + 8*a^2*α1*sin(e2)*sin(2*μ)*sin(e1)^3* sin(φ)^2*sin(2*(e3 - (t+phase)*ω))) + 8*a^2*α1* cos(e2)^3*(cos(e3 - (t+phase)*ω)*(-2*cot(e2)*sin(μ)^2 + sin(2*μ)*sin(e1))* sin(2*φ) - (2*sin(μ)^2 + cot(e2)*sin(2*μ)*sin(e1))*sin(φ)^2* sin(2*(e3 - (t+phase)*ω))) + cos(e2)*(8*cos(μ)* cos(e1)*(4*a^2*cos(φ)*cot(e2) + sin(φ)*(-3 + 4*a^2 + 3*α1*cos((t+phase)*ω)*sin(e2)*sin(μ)* sin(φ)*sin(e3))* sin(e3 - (t+phase)*ω)) + 2*α1* cos(μ)^2*(8*a^2*cos(e1)^2*cos(e3 - (t+phase)*ω)* cot(e2)* sin(2*φ) + (4*a^2 + (-3 + 4*a^2)*cos(2*e1))* sin(φ)^2*sin(2*(e3 - (t+phase)*ω))) + α1*(-4*a^2*cos(2*(e2 + μ))* cos(φ)*cos(e3 - (t+phase)*ω)*cot(e2)* sin(e1)^2*sin(φ) + 8*a^2*cos(e1)^2*cos(e3)*cos((t+phase)*ω)* sin(e2)^2*sin(2*μ)*sin(e1)* sin(2*φ) + 8*a^2*cos(e3)*cos((t+phase)*ω)*sin(e2)^2* sin(2*μ)*sin(e1)^3*sin(2*φ) + 2*cos(e3 - (t+phase)*ω)* sin(e1)*(3*sin(2*μ) + a^2*(6 + 2*cos(2*e2) - cos(2*(e2 - μ)) + 2*cos(2*μ))*cot(e2)*sin(e1))* sin(2*φ) - 16*a^2*cos(e1)^2*cos((t+phase)*ω)^2*sin(e2)^2* sin(μ)^2*sin(φ)^2*sin(2*e3) - 16*a^2*cos((t+phase)*ω)^2*sin(e2)^2*sin(μ)^2* sin(e1)^2*sin(φ)^2*sin(2*e3) + 16*a^2*cos((t+phase)*ω)^2*sin(e1)^4*sin(φ)^2* sin(2*e3) + 4*a^2*cos((t+phase)*ω)^2*sin(2*e1)^2*sin(φ)^2* sin(2*e3) + 8*a^2*cos(e1)^2*sin(e2)^2*sin(2*μ)* sin(e1)*sin(2*φ)*sin(e3)* sin((t+phase)*ω) + 8*a^2*sin(e2)^2*sin(2*μ)*sin(e1)^3* sin(2*φ)*sin(e3)*sin((t+phase)*ω) + 16*a^2*cos(e1)^2*sin(e2)^2*sin(μ)^2* sin(φ)^2*sin(2*e3)*sin((t+phase)*ω)^2 - 16*a^2*sin(e1)^4*sin(φ)^2*sin(2*e3)* sin((t+phase)*ω)^2 - 4*a^2*sin(2*e1)^2*sin(φ)^2*sin(2*e3)* sin((t+phase)*ω)^2 + 16*a^2*cos(e1)^2*cos(e3)^2*sin(e2)^2* sin(μ)^2*sin(φ)^2*sin(2*(t+phase)*ω) + 16*a^2*cos(e3)^2*sin(e2)^2*sin(μ)^2* sin(e1)^2*sin(φ)^2*sin(2*(t+phase)*ω) - 16*a^2*cos(e3)^2*sin(e1)^4*sin(φ)^2* sin(2*(t+phase)*ω) - 4*a^2*cos(e3)^2*sin(2*e1)^2*sin(φ)^2* sin(2*(t+phase)*ω) - 16*a^2*cos(e1)^2*sin(e2)^2*sin(μ)^2* sin(φ)^2*sin(e3)^2*sin(2*(t+phase)*ω) - 8*a^2*cos(e1)^2*cot(e2)*sin(2*μ)*sin(e1)* sin(φ)^2*sin(e3)^2* sin(2*(t+phase)*ω) + 16*a^2*sin(e1)^4*sin(φ)^2*sin(e3)^2* sin(2*(t+phase)*ω) + 4*a^2*sin(2*e1)^2*sin(φ)^2*sin(e3)^2* sin(2*(t+phase)*ω) - 18*a^2*cos(e1)^2*cos(e3)*cot(e2)*sin(2*μ)* sin(e1)*sin(φ)^2* sin(e3 - 2*(t+phase)*ω) + 6*a^2*cos(2*e2)*cos(e1)^2*cos(e3)* cot(e2)*sin(2*μ)*sin(e1)*sin(φ)^2* sin(e3 - 2*(t+phase)*ω) + 9*sin(e2)*sin(2*e1)*sin(2*φ)* sin(e3 - (t+phase)*ω) - 3*cos(2*μ)*sin(e2)*sin(2*e1)* sin(2*φ)*sin(e3 - (t+phase)*ω) - 12*cos(e1)*cos(e3)*sin(e2)*sin(2*μ)* sin(φ)^2*sin((t+phase)*ω)* sin(e3 - (t+phase)*ω) + 9*cos(e1)^2*sin(φ)^2* sin(2*(e3 - (t+phase)*ω)) - 3*cos(2*μ)*cos(e1)^2*sin(φ)^2* sin(2*(e3 - (t+phase)*ω)) + 8*a^2*cos(e1)^2*cot(e2)*sin(2*μ)*sin(e1)* sin(φ)^2*sin(2*(e3 - (t+phase)*ω)) - 9*sin(e1)^2*sin(φ)^2* sin(2*(e3 - (t+phase)*ω)) + 3*cos(2*μ)*sin(e1)^2*sin(φ)^2* sin(2*(e3 - (t+phase)*ω)) + 2*a^2*(-3 + cos(2*e2))*cos(e1)^2*cos(e3)* cot(e2)*sin(2*μ)*sin(e1)*sin(φ)^2* sin(e3 + 2*(t+phase)*ω))))
  de2 =  (1/32)*(-4*α1*cos(e2)^3*cos(e3 - (t+phase)*ω)* sin(2*e1)*sin(2*φ) + 4*α1* cos(e2)^2*((-2*cos(e3 - (t+phase)*ω)^2 + cos(2*φ)*(-3 + cos(2*(e3 - (t+phase)*ω))))* sin(2*μ)*sin(e1) - (1 + 3*cos(2*μ) + 2*cos(2*e1)*sin(μ)^2)* sin(2*φ)*sin(e3 - (t+phase)*ω)) + 2*(16*mg*cos(Θg)*cos(Φg)* sin(e2) + α1* cos(e1)^2*(cos(2*μ) - cos(2*e1) + 6*cos(μ)^2*cos(2*φ))*sin(2*e2) + 6*α1*cos(μ)^2*cos(2*φ)*sin(2*e2)* sin(e1)^2 - 16*cos(φ)*cos(e3)^2*sin(e2)*sin(μ)* sin(e1)^2 + 8*α1*cos(φ)^2*cos(e3)^2*sin(e2)^2* sin(2*μ)*sin(e1)^3 + 2*α1*cos(μ)^2*cos(2*(e3 - (t+phase)*ω))* sin(2*e2)*sin(e1)^2*sin(φ)^2 + 2*α1*cos(μ)^2*cos(2*(e3 + (t+phase)*ω))* sin(2*e2)*sin(e1)^2*sin(φ)^2 + 4*α1*cos(e1)^3*cos(e3 - (t+phase)*ω)* sin(e2)*sin(2*μ)*sin(2*φ) + 16*cos(μ)*cos((t+phase)*ω)*sin(e2)*sin(e1)^3* sin(φ)*sin(e3) - 8*α1*cos(e3)^2*cos((t+phase)*ω)*sin(e2)^2* sin(μ)^2*sin(e1)^2*sin(2*φ)*sin(e3) - 8*α1*cos((t+phase)*ω)*sin(2*e2)*sin(2*μ)* sin(e1)^3*sin(2*φ)*sin(e3) + 8*α1*cos(μ)^2*cos((t+phase)*ω)*sin(e2)^2* sin(e1)^4*sin(2*φ)*sin(e3) + 2*α1*cos((t+phase)*ω)*sin(e2)^2*sin(2*e1)^2* sin(2*φ)*sin(e3) + 2*α1*cos(μ)^2*cos((t+phase)*ω)*sin(e2)^2* sin(2*e1)^2*sin(2*φ)*sin(e3) - 16*cos(φ)*sin(e2)*sin(μ)*sin(e1)^2* sin(e3)^2 + 8*α1*cos(φ)^2*sin(e2)^2*sin(2*μ)* sin(e1)^3*sin(e3)^2 - 8*α1*cos((t+phase)*ω)^2*sin(e2)^2*sin(2*μ)* sin(e1)^3*sin(φ)^2*sin(e3)^2 - 8*α1*cos((t+phase)*ω)*sin(e2)^2*sin(μ)^2* sin(e1)^2*sin(2*φ)*sin(e3)^3 - 16*cos(μ)*cos(e3)*sin(e2)*sin(e1)^3* sin(φ)*sin((t+phase)*ω) + 8*α1*cos(e3)^3*sin(e2)^2*sin(μ)^2* sin(e1)^2*sin(2*φ)*sin((t+phase)*ω) + 8*α1*cos(e3)*sin(2*e2)*sin(2*μ)* sin(e1)^3*sin(2*φ)*sin((t+phase)*ω) - 8*α1*cos(μ)^2*cos(e3)*sin(e2)^2* sin(e1)^4*sin(2*φ)*sin((t+phase)*ω) - 2*α1*cos(e3)*sin(e2)^2*sin(2*e1)^2* sin(2*φ)*sin((t+phase)*ω) - 2*α1*cos(μ)^2*cos(e3)*sin(e2)^2* sin(2*e1)^2*sin(2*φ)*sin((t+phase)*ω) + 8*α1*cos(e3)*sin(e2)^2*sin(μ)^2* sin(e1)^2*sin(2*φ)*sin(e3)^2* sin((t+phase)*ω) - 8*α1*cos(e3)^2*sin(e2)^2*sin(2*μ)* sin(e1)^3*sin(φ)^2*sin((t+phase)*ω)^2 + α1*(4* sin(e1)^2*(cos(2*μ)*sin(2*e2) + sin(e2)^2*sin(2*μ)*sin(e1)) + sin(2*e2)*sin(2*e1)^2)*sin(φ)^2* sin(2*e3)*sin(2*(t+phase)*ω) + 4*α1*cos(e1)*cos(e3 - (t+phase)*ω)*sin(e2)* sin(e1)*(sin(2*μ)*sin(e1)*sin(2*φ) - 4*sin(μ)^2*sin(φ)^2* sin(e3 - (t+phase)*ω)) + 4*α1* cos(e1)^4*(cos(2*(e3 - (t+phase)*ω))*sin(2*e2)* sin(φ)^2 + 2*sin(e2)^2*sin(2*φ)* sin(e3 - (t+phase)*ω)) + cos(e1)^2*(2*(-8*cos(φ)*sin(e2)* sin(μ) + 4*α1*cos(φ)^2*sin(e2)^2* sin(2*μ)*sin(e1) + α1*sin(2*e2)* sin(φ)^2*((cos(2*μ) - cos(2*e1))* cos(2*e3)*cos(2*(t+phase)*ω) - 2*sin(μ)^2*sin(2*e3)* sin(2*(t+phase)*ω))) - (16* sin(e2)*(α1*cos(φ)*sin(e2)* sin(μ)^2 - cos(μ)*sin(e1))* sin(φ) + α1*(8 + cos(2*e3))* sin(2*e2)*sin(2*μ)*sin(e1)* sin(2*φ))*sin(e3 - (t+phase)*ω) - 8*α1*sin(e2)^2*sin(2*μ)*sin(e1)* sin(φ)^2*sin(e3 - (t+phase)*ω)^2)) + cos(e2)*(8*α1*cos(e1)^4*sin(e2) - 32*cos(e3)*(mg*sin(Θg) + α1* cos(φ)^2*cos(e3)*sin(e2)*sin(μ)^2)* sin(e1)^2 + 32*cos(μ)*cos(φ)*sin(e1)^3 - 16*α1*cos(e1)^3*cos(e3 - (t+phase)*ω)* sin(e1)*sin(2*φ) - 2*α1*(sin(e2)^2*(cos(2*μ)* cos(e3 - (t+phase)*ω) + (cos(e3 - (t+phase)*ω) + 2*cos(e3 + (t+phase)*ω))*sin(μ)^2)* sin(2*e1) - cos(e3 - (t+phase)*ω)*sin(4*e1))* sin(2*φ) + α1* cos(μ)^2*(8*sin(e2)* sin(e1)^2 + (7 + cos(2*e2))* cos(e3 - (t+phase)*ω)*sin(2*e1)* sin(2*φ)) + 32*sin(e1)^2*(cos((t+phase)*ω)*sin(μ)* sin(φ) - mg*cos(Θg)*sin(Φg))* sin(e3) - 32*α1*sin(e2)*sin(μ)^2* sin(e1)^2*(cos(φ)^2 - cos((t+phase)*ω)^2*sin(φ)^2)*sin(e3)^2 - 8*sin(μ)*(4*cos(e3)*sin(e1)^2* sin(φ) + α1*sin(e2)^2*sin(μ)* sin(2*e1)*sin(2*φ)*sin(e3))* sin((t+phase)*ω) + 32*α1*cos(e3)^2*sin(e2)*sin(μ)^2* sin(e1)^2*sin(φ)^2*sin((t+phase)*ω)^2 - 4*cos(e1)^2*(-8*cos(μ)*cos(φ)*sin(e1) - cos((t+phase)*ω)*(8*sin(μ)*sin(φ) + α1* cos(2*e3)*sin(e2)*sin(2*μ)*sin(e1)* sin(2*φ))*sin(e3) + 8*mg*cos(Θg)*sin(Φg)*sin(e3) + cos(e3)*(8*mg* sin(Θg) + (8*sin(μ)* sin(φ) + α1*cos(2*e3)* sin(e2)*sin(2*μ)*sin(e1)* sin(2*φ))*sin((t+phase)*ω))) - 8*α1*cos(e1)*sin(2*μ)*sin(φ)^2* sin(2*(e3 - (t+phase)*ω))))
  de3 =  (1/8)*(8*cos(e1)^2*cos(e3)*cos((t+phase)*ω)*sin(e2)* sin(μ)*sin(φ) + α1* cos(2*(e2 - μ))*cos(φ)* cos(e3 - (t+phase)*ω)*cot(e2)*sin(e1)^2* sin(φ) + α1*cos(2*(e2 + μ))*cos(φ)* cos(e3 - (t+phase)*ω)*cot(e2)*sin(e1)^2* sin(φ) + 8*cos(e3)*cos((t+phase)*ω)*sin(e2)*sin(μ)* sin(e1)^2*sin(φ) - 4*α1*cos(e1)^3*cos(e3)^2*cos(2*(t+phase)*ω)* sin(e2)*sin(2*μ)*sin(φ)^2 - 4*α1*cos(e1)*cos(e3)^2*cos(2*(t+phase)*ω)* sin(e2)*sin(2*μ)*sin(e1)^2*sin(φ)^2 - 2*α1*cos(e1)^2*cos(e3 - (t+phase)*ω)* sin(e2)^2*sin(2*μ)*sin(e1)* sin(2*φ) - α1*cos(e3 - (t+phase)*ω)* cot(e2)*sin(e1)^2*sin(2*φ) - α1*cos(2*e2)*cos(e3 - (t+phase)*ω)* cot(e2)*sin(e1)^2*sin(2*φ) - α1* cos(2*μ)*cos(e3 - (t+phase)*ω)*cot(e2)* sin(e1)^2*sin(2*φ) - 4*α1*cos(e1)^2*cos(e3 - (t+phase)*ω)*cot(e2)* sin(e1)^2*sin(2*φ) + 2*α1*cos(2*e1)*cos(e3 - (t+phase)*ω)* cot(e2)*sin(e1)^2*sin(2*φ) - 2*α1*cos(e3 - (t+phase)*ω)*sin(2*μ)*sin(e1)^3* sin(2*φ) - 8*mg*cos(Θg)*cos(e1)^2*cos(e3)* sin(e2)*sin(Φg) - 8*mg*cos(Θg)*cos(e3)*sin(e2)* sin(e1)^2*sin(Φg) + 8*mg*cos(e1)^2*sin(e2)*sin(Θg)* sin(e3) + 8*mg*sin(e2)*sin(Θg)*sin(e1)^2* sin(e3) + 8*α1*cos(e1)^3*cos(φ)*cos((t+phase)*ω)* sin(e2)*sin(e1)*sin(φ)*sin(e3) + 8*α1*cos(e1)*cos(φ)*cos((t+phase)*ω)* sin(e2)*sin(e1)^3*sin(φ)*sin(e3) + 8*α1*cos(e1)^2*cos(e3)*cos((t+phase)*ω)^2* sin(e2)^2*sin(μ)^2*sin(φ)^2*sin(e3) - 8*α1*cos(e1)^2*cos(e3)*cos((t+phase)*ω)^2* sin(e1)^2*sin(φ)^2*sin(e3) + 8*α1*cos(e3)*cos((t+phase)*ω)^2*sin(e2)^2* sin(μ)^2*sin(e1)^2*sin(φ)^2*sin(e3) - 8*α1*cos(e3)*cos((t+phase)*ω)^2*sin(e1)^4* sin(φ)^2*sin(e3) - 8*α1*cos(e1)^3*cos(φ)*cos(e3)* sin(e2)*sin(e1)*sin(φ)*sin((t+phase)*ω) - 8*α1*cos(e1)*cos(φ)*cos(e3)* sin(e2)*sin(e1)^3*sin(φ)*sin((t+phase)*ω) + 8*cos(e1)^2*sin(e2)*sin(μ)*sin(φ)* sin(e3)*sin((t+phase)*ω) + 8*sin(e2)*sin(μ)*sin(e1)^2*sin(φ)* sin(e3)*sin((t+phase)*ω) - 8*α1*cos(e1)^2*cos(e3)*sin(e2)^2* sin(μ)^2*sin(φ)^2*sin(e3)*sin((t+phase)*ω)^2 - 8*α1*cos(e3)*sin(e2)^2*sin(μ)^2* sin(e1)^2*sin(φ)^2*sin(e3)*sin((t+phase)*ω)^2 + 8*α1*cos(e3)*sin(e1)^4*sin(φ)^2* sin(e3)*sin((t+phase)*ω)^2 + α1*sin(2*e1)^2* sin(φ)^2*sin(2*e3)*sin((t+phase)*ω)^2 - 4*α1*cos(e1)^2*cos(e3)^2*sin(e2)^2* sin(μ)^2*sin(φ)^2*sin(2*(t+phase)*ω) + 4*α1*cos(e1)^2*cos(e3)^2*sin(e1)^2* sin(φ)^2*sin(2*(t+phase)*ω) - 4*α1*cos(e3)^2*sin(e2)^2*sin(μ)^2* sin(e1)^2*sin(φ)^2*sin(2*(t+phase)*ω) + 4*α1*cos(e3)^2*sin(e1)^4*sin(φ)^2* sin(2*(t+phase)*ω) + 4*α1*cos(e1)^2*sin(e2)^2*sin(μ)^2* sin(φ)^2*sin(e3)^2*sin(2*(t+phase)*ω) + α1*cos(e1)^2*sin(2*e2)*sin(2*μ)* sin(e1)*sin(φ)^2*sin(e3)^2*sin(2*(t+phase)*ω) + 4*α1*sin(e2)^2*sin(μ)^2*sin(e1)^2* sin(φ)^2*sin(e3)^2*sin(2*(t+phase)*ω) - 4*α1*sin(e1)^4*sin(φ)^2*sin(e3)^2* sin(2*(t+phase)*ω) + α1*cos(e1)*cot(e2)*sin(2*μ)* sin(2*e1)*sin(φ)^2*sin(e3)^2* sin(2*(t+phase)*ω) - α1*sin(2*e1)^2*sin(φ)^2* sin(e3)^2*sin(2*(t+phase)*ω) - 4*α1*cos(e1)^3*sin(e2)*sin(2*μ)* sin(φ)^2*sin(2*e3)*sin(2*(t+phase)*ω) - 4*α1*cos(e1)*sin(e2)*sin(2*μ)* sin(e1)^2*sin(φ)^2*sin(2*e3)* sin(2*(t+phase)*ω) + (1/4)* cos(e2)*(4*α1* cos(e1)^3*(2*(1 + 3*cos(2*φ))* sin(e1) + (cos(2*(e3 + (t+phase)*ω))*cot(e2)* sin(2*μ) + cos(2*(e3 - (t+phase)*ω))*(-2*cot(e2)* sin(2*μ) + 4*sin(e1)))* sin(φ)^2) + 32*cos(e3)*cot(e2)* sin(e1)^2*(cos((t+phase)*ω)*sin(μ)*sin(φ) - mg*cos(Θg)*sin(Φg)) + 4*sin(e3)*(α1* cos((t+phase)*ω)^2*(-2*cos(2*μ)*sin(2*e1) + sin(4*e1))*sin(φ)^2*sin(e3) + 8*cot(e2)* sin(e1)^2*(mg*sin(Θg) + sin(μ)*sin(φ)*sin((t+phase)*ω))) + 32*cos(e1)^2*(cos(e3)*(cos((t+phase)*ω)* sin(μ)*(cot(e2) + α1*cos(φ)* sin(e2)*sin(μ))*sin(φ) - mg*cos(Θg)*cot(e2)* sin(Φg)) + sin(e3)*(mg*cot(e2)*sin(Θg) + sin(μ)*(cot(e2) + α1*cos(φ)* sin(e2)*sin(μ))*sin(φ)* sin((t+phase)*ω))) + α1*(5*cos(3*e1)* sin(2*μ) - 8*cot(e2)*sin(2*e1))* sin(2*φ)*sin(e3 - (t+phase)*ω) + α1* cos(e1)*(4* sin(e1)*(8*cos(φ)^2*sin(e1)^2 + 5*cos((t+phase)*ω)*sin(2*μ)*sin(e1)* sin(2*φ)*sin(e3) - 8*cos((t+phase)*ω)^2*sin(μ)^2*sin(φ)^2* sin(e3)^2 + sin(e1)*((-cos(e3))* sin((t+phase)*ω)*(5*sin(2*μ)*sin(2*φ) + 8*cos(e3)*sin(e1)*sin(φ)^2* sin((t+phase)*ω)) + 4*((-cot(e2))*sin(2*μ) + sin(e1))* sin(φ)^2*sin(2*e3)* sin(2*(t+phase)*ω))) + 11*sin(2*μ)*sin(2*φ)* sin(e3 - (t+phase)*ω))) + (1/4)*α1*cot(e2)* sin(2*μ)*((5 + cos(2*e2))* sin(e1) + (-3 + cos(2*e2))*sin(3*e1))* sin(φ)^2*sin(2*(e3 - (t+phase)*ω)) + 2*α1* cos(e2)^2*(cos(e3 - (t+phase)*ω)*(2*cot(e2)*sin(μ)^2 - cos(e1)^2*sin(2*μ)*sin(e1))* sin(2*φ) + 2*sin(μ)^2*sin(φ)^2* sin(2*(e3 - (t+phase)*ω))) - 2*α1* cos(μ)^2*(sin(e2)*sin(2*e1)*sin(2*φ)* sin(e3 - (t+phase)*ω) + cos(e2)*(4*cos(e1)*cos(φ)^2*sin(e1) - sin(2*e1)* sin(e3 - (t+phase)*ω)*(cot(e2)*sin(2*φ) + 2*sin(φ)^2*sin(e3 - (t+phase)*ω))) + 2*cos(e1)^2*(cos(e3 - (t+phase)*ω)*cot(e2)* sin(2*φ) + sin(φ)^2*sin(2*(e3 - (t+phase)*ω)))) + cos(μ)* cos(e1)*(2* sin(e1)^2*(-4*cos(φ)*cot(e2) + 4*α1*cos(e2)*cos(φ)^2*cot(e2)* sin(μ) + sin(φ)*(α1*sin(μ)* sin(φ)*(-((-1 + cos(2*e2) + 2*cos(2*e3))*cos(2*(t+phase)*ω)* csc(e2)) + 2*cos(e2)* cot(e2)*(-1 + sin(2*e3)*sin(2*(t+phase)*ω))) - 4*sin(e3 - (t+phase)*ω))) + cos(e1)^2*(-8*cos(φ)*cot(e2) + 8*α1*cos(e2)*cos(φ)^2*cot(e2)* sin(μ) - 2*α1*sin(μ)* sin(φ)^2*(cos(e2)*(2 + cos(2*(e3 + (t+phase)*ω)))*cot(e2) - 4*cos(2*(t+phase)*ω)*sin(e2)*sin(e3)^2) - 8*sin(φ)* sin(e3 - (t+phase)*ω)) - α1*(-3 + cos(2*e2))*cos(e1)*cos(e3)*cot(e2)* sin(μ)*sin(e1)* sin(φ)^2*(3*sin(e3 - 2*(t+phase)*ω) + sin(e3 + 2*(t+phase)*ω))))
end φ ω a μ mg α1 Φg Θg phase
#     >-###'>                    <'###-<
model_ = @ode_def begin
  de1 = (1/(4*a^2))*(-3*cos(e3 - (t+phase)*ω)*sin(e1)* sin(φ) + cos(e1)*(cos(φ)*(3 - 4*a^2 + 4*a^2*csc(e2)^2)*sin(e2) + (-3 + 4*a^2)*cos(e2)*sin(φ)* sin(e3 - (t+phase)*ω)))
  de2 =  sin(e1)*(cos(e2)*cos(φ) + sin(e2)*sin(φ)*sin(e3 - (t+phase)*ω))
  de3 =  (-cos(e1))*(cos(φ)*cot(e2) + sin(φ)*sin(e3 - (t+phase)*ω))
end φ ω a μ mg α1 Φg Θg phase



function eval_forward_model_flexible_integration_subroutine(rhs,end_time,φ, ω, a, μ, mg, α1, Φg, Θg, phase; xi=0.0000001, yi=0.0000001, ϕ0=0.0)
    EA = [x for x in get_EA(xi, yi, ϕ0=ϕ0)]
    p = [φ, ω, a, μ, mg, α1, Φg, Θg, phase]

    prob = ODEProblem(rhs,EA,(0.0,end_time),p, reltol=1e-6, abstol=1e-7) #go 10% longer? 
    res  = OrdinaryDiffEq.solve(prob,RadauIIA5()) 
    return res 
end
        

function eval_forward_model_ua_flexible(integration_times,φ, ω, a, μ, mg, α1, Φ, Θ, Φg, Θg, phase; xi=0.0000001, yi=0.0000001, ϕ0=0.0)
    let res 
        end_time=last(integration_times)*1.10
        
        m="g"
                
        if (Φg != 0.0) || (Θg != 0)
            m=m*"12"
        end
                
        if α1 != 0.0
            m="p"*m 
        end
        if μ != 0.0
            m="a"*m 
        end
        
        rhs_function_name = "model_"*m
        rhs_function = eval(Meta.parse(rhs_function_name))                
                
        res = eval_forward_model_flexible_integration_subroutine(rhs_function, end_time, φ, ω, a, μ, mg, α1, Φg, Θg, phase; xi=xi ,yi=yi,  ϕ0=ϕ0)
        rq = res(integration_times).u    
                
        r=map(x -> xp_ua_fn_tilt(x, Φ, Θ),rq)
                
        return r 
                
    end
end


function eval_one_exp(e,p, detail=false, euler="capmodel_ea")
    #print(p)
    
    B=5.0 #mT
    η=0.000865 #Pa s
    a=2.0 #um 
    log10_e=0.4342944819032518 
    scale=0.58

    #Multifunctional Superparamagnetic Janus Particles
    #Kai P. Yuet, Dae Kun Hwang, Ramin Haghgooie, and Patrick S. Doyle*
    
    
    
    exp_idx,expr = e
            
    
    exp_idx = exp_idx-1 #julia enumerates from 1, but I already adjusted the indexing elsewhere
    R = p[end] #.* 0.58 to convert to uM     
    
    if expr < 0
        R = R + 28.0
        scale = 0.318
    end 
            
    a_nondim=a/(R*scale)
    m = exp(p[end-1]/log10_e)
    K = (( m * 10^-14 ) * (B * 10^-3))/(η * π * a * (10^-6) * (R  * scale * (10^-6))^2   )    
    μ  = p[end-2] #rads 
    mg_nondim =exp(p[end-3]/log10_e) #mg * (R* 0.58*(10^-6))/(( m * 10^-14 ) * (B * 10^-3))
    mg = mg_nondim
    
    
    Δα = p[end-4] # 10 - 20
        

    σ  = exp(p[end-5]/log10_e)
    Φ = p[end-6] 
    Θ  = p[end-7]
    Φg = p[end-8] 
    Θg  = p[end-9]
        
    if σ < 0 
        return -Inf
    end

    #ϵ = 1 - 10^p[Int64(exp_idx*N_LOCAL +4)] #p[end-6]
    ϕ0 = p[Int64(exp_idx*N_LOCAL +4)] 
    
    data_dict=D
    if expr < 0
        expr= -expr
        data_dict=Dk
    end
    phi,f,fps,drop,idx,particle_full,dropd = data_dict[string(expr)]
    START = length(particle_full)÷3
    particle_full = transpose(hcat(particle_full...)) #already rows x columns 
    
    ω = f*2*pi
    log10w = log10(f*2*pi)  
        
    k_scale=1/((( 3.0 * 10^-14 ) * (B * 10^-3))/(6 * η * π * a * (10^-6) * ( (72*scale) * (10^-6))^2   ))
    f_scale=1/(f)
    t_scale=max(f_scale,k_scale) #set some sort of bound. 
    
    #times=particle_full[1:end,3].* 1.0/ fps
    #times=times.-minimum(times)
    
    # findfirst(x -> x >= t_scale, times)
    
    
    times=particle_full[START:end,3].* 1.0/ fps    
    times=times.-minimum(times)
    
    particle  = particle_full[START:end,1:2] 
    
    drop_center = [dropd[1],dropd[2]] 
    tau=t_scale 
    
    ϵ = exp(-1/(tau*fps)) # convert to FRAMES for the sivia correlation model  
    
    #ϵ=0.

    scaled_start = ([p[Int64(exp_idx*N_LOCAL +2)],p[Int64(exp_idx*N_LOCAL +3)]] .- drop_center)./(R) 
        
    iposR=(scaled_start[1]^2+scaled_start[2]^2)^0.5

    x0,y0 = scaled_start[1], scaled_start[2]
    
    if iposR > 1. #move to edge
        x0=scaled_start[1]/(iposR*1.00001)
        y0=scaled_start[2]/(iposR*1.00001) #avoid issue w. floating point math at exactly R

    #elseif  iposR < 0.00001 #nudge from center
    #    x0,y0 = 0.00001,0.00001
    end
    if false #
        println("xi,yi ",[x0,y0])

    end
    

    nondim_time = times.*K
    phase=(p[Int64(exp_idx*N_LOCAL +1)] / ω+ (START/fps))*K#same scaling as times  
    
    if false #
            println("pars ",[phi,ω/K, a_nondim, μ, mg, Δα, phase])
    end

    let raw_res
        
        raw_res=eval_forward_model_ua_flexible(nondim_time,phi,ω/K, a_nondim, μ, mg, Δα,  Φ, Θ, Φg, Θg, phase,xi=x0,yi=y0,ϕ0=ϕ0)

        
    
        raw_res_arr=hcat(raw_res...)
        rescaled_res=transpose((raw_res_arr.*R ).+drop_center)

        r = rescaled_res[1:end,:].-particle[1:end,:]
        all_LL=0
        if detail #formerly plot_one_exp
            return [rescaled_res[1:end,:],particle[1:end,:],times]
        else
            all_LL += corr_ll(r[1:end,1], ϵ, σ)  #all offsets by 1 since we don't count the start. 
            all_LL += corr_ll(r[1:end,2], ϵ, σ)  
            return all_LL  
        end
    end

end

function par_map(p,model,ALL_RUNS)
    vals=Dict('a'=>-2,'g'=>-3,'p'=>-4,'σ'=>-5,'Φ'=>-6,'Θ'=>-7,'1'=>-8,'2'=>-9,'ϕ'=>4)
    n_exps=length(ALL_RUNS)
    all_pars=zeros(N_LOCAL*n_exps+N_GLOBAL)
    
        
    #all_pars[1:3*length(ALL_RUNS)] = p[1:3*length(ALL_RUNS)] 
        
    all_pars[end+vals['σ']] = log10(4/.58) #0.0 #sigma = 1 by default
    all_pars[end+vals['g']] = -100000000.0 #0 log space 

    all_pars[end] = p[end] 
    all_pars[end-1] = p[end-1]
    
        
    extra_local = [vals[x] for x in model if vals[x] > 0] 
    n_extra_local = length(extra_local) 
        
    all_idxs=[vals[x] for x in model if vals[x] < 0] #skip globals
    
    for new_idx in enumerate(sort(all_idxs,rev=true))
            all_pars[end+new_idx[2]] = p[end-new_idx[1]-1]  
    end

    for e in 1:n_exps            
                        
        for par in 1:3
            all_pars[(e-1)*N_LOCAL + par] = p[(e-1)*(3+n_extra_local)+par]
        end
            
        for (old_idx,new_idx) in enumerate(extra_local)
            all_pars[(e-1)*N_LOCAL + new_idx] = p[(e-1)*(3+n_extra_local)+3+old_idx]
        end 
            
    end
   
        
    return all_pars
    
end

function par_umap(p,model,ALL_RUNS)
    
    vals=Dict('a'=>-2,'g'=>-3,'p'=>-4,'σ'=>-5,'Φ'=>-6,'Θ'=>-7,'1'=>-8,'2'=>-9,'ϕ'=>4)
        
    n_exps=length(ALL_RUNS)
    extra_local = [vals[x] for x in model if vals[x] > 0] 
    n_global_to_write = length([vals[x] for x in model if vals[x] < 0] )
    n_extra_local = length(extra_local) 
        
    all_pars=Vector{typeof(p[1])}(undef,(3+n_extra_local)*n_exps+2+n_global_to_write)
        

    all_pars[end] = p[end] 
    all_pars[end-1] = p[end-1]
    all_idxs=[vals[x] for x in model if vals[x] < 0]
    for new_idx in enumerate(sort(all_idxs,rev=true))
            all_pars[end-new_idx[1]-1] = p[end+new_idx[2]]  
    end
        
     
    for e in 1:n_exps            
        for par in 1:3
             all_pars[(e-1)*(3+n_extra_local)+par] = p[(e-1)*N_LOCAL + par] 
        end
        for (old_idx,new_idx) in enumerate(extra_local)
             all_pars[(e-1)*(3+n_extra_local)+3+old_idx] =p[(e-1)*N_LOCAL + new_idx]
        end 
    end
        
        
    return all_pars
    
end

function get_priors_sigma(ALL_RUNS, normpar=false)
    all_sigmas=[]
    for (exp_idx,expr) in enumerate(ALL_RUNS)
        push!(all_sigmas, 1.) #phase - fairly well defined          
        push!(all_sigmas, 5.)         
        push!(all_sigmas, 5.)         
        push!(all_sigmas, 3.)         
    end

    push!(all_sigmas, 0.01) # g1
    push!(all_sigmas, 0.01) # g2
    push!(all_sigmas, 0.02) # a1
    push!(all_sigmas, 0.02) # a2
    push!(all_sigmas, 1.0) #σ
    push!(all_sigmas, 1.0) #p
    push!(all_sigmas, 1.0) #mg 
    push!(all_sigmas, 0.1) #μ 
    push!(all_sigmas, 1.0) #m
    push!(all_sigmas, 0.25) #R
    
    return all_sigmas
end



function get_priors(ALL_RUNS, normpar=false)
    all_priors=[]
    all_bounds=[]
    dropd=0
    for (exp_idx,expr) in enumerate(ALL_RUNS)
        data_dict=D
        if expr < 0
            expr= -expr
            data_dict=Dk
        end

        phi,f,fps,drop,idx,particle_full,dropd = data_dict[string(expr)]
        B=5.0 #mT
        η=0.000865 #Pa s
        a=2.0 #um 

        #k_scale=1/((( 3.0 * 10^-14 ) * (B * 10^-3))/(6 * η * π * a * (10^-6) * ( (72*.58) * (10^-6))^2   ))
        #f_scale=1/(f)
        #t_scale=max(f_scale,k_scale) #set some sort of bound. 
        #times=particle_full[1:end,3].* 1.0/ fps
        #times=times.-minimum(times)

        START =length(particle_full)÷3 # findfirst(x -> x >= t_scale, times)

        push!(all_priors,-3.1415/2.) 
        push!(all_priors,particle_full[START][1]) #dropc x
        push!(all_priors,particle_full[START][2]) #dropc y
        
        
        push!(all_bounds,[-pi,pi])
        push!(all_bounds,[particle_full[START][1]-5,particle_full[START][1]+5])
        push!(all_bounds,[particle_full[START][2]-5,particle_full[START][2]+5])

        push!(all_priors,3.14) #angle of particle 
        push!(all_bounds,[-2.0*3.1415,4.0*3.1415])
    end
    push!(all_priors, 0.0) #Θg 
    push!(all_bounds, [-0.2,0.2])
 
    push!(all_priors, 0.0) #Φg 
    push!(all_bounds, [-0.2,0.2])
    
    push!(all_priors, 0.0) #Θ 
    push!(all_bounds, [-0.2,0.2])

    push!(all_priors, 0.0) #Φ 
    push!(all_bounds, [-0.2,0.2])
    
    push!(all_priors, 0.0) #σ
    push!(all_bounds,[-5.0,1.0]) 
        
    push!(all_priors,2.0) #p
    push!(all_bounds,[-10.0,10.0])

    push!(all_priors,-2.0) #mg 
    push!(all_bounds,[-15.0,1.0]) #mg 

    push!(all_priors,0.0) #μ 
    push!(all_bounds,[-3.1415/2.0,3.1415/2.0])    
    
    push!(all_priors, 0.5) #m
    push!(all_bounds,[-15.0,1.0])
    
    push!(all_priors,68.0) #R
    push!(all_bounds,[45.0,85.0])
    return [all_priors, [x[1] for x in all_bounds],[x[2] for x in all_bounds]]
end


function getpar(x,model,ALL_RUNS)
    function getH(par)
        t_LL=0
        for e in enumerate(ALL_RUNS)
            t_LL += -eval_one_exp(e,par) 
        end
        t_LL += getPriorSO(par,model,ALL_RUNS)
        return t_LL
    end
    return(getH(par_map(x,model,ALL_RUNS)))
end



function get_parG!(storage,x,model,ALL_RUNS)
    function getH(par)
        t_LL=0
        for e in enumerate(ALL_RUNS)
            t_LL += -eval_one_exp(e,par) 
        end
        t_LL += getPriorSO(par,model,ALL_RUNS)
        return t_LL
    end
    #cfg  = ForwardDiff.GradientConfig(getH,par_map(x,model,ALL_RUNS),ForwardDiff.Chunk{1}())
    G = ForwardDiff.gradient(getH,par_map(x,model,ALL_RUNS))#,cfg)
    storage[1:end] = G[Int.(par_umap(1:N_LOCAL*length(ALL_RUNS)+N_GLOBAL,model,ALL_RUNS))][1:end] 
end



function get_nlopt_res_exp_apg(model,n,newstart=false; xtola=1e-6,xtolr=1e-6,usegrad=false)

    lower,upper=0,0
    ALL_RUNS=get_batch(n)
    
    x0,lower,upper=[par_umap(x,model,ALL_RUNS) for x in get_priors(ALL_RUNS)]
    #x0 = par_umap(mu,model,ALL_RUNS)
    
    if newstart != false 
        x0 = newstart
        #print(".")
    end

    for (i,II) in enumerate(x0)
        if x0[i] < lower[i]
            x0[i] = lower[i]
        end
        if x0[i] > upper[i]
            x0[i] = upper[i]
        end
    end

    
    
    function g!(storage,x) 
        return get_parG!(storage,x,model,ALL_RUNS)
    end
    
    function f(x,grad=[],p=0)
        start=CPUTime.time()
        #println(x)

        r = getpar(x,model,ALL_RUNS) 
        if length(grad) > 0
            g!(grad,x)
        end
        return r 
    end



    fx0=f(x0)
    
    n_par=length(x0)

    
    method = :LN_SBPLX
    if usegrad
        method = :LD_LBFGS
    end
    
    opt=NLopt.Opt(method ,n_par)

    
    opt.min_objective = f
    opt.xtol_abs = xtola
    opt.xtol_rel = xtolr
    
    opt.lower_bounds=lower
    opt.upper_bounds=upper
    opt.maxtime = 600

    (minf,minx,ret) = NLopt.optimize(opt, x0) #Optim.minimizer(tuned)
    print(minf,minx,ret)
    open("tmp_out/"*basename_  *randstring(100)* "_"*model*"_" * string(n)*".json","w") do f
        JSON.print(f,[minx,minf,x0,fx0,ret])
    end    
    return minx   
end
function get_batch(n)
    ALL_RUNS=0# 
    
    if n<0
        n = -1*n
        ALL_RUNS=[-1*x for x in shuff_list[n*1+1:(n+1)*1] ]
    else
        ALL_RUNS=shuff_list[n*4+1:(n+1)*4]
    end 
    return ALL_RUNS
end




function postprocess_models(models, batches, basename_)
    for m in models
        for N in batches #0:last_n_batch
            n = string(N)

            start_pos = []
            end_pos = []
            start_val = []
            end_val = []

            for fn in Glob.glob("tmp_out/" * basename_ * "*" * "_" * m * "_" * n * ".json")
                l = JSON.parse(read(fn, String))
                push!(start_pos, l[3])
                push!(start_val, l[4])
                push!(end_pos, l[1])
                push!(end_val, l[2])
            end

            start_pos = Transpose(hcat(start_pos...))
            end_pos = Transpose(hcat(end_pos...))

            best_pos = end_pos[argmin(end_val), 1:end]
            best_val = end_val[argmin(end_val)]

            open(basename_ * "_" * m * "_" * n * ".json", "w") do f
                JSON.print(f, [Dict("exps" => get_batch(N), "res" => best_pos, "fitval" => best_val)])
            end
        end
    end
end
function model_b_is_subset_of_a(a,b)
    set_str1 = Set(a)
    set_str2 = Set(b)
    return issubset(set_str2, set_str1)
end




function gen_list_to_run(models,batches,basename_)
    
    all_to_test=[]
    all_to_run=[]

    for exp in batches 
        for model in models
            pre_models=[]
            for possible_pre in ["ΦΘϕ","aΦΘϕ","pΦΘϕ","gΦΘϕ","g12ΦΘϕ","agΦΘϕ","ag12ΦΘϕ","pgΦΘϕ","pg12ΦΘϕ","apΦΘϕ","apgΦΘϕ","apg12ΦΘϕ"]
                if possible_pre == model
                    continue
                end
                if model_b_is_subset_of_a(model,possible_pre)
                    push!(pre_models,possible_pre)
                end
            end
            push!(all_to_test,[model*"",exp])
 
            best_pre=""
            best_score=Inf 
 
            for pre_model in pre_models
                try
                    val=JSON.parse(read(basename_*"_"*pre_model*""*"_" * string(exp) * ".json", String))[1]["fitval"]
                    if val < best_score
                        best_pre = pre_model
                        best_score = val
                    end
                catch
                    println("can't load pre_file!")
                end
            end
            if pre_models == []
                continue
            end
            
            try
                val=JSON.parse(read(basename_*"_"*best_pre*""*"_" * string(exp) * ".json", String))[1]["fitval"]
                par=JSON.parse(read(basename_*"_"*best_pre*""*"_" * string(exp) * ".json", String))[1]["res"]
                par_new=par_umap(par_map(Float64.(par),best_pre,get_batch(0)),model,get_batch(0)) 
                push!(all_to_run,[model,exp,Float64.(par_new)])
            catch
                println("can't load pre_file!(2)")
            end
        end
    end


    for pair in all_to_test
        all_pars=par_umap(get_priors(get_batch(pair[2]))[1], pair[1],get_batch(pair[2]))
        mu,lower,upper=get_priors_inner(get_batch(pair[2]))
        mu=par_umap(mu,pair[1],get_batch(pair[2]))
        lower=par_umap(lower,pair[1],get_batch(pair[2]))
        upper=par_umap(upper,pair[1],get_batch(pair[2]))
        sigmas = par_umap(get_priors_sigma(get_batch(pair[2])),pair[1],get_batch(pair[2]))
        include_in_LHC = par_umap(include_par_in_LHC(get_batch(pair[2])),pair[1],get_batch(pair[2]))



        filtered_indices = filter(i -> include_in_LHC[i], 1:length(include_in_LHC)) #only values we want to LHC scale 
        fidx_dict=Dict(index => i for (i, index) in enumerate(filtered_indices))
        bounds = collect(zip(lower[filtered_indices], upper[filtered_indices]))
        scaled_plan = scaleLHC(plan_by_size[sum(include_in_LHC)],bounds)

        println(size(scaled_plan))
        println(scaled_plan[1,1:end])

        for pt_i in 1:size(scaled_plan)[1]
            pt=scaled_plan[pt_i,1:end]
            par=[]
            for (pt_i_j,(mu_l,sig_l,flag_i)) in enumerate(zip(mu,sigmas,include_in_LHC))
                if flag_i
                    push!(par,pt[fidx_dict[pt_i_j]])
                else
                    push!(par,randn()*sig_l+mu_l)
                end
            end
            push!(all_to_run,[pair[1],pair[2],Float64.(par)])
        end
    end 

    return all_to_run

end