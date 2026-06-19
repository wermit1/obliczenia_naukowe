using Pkg
Pkg.add(["Plots", "LinearAlgebra", "BenchmarkTools", "FFTW", "Dates", "DelimitedFiles"])

using Plots
using LinearAlgebra
using BenchmarkTools
using FFTW
using Dates
using DelimitedFiles

# Ustawienia wykresów
default(linewidth=2, framestyle=:grid, titlefontsize=10)
gr()
ENV["GKSwstype"] = "nul"

println("Pakiety załadowane.")

# ============================================
# FOLDER NA WYNIKI
# ============================================
results_dir = joinpath(@__DIR__, "wyniki_projektu1")
if !isdir(results_dir)
    mkdir(results_dir)
    println("Utworzono folder: $results_dir")
else
    println("Folder istnieje: $results_dir")
end

# Stała grawitacji
const G = 0.000295912208
println("G = $G")

# ============================================
# FUNKCJE PODSTAWOWE
# ============================================

function compute_accelerations(positions, masses, G, softening=0.0)
    N = length(masses)
    acc = zeros(3, N)
    
    for i in 1:N
        for j in i+1:N
            dr = positions[:, j] - positions[:, i]
            r2 = dot(dr, dr) + softening^2
            r = sqrt(r2)
            r3 = r2 * r
            factor = G / r3
            
            acc[:, i] += factor * masses[j] * dr
            acc[:, j] -= factor * masses[i] * dr
        end
    end
    return acc
end
println("Funkcja przyspieszeń zdefiniowana.")

function total_energy(positions, velocities, masses, G)
    N = length(masses)
    Ek = 0.5 * sum(masses[i] * norm(velocities[:, i])^2 for i in 1:N)
    Ep = 0.0
    
    for i in 1:N
        for j in i+1:N
            dr = norm(positions[:, j] - positions[:, i])
            Ep -= G * masses[i] * masses[j] / dr
        end
    end
    return Ek + Ep
end
println("Funkcja energii zdefiniowana.")

function total_angular_momentum(positions, velocities, masses)
    L = zeros(3)
    for i in 1:length(masses)
        L += masses[i] * cross(positions[:, i], velocities[:, i])
    end
    return L
end
println("Funkcja momentu pędu zdefiniowana.")

# ============================================
# INTEGRATORY
# ============================================

function euler_step(r, v, masses, G, dt, softening=0.0)
    a = compute_accelerations(r, masses, G, softening)
    return r + v*dt, v + a*dt
end

function rk4_step(r, v, masses, G, dt, softening=0.0)
    a1 = compute_accelerations(r, masses, G, softening)
    r2 = r + 0.5*dt*v
    v2 = v + 0.5*dt*a1
    a2 = compute_accelerations(r2, masses, G, softening)
    r3 = r + 0.5*dt*v2
    v3 = v + 0.5*dt*a2
    a3 = compute_accelerations(r3, masses, G, softening)
    r4 = r + dt*v3
    v4 = v + dt*a3
    a4 = compute_accelerations(r4, masses, G, softening)
    r_new = r + dt/6*(v + 2*v2 + 2*v3 + v4)
    v_new = v + dt/6*(a1 + 2*a2 + 2*a3 + a4)
    return r_new, v_new
end

function verlet_step(r, v, masses, G, dt, softening=0.0)
    a = compute_accelerations(r, masses, G, softening)
    r_new = r + v*dt + 0.5*a*dt^2
    a_new = compute_accelerations(r_new, masses, G, softening)
    v_new = v + 0.5*(a + a_new)*dt
    return r_new, v_new
end

function leapfrog_step(r, v, masses, G, dt, softening=0.0)
    a = compute_accelerations(r, masses, G, softening)
    v_half = v + 0.5*dt*a
    r_new = r + dt*v_half
    a_new = compute_accelerations(r_new, masses, G, softening)
    v_new = v_half + 0.5*dt*a_new
    return r_new, v_new
end

function forest_ruth_step(r, v, masses, G, dt, softening=0.0)
    c1 = 0.6756035959798288
    c2 = -0.1756035959798288
    c3 = -0.1756035959798288
    c4 = 0.6756035959798288
    d1 = 1.3512071919596576
    d2 = -0.7024143839193152
    d3 = -0.7024143839193152
    d4 = 1.3512071919596576
    
    a = compute_accelerations(r, masses, G, softening)
    v1 = v + c1 * dt * a
    r1 = r + d1 * dt * v1
    a1 = compute_accelerations(r1, masses, G, softening)
    v2 = v1 + c2 * dt * a1
    r2 = r1 + d2 * dt * v2
    a2 = compute_accelerations(r2, masses, G, softening)
    v3 = v2 + c3 * dt * a2
    r3 = r2 + d3 * dt * v3
    a3 = compute_accelerations(r3, masses, G, softening)
    v_new = v3 + c4 * dt * a3
    r_new = r3 + d4 * dt * v_new
    return r_new, v_new
end

println("Wszystkie integratory zdefiniowane (Euler, RK4, Verlet, Leapfrog, Forest-Ruth).")

# ============================================
# SYMULACJA
# ============================================

function run_simulation(integrator, name, r0, v0, masses, G, dt, n_steps, softening=0.0)
    N = length(masses)
    r = copy(r0)
    v = copy(v0)
    
    history_r = zeros(3, N, n_steps+1)
    history_v = zeros(3, N, n_steps+1)
    history_E = zeros(n_steps+1)
    history_L = zeros(n_steps+1)
    
    history_r[:, :, 1] = r
    history_v[:, :, 1] = v
    history_E[1] = total_energy(r, v, masses, G)
    history_L[1] = norm(total_angular_momentum(r, v, masses))
    
    for step in 1:n_steps
        r, v = integrator(r, v, masses, G, dt, softening)
        history_r[:, :, step+1] = r
        history_v[:, :, step+1] = v
        history_E[step+1] = total_energy(r, v, masses, G)
        history_L[step+1] = norm(total_angular_momentum(r, v, masses))
    end
    
    println("Symulacja $name zakończona. dt=$dt, n_steps=$n_steps")
    return (r=history_r, v=history_v, E=history_E, L=history_L, t=(0:n_steps)*dt, name=name)
end
println("Funkcja run_simulation zdefiniowana.")

# ============================================
# WARUNKI POCZĄTKOWE
# ============================================

masses = [1.0, 3.003e-6, 3.227e-7, 4.731e-7, 2.528e-7]

positions0 = [
    0.0 0.0 0.0;
    0.999978 0.005118 0.0;
    1.523679 0.0 0.034;
    0.728213 0.0 -0.022;
    0.307499 0.0 -0.057
]'

velocities0 = [
    0.0 0.0 0.0;
    -0.005018 0.017202 0.0;
    0.0 0.01397 0.0012;
    0.0 0.02078 -0.0008;
    0.0 0.02788 -0.0075
]'

println("Warunki początkowe zdefiniowane.")

# ============================================
# 1. KATASTROFA - za duży dt
# ============================================
println("\n=== 1. KATASTROFA NUMERYCZNA: za duży dt ===")
res_bad = run_simulation(rk4_step, "RK4 (złe dt)", positions0, velocities0, masses, G, 10.0, 500)

p1 = plot(res_bad.E .- res_bad.E[1], title="KATASTROFA NUMERYCZNA - dt = 10 dni", xlabel="Krok", ylabel="ΔE", label="Energia rośnie", linewidth=2)
savefig(p1, joinpath(results_dir, "katastrofa_dt10.png"))
println("Zapisano: katastrofa_dt10.png")

# ============================================
# 2. PORÓWNANIE RK4 vs VERLET
# ============================================
println("\n=== 2. PORÓWNANIE RK4 vs VERLET ===")
dt_good = 0.5
n_steps_good = 2000

res_RK4 = run_simulation(rk4_step, "RK4", positions0, velocities0, masses, G, dt_good, n_steps_good)
res_Verlet = run_simulation(verlet_step, "Verlet", positions0, velocities0, masses, G, dt_good, n_steps_good)

E0 = res_RK4.E[1]

p2 = plot(res_RK4.t, res_RK4.E ./ E0 .- 1, label="RK4", linewidth=2)
plot!(p2, res_Verlet.t, res_Verlet.E ./ E0 .- 1, label="Verlet", linewidth=2)
title!(p2, "Zachowanie energii - porównanie")
xlabel!(p2, "Czas [dni]")
ylabel!(p2, "Względna zmiana energii")
savefig(p2, joinpath(results_dir, "energia_RK4_vs_Verlet.png"))
println("Zapisano: energia_RK4_vs_Verlet.png")

# ============================================
# 3. ANIMACJA ORBIT 2D
# ============================================
println("\n=== 3. ANIMACJA ORBIT 2D ===")

anim = @animate for step in 1:50:n_steps_good+1
    plot(title="Orbity planet (RK4)", xlim=(-1.8, 1.8), ylim=(-1.8, 1.8), aspect_ratio=:equal)
    scatter!([0], [0], color=:yellow, markersize=10, label="Słońce")
    scatter!([res_RK4.r[1,2,step]], [res_RK4.r[2,2,step]], color=:blue, markersize=6, label="Ziemia")
    scatter!([res_RK4.r[1,3,step]], [res_RK4.r[2,3,step]], color=:red, markersize=5, label="Mars")
    scatter!([res_RK4.r[1,4,step]], [res_RK4.r[2,4,step]], color=:orange, markersize=5, label="Wenus")
    scatter!([res_RK4.r[1,5,step]], [res_RK4.r[2,5,step]], color=:gray, markersize=4, label="Merkury")
end

gif(anim, joinpath(results_dir, "orbity_planet.gif"), fps=20)
println("Zapisano: orbity_planet.gif")

# ============================================
# 4. MOMENT PĘDU
# ============================================
println("\n=== 4. MOMENT PĘDU ===")

p4 = plot(res_RK4.t, (res_RK4.L .- res_RK4.L[1])./res_RK4.L[1], label="RK4", title="Zachowanie momentu pędu")
plot!(p4, res_Verlet.t, (res_Verlet.L .- res_Verlet.L[1])./res_Verlet.L[1], label="Verlet", linewidth=2)
xlabel!(p4, "Czas [dni]")
ylabel!(p4, "Δ|L|/|L₀|")
savefig(p4, joinpath(results_dir, "moment_pedu.png"))
println("Zapisano: moment_pedu.png")

# ============================================
# 5. FFT SPEKTRUM MOCY
# ============================================
println("\n=== 5. SPEKTRUM MOCY FFT ===")

x_earth = [res_RK4.r[1, 2, i] for i in 1:n_steps_good+1]
fs = 1 / dt_good
freqs = fftfreq(length(x_earth), fs)
power = abs.(fft(x_earth)).^2
idx = freqs .> 0

p5 = plot(freqs[idx], power[idx], linewidth=1, title="Spektrum mocy – orbita Ziemi")
vline!([1/365.25], label="1 rok⁻¹", linestyle=:dash, linewidth=2)
xlabel!("Częstotliwość [1/dzień]")
ylabel!("Moc")
xlims!(0, 0.01)
savefig(p5, joinpath(results_dir, "spektrum_mocy.png"))
println("Zapisano: spektrum_mocy.png")

# ============================================
# 6. SOFTENING
# ============================================
println("\n=== 6. SOFTENING ===")

softening_vals = [0.0, 0.001, 0.01, 0.1]
soft_results = []

for eps in softening_vals
    println("Test softening = $eps")
    r = copy(positions0)
    v = copy(velocities0)
    E_hist = [total_energy(r, v, masses, G)]
    for _ in 1:5000
        a = compute_accelerations(r, masses, G, eps)
        v += a
        r += v
        push!(E_hist, total_energy(r, v, masses, G))
    end
    push!(soft_results, E_hist)
end

p6 = plot(soft_results[1] .- soft_results[1][1], label="soft=0.0 (brak)", title="Wpływ softening na stabilność")
for i in 2:4
    plot!(p6, soft_results[i] .- soft_results[i][1], label="soft=$(softening_vals[i])")
end
xlabel!("Krok")
ylabel!("ΔE")
savefig(p6, joinpath(results_dir, "softening.png"))
println("Zapisano: softening.png")

# ============================================
# 7. KATASTROFA - bliski przelot
# ============================================
println("\n=== 7. KATASTROFA: bliski przelot ===")
positions_close = copy(positions0)
positions_close[:, 3] = [0.1, 0.0, 0.0]
res_close = run_simulation(rk4_step, "close", positions_close, velocities0, masses, G, 0.5, 1000)

p7 = plot(res_close.E .- res_close.E[1], title="Katastrofa: bliski przelot", xlabel="Krok", ylabel="ΔE", label="Energia wybucha", linewidth=2)
savefig(p7, joinpath(results_dir, "katastrofa_bliski_przelot.png"))
println("Zapisano: katastrofa_bliski_przelot.png")

# ============================================
# 8. ANALIZA BŁĘDU W ZALEŻNOŚCI OD DT
# ============================================
println("\n=== 8. ANALIZA BŁĘDU W ZALEŻNOŚCI OD DT ===")

dt_values = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
errors_rk4 = Float64[]
errors_verlet = Float64[]

for dt_test in dt_values
    n_steps_test = Int(round(365 / dt_test))
    res_rk4_test = run_simulation(rk4_step, "RK4_test", positions0, velocities0, masses, G, dt_test, n_steps_test)
    res_verlet_test = run_simulation(verlet_step, "Verlet_test", positions0, velocities0, masses, G, dt_test, n_steps_test)
    
    E0_test = res_rk4_test.E[1]
    err_rk4 = abs((res_rk4_test.E[end] - E0_test) / E0_test)
    err_verlet = abs((res_verlet_test.E[end] - E0_test) / E0_test)
    
    push!(errors_rk4, err_rk4)
    push!(errors_verlet, err_verlet)
    
    println("dt=$dt_test dni: RK4 błąd=$(round(err_rk4, digits=8)), Verlet błąd=$(round(err_verlet, digits=8))")
end

p8 = plot(dt_values, errors_rk4, marker=:circle, label="RK4", linewidth=2, 
          title="Błąd energii po 1 roku symulacji", 
          xlabel="Krok czasowy dt [dni]", ylabel="|ΔE/E₀|", yscale=:log10)
plot!(p8, dt_values, errors_verlet, marker=:square, label="Verlet", linewidth=2, yscale=:log10)
savefig(p8, joinpath(results_dir, "error_vs_dt.png"))
println("Zapisano: error_vs_dt.png")

# ============================================
# 9. BENCHMARK
# ============================================
println("\n=== 9. BENCHMARK ===")
r_test = copy(positions0)
v_test = copy(velocities0)
println("Benchmark Verleta (1000 kroków):")
b = @benchmark verlet_step($r_test, $v_test, $masses, $G, 0.5)
display(b)

open(joinpath(results_dir, "benchmark.txt"), "w") do f
    write(f, string(b))
end
println("Zapisano: benchmark.txt")

# ============================================
# 10. ZAPIS DANYCH DO CSV
# ============================================
writedlm(joinpath(results_dir, "energia_RK4.csv"), [res_RK4.t res_RK4.E], ',')
writedlm(joinpath(results_dir, "energia_Verlet.csv"), [res_Verlet.t res_Verlet.E], ',')
println("Zapisano CSV")

# ============================================
# 11. WYKŁADNIK LAPUNOWA
# ============================================
println("\n=== 11. WYKŁADNIK LAPUNOWA ===")

function lyapunov_exponent_simple(dt=1.0, total_time=5000.0, eps=1e-8)
    r_sun = [0.0, 0.0, 0.0]
    v_sun = [0.0, 0.0, 0.0]
    r_earth = [1.0, 0.0, 0.0]
    v_earth = [0.0, 0.017202, 0.0]
    
    r_earth_pert = [1.0 + eps, 0.0, 0.0]
    v_earth_pert = [0.0, 0.017202, 0.0]
    
    n_steps = floor(Int, total_time / dt)
    d = zeros(n_steps + 1)
    times = zeros(n_steps + 1)
    d[1] = eps
    times[1] = 0.0
    
    for step in 1:n_steps
        times[step+1] = step * dt
        
        dr = r_earth - r_sun
        r3 = norm(dr)^3
        a_earth = -G * 1.0 * dr / r3
        v_earth += a_earth * dt
        r_earth += v_earth * dt
        
        dr_pert = r_earth_pert - r_sun
        r3_pert = norm(dr_pert)^3
        a_earth_pert = -G * 1.0 * dr_pert / r3_pert
        v_earth_pert += a_earth_pert * dt
        r_earth_pert += v_earth_pert * dt
        
        dr_norm = norm(r_earth_pert - r_earth)
        d[step+1] = dr_norm
        
        if dr_norm > 1e-5
            r_earth_pert = r_earth + (r_earth_pert - r_earth) * (eps / dr_norm)
        end
    end
    
    log_d = log.(max.(d, 1e-16))
    idx_start = max(1, floor(Int, n_steps / 4))
    
    if n_steps - idx_start > 10
        A = hcat(times[idx_start:end], ones(length(times[idx_start:end])))
        coeffs = A \ log_d[idx_start:end]
        lambda_val = coeffs[1]
    else
        lambda_val = 0.0
    end
    
    p11 = plot(times, log_d, label="log(odległość)", title="Wykładnik Lapunowa - chaos")
    if lambda_val != 0.0
        plot!(p11, times, lambda_val.*times .+ (log_d[end] - lambda_val*times[end]), 
              label="λ=$(round(lambda_val, digits=6))", linestyle=:dash)
    end
    xlabel!("Czas [dni]")
    ylabel!("ln(odległość)")
    savefig(p11, joinpath(results_dir, "lyapunov.png"))
    println("Zapisano: lyapunov.png")
    
    return lambda_val
end

lambda = lyapunov_exponent_simple(1.0, 5000.0, 1e-8)
println("Wykładnik Lapunowa ≈ $lambda 1/dzień")
if lambda > 1e-8
    println("Czas Lapunowa ≈ $(round(1/lambda, digits=1)) dni ≈ $(round(1/lambda/365.25, digits=2)) lat")
end

# ============================================
# 12. MAPA POINCARÉGO
# ============================================
println("\n=== 12. MAPA POINCARÉGO ===")

function poincare_map(n_points=300, dt=0.5)
    m_sun = 1.0
    m_jupiter = 9.548e-4
    m_earth = 0.0
    
    masses_p = [m_sun, m_jupiter, m_earth]
    
    r_sun = [0.0, 0.0, 0.0]
    r_jupiter = [5.203, 0.0, 0.0]
    r_earth = [1.0, 0.0, 0.0]
    
    v_sun = [0.0, 0.0, 0.0]
    v_jupiter = [0.0, 0.00741, 0.0]
    v_earth = [0.0, 0.017202, 0.0]
    
    positions = hcat(r_sun, r_jupiter, r_earth)
    velocities = hcat(v_sun, v_jupiter, v_earth)
    
    points_r = Float64[]
    points_pr = Float64[]
    
    step = 0
    while length(points_r) < n_points && step < 50000
        positions, velocities = rk4_step(positions, velocities, masses_p, G, dt)
        step += 1
        
        r_earth_current = positions[:, 3]
        r_jupiter_current = positions[:, 2]
        
        if abs(r_earth_current[1] - r_jupiter_current[1]) < 0.1 && step % 10 == 0
            r_radial = norm(r_earth_current[1:2])
            if r_radial > 0
                v_radial = (velocities[1, 3] * r_earth_current[1] + velocities[2, 3] * r_earth_current[2]) / r_radial
                push!(points_r, r_radial)
                push!(points_pr, v_radial)
            end
        end
    end
    
    if length(points_r) > 0
        p12 = scatter(points_r, points_pr, markersize=2, alpha=0.5, 
                  title="Mapa Poincarégo - układ Ziemia+Jowisz",
                  xlabel="r [AU]", ylabel="v_r [AU/dzień]", legend=false)
        savefig(p12, joinpath(results_dir, "poincare_map.png"))
        println("Zapisano: poincare_map.png")
    end
    return points_r, points_pr
end

poincare_map(300, 0.5)

# ============================================
# 13. ANIMACJA 3D (Plots, bez GLMakie)
# ============================================
println("\n=== 13. ANIMACJA 3D (Plots) ===")

# Przygotowanie danych do animacji 3D
positions_3d = res_RK4.r
n_frames = size(positions_3d, 3)
frames_3d = 1:50:n_frames

anim3d = @animate for frame in frames_3d
    # Pobierz pozycje planet w danej klatce
    x_sun = 0
    y_sun = 0
    z_sun = 0
    
    x_earth = positions_3d[1, 2, frame]
    y_earth = positions_3d[2, 2, frame]
    z_earth = positions_3d[3, 2, frame]
    
    x_mars = positions_3d[1, 3, frame]
    y_mars = positions_3d[2, 3, frame]
    z_mars = positions_3d[3, 3, frame]
    
    x_venus = positions_3d[1, 4, frame]
    y_venus = positions_3d[2, 4, frame]
    z_venus = positions_3d[3, 4, frame]
    
    x_mercury = positions_3d[1, 5, frame]
    y_mercury = positions_3d[2, 5, frame]
    z_mercury = positions_3d[3, 5, frame]
    
    # Wykres 3D
    plot3d(1, title="Orbity planet 3D", xlim=(-2, 2), ylim=(-2, 2), zlim=(-1, 1),
           xlabel="x [AU]", ylabel="y [AU]", zlabel="z [AU]",
           camera=(30, 30))
    
    # Słońce
    scatter3d!([x_sun], [y_sun], [z_sun], color=:yellow, markersize=10, label="Słońce")
    # Ziemia
    scatter3d!([x_earth], [y_earth], [z_earth], color=:blue, markersize=6, label="Ziemia")
    # Mars
    scatter3d!([x_mars], [y_mars], [z_mars], color=:red, markersize=5, label="Mars")
    # Wenus
    scatter3d!([x_venus], [y_venus], [z_venus], color=:orange, markersize=5, label="Wenus")
    # Merkury
    scatter3d!([x_mercury], [y_mercury], [z_mercury], color=:gray, markersize=4, label="Merkury")
end

gif(anim3d, joinpath(results_dir, "orbity_3d.gif"), fps=20)
println("Zapisano: orbity_3d.gif (animacja 3D)")

# ============================================
# 14. TABELA PORÓWNAWCZA
# ============================================
println("\n" * "="^70)
println("TABELA PORÓWNAWCZA INTEGRATORÓW")
println("="^70)
println("| Metoda       | Rząd | Symplektyczna | Zachowanie energii | Uwagi                     |")
println("|--------------|------|---------------|--------------------|---------------------------|")
println("| Euler        | 1    | Nie           | ❌ Katastrofa      | Nie używać                |")
println("| RK4          | 4    | Nie           | ⚠️ Dryf           | Krótkie symulacje         |")
println("| Verlet       | 2    | ✅ Tak         | ✅ Stabilna        | Długie symulacje ★★★      |")
println("| Leapfrog     | 2    | ✅ Tak         | ✅ Stabilna        | Astrofizyka               |")
println("| Forest-Ruth  | 4    | ✅ Tak         | ✅✅ Bardzo dobra  | Wysoka precyzja           |")
println("="^70)

# ============================================
# 15. PRAKTYCZNE ZNACZENIE
# ============================================
println("\n" * "="^70)
println("PRAKTYCZNE ZNACZENIE")
println("="^70)
println("1. Misje kosmiczne (NASA/ESA) – asysty grawitacyjne")
println("2. Odkrywanie egzoplanet – symulacje stabilności")
println("3. Mechanika nieba – przewidywanie orbit")
println("4. Astrofizyka – symulacje gromad galaktyk")
println("="^70)

# ============================================
# 16. PODSUMOWANIE PLIKÓW
# ============================================
println("\n" * "="^70)
println("WSZYSTKIE WYNIKI ZAPISANO W FOLDERZE:")
println("   $results_dir")
println("="^70)
println("Zawartość folderu:")
for file in readdir(results_dir)
    println("   - $file")
end
println("\n=== WSZYSTKO GOTOWE ===")