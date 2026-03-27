import numpy as np

rho = 1000  # water density kg/m^3
A = 0.045   # frontal area m^2

def drag(v, Cd):
    return 0.5 * rho * v**2 * Cd * A

def model(params):
    v = params["velocity"]
    flex = params["flex"]          # 0–0.3
    feather = params["feather"]    # degrees
    coating = params["coating"]    # 0 none, 1 ribbed, 2 gel
    
    # baseline drag coefficient
    Cd = 0.9
    
    # modifiers
    Cd *= (1 - 0.25*flex)
    Cd *= (1 - 0.01*feather)
    
    if coating == 1:
        Cd *= 0.85
    elif coating == 2:
        Cd *= 0.75
    
    return drag(v, Cd)

# sweep
velocities = np.linspace(0.5, 3.0, 6)
for v in velocities:
    result = model({"velocity": v, "flex": 0.25, "feather": 20, "coating": 2})
    print(f"Velocity {v:.2f} m/s -> Drag {result:.2f} N")
