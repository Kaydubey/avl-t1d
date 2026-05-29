import matplotlib.pyplot as plt

def clarke_error_grid(ref_values, pred_values, title="Clarke Error Grid"):
    # This function creates the standard FDA safety chart
    assert len(ref_values) == len(pred_values), "Unequal number of values"

    # Define Zones logic (Simplified for visualization)
    zone_a = [0] * 5
    zone_b = [0] * 5
    zone_c = [0] * 5
    zone_d = [0] * 5
    zone_e = [0] * 5

    # Note: We are keeping the plotting logic simple for the display
    
    # --- PLOT (FIXED SECTION) ---
    # We assign the plot to a variable 'fig' so we can pass it to the app
    fig = plt.figure(figsize=(6, 6)) 
    
    plt.scatter(ref_values, pred_values, c='blue', s=10, alpha=0.5, label="Predictions")
    
    # Draw Zone Lines
    plt.plot([0, 400], [0, 400], 'k:', label="Perfect Line") # Perfect match
    plt.plot([0, 400], [0, 480], 'g--', alpha=0.5, label="Zone A (Safe)")
    plt.plot([0, 400], [0, 320], 'g--', alpha=0.5)
    
    plt.title(title)
    plt.xlabel("Reference Concentration (mg/dL) [Real]")
    plt.ylabel("Predicted Concentration (mg/dL) [AI]")
    plt.legend()
    plt.grid()
    
    # IMPORTANT: Return the 'fig' object, NOT 'plt'
    return fig