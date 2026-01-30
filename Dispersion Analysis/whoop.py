def drogue_trigger(p, h, y):
    # Try accessing y as an array first
    try:
        if len(y) > 5:  # Check if y has enough elements
            return y[5] < 0  # Negative vertical velocity
    except:
        pass
    
    # Fallback: trigger based on time or altitude
    return h > 10000  # Adjust as needed