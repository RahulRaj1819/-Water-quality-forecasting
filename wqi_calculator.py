def analyze_water_quality(ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity):
    """
    Calculates Water Quality Index (0-100) and Parameter Impacts.
    """
    parameters = {
        "pH": {"value": ph, "ideal": 7.0, "min": 6.5, "max": 8.5, "weight": 0.15},
        "Hardness": {"value": hardness, "ideal": 120, "min": 60, "max": 200, "weight": 0.05},
        "Solids": {"value": solids, "ideal": 500, "min": 0, "max": 1000, "weight": 0.10},
        "Chloramines": {"value": chloramines, "ideal": 2, "min": 0, "max": 4, "weight": 0.15},
        "Sulfate": {"value": sulfate, "ideal": 150, "min": 0, "max": 250, "weight": 0.10},
        "Conductivity": {"value": conductivity, "ideal": 200, "min": 0, "max": 400, "weight": 0.10},
        "Organic Carbon": {"value": organic_carbon, "ideal": 2, "min": 0, "max": 4, "weight": 0.10},
        "Trihalomethanes": {"value": trihalomethanes, "ideal": 40, "min": 0, "max": 80, "weight": 0.15},
        "Turbidity": {"value": turbidity, "ideal": 1, "min": 0, "max": 5, "weight": 0.10},
    }
    
    total_score = 0
    impacts = []
    
    for name, p in parameters.items():
        v = p["value"]
        mi = p["min"]
        ma = p["max"]
        ideal = p["ideal"]
        
        if mi <= v <= ma:
            # Good range
            range_span = ma - mi if ma != mi else 1
            score = 100 - (abs(v - ideal) / range_span) * 50
            status = "Good"
            color = "#34d399" # green
        else:
            # Bad range
            if v < mi:
                deviation = (mi - v) / mi if mi != 0 else 1
            else:
                deviation = (v - ma) / ma if ma != 0 else 1
            score = max(0, 50 - deviation * 100)
            status = "Poor"
            color = "#f87171" # red
            
        score = min(100, max(0, score))
        total_score += score * p["weight"]
        
        impacts.append({
            "name": name,
            "value": v,
            "score": score,
            "status": status,
            "color": color,
            "impact": "Positive" if score > 70 else "Negative" if score < 40 else "Neutral",
            "deviation_pct": abs(score - 100) # For sorting
        })
        
    wqi = min(100, max(0, total_score))
    
    # Sort impacts by how negative they are
    impacts.sort(key=lambda x: x["score"])
    
    return round(wqi, 1), impacts
