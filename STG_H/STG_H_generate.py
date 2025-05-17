import random
import numpy as np
import json


# --- Data Generation Process (DGP) ---
import random

def simulate_air_quality_scenario(ood_mode=False):
    # --- Step 0: Define uniform ranges for causal factors ---
    f1_min, f1_max = 30, 70     
    f2_min, f2_max = 20, 60    
    f3_min, f3_max = 10, 50     
    f4_min, f4_max = 10, 30     
    f5_min, f5_max = 40, 80     
    f6_min, f6_max = 50, 90    
    # --- Step 1: Sample causal variables ---
    f1 = random.uniform(f1_min, f1_max) + random.normalvariate(0, 2)
    f2 = random.uniform(f2_min, f2_max) + random.normalvariate(0, 2)
    f3 = random.uniform(f3_min, f3_max) + random.normalvariate(0, 1.5)
    f4 = random.uniform(f4_min, f4_max) + random.normalvariate(0, 1)
    f5 = random.uniform(f5_min, f5_max) + random.normalvariate(0, 3)
    f6 = random.uniform(f6_min, f6_max) + random.normalvariate(0, 3)
    # --- Step 2: Generate AQI ---
    raw_aqi = (
            0.3 * f1 +
            0.25 * f2 +
            0.2 * f3 +
            0.15 * f4 -
            0.2 * f5 -
            0.3 * f6
    )
    # Normalize AQI to [0, 100] and round
    raw_min, raw_max = -20, 30  # Estimated AQI bounds
    clipped = max(min(raw_aqi, raw_max), raw_min)
    aqi = round((clipped - raw_min) / (raw_max - raw_min) * 100)
    # --- Step 3: Generate spurious (non-causal but correlated) variables ---
    if not ood_mode:
        se1 = 0.4 * f6 + random.normalvariate(0, 2)
        se2 = 0.2 * f4 + random.normalvariate(0, 3)
        se3 = 1.1 * f1 + random.normalvariate(0, 1.5)
        se4 = 0.15 * f2 + random.normalvariate(0, 2)
        se5 = -0.1 * f5 + random.normalvariate(0, 1)
    else:
        
        se1 = random.uniform(0.1 * f6_min, 0.1 * f6_max)
        se2 = random.uniform(0.2 * f4_min, 0.2 * f4_max)
        se3 = random.uniform(0.1 * f1_min, 0.1 * f1_max)
        se4 = random.uniform(0.15 * f2_min, 0.15 * f2_max)
        se5 = random.uniform(-0.1 * f5_max, -0.1 * f5_min)
    # --- Step 4: Irrelevant variables (for noise/distraction) ---
    ir1 = random.randint(1, 100)                          
    ir2 = random.uniform(80, 120) + random.normalvariate(0, 3)  
    ir3 = random.randint(10, 50)                         
    return {
        "F1_IndusEmit": f1,
        "F2_VehicEmit": f2,
        "F3_EnergyCoalDep": f3,
        "F4_AgriBurn": f4,
        "F5_UrbanGreen": f5,
        "F6_MeteoDisp": f6,
        "SE1_IceCreamSales": se1,
        "SE2_AllergyComplaints": se2,
        "SE3_RenewStockIndex": se3,
        "SE4_PublicTransSub": se4,
        "SE5_WaterAwareScore": se5,
        "IR1_CityAnniversary": ir1,
        "IR2_AvgNetSpeed": ir2,
        "IR3_NumLibraries": ir3,
        "AQI": aqi
    }



# --- Natural Language Generation (English, with Numerical Values, Single Template) ---

def generate_natural_language_question(factors_data):
    # Extract values
    f1, f2, f3, f4, f5, f6 = (
        factors_data["F1_IndusEmit"],
        factors_data["F2_VehicEmit"],
        factors_data["F3_EnergyCoalDep"],
        factors_data["F4_AgriBurn"],
        factors_data["F5_UrbanGreen"],
        factors_data["F6_MeteoDisp"]
    )
    se1, se2, se3, se4, se5 = (
        factors_data["SE1_IceCreamSales"],
        factors_data["SE2_AllergyComplaints"],
        factors_data["SE3_RenewStockIndex"],
        factors_data["SE4_PublicTransSub"],
        factors_data["SE5_WaterAwareScore"]
    )
    ir1_anniv, ir2_netspeed, ir3_libs = (
        factors_data["IR1_CityAnniversary"],
        factors_data["IR2_AvgNetSpeed"],
        factors_data["IR3_NumLibraries"]
    )

    # Constructing phrases for causal factors
    sf1_text = f"Industrial Emissions: {f1:.1f}"
    sf2_text = f"Vehicle Emissions: {f2:.1f}"
    sf3_text = f"Coal Dependency: {f3 :.1f}%"
    sf4_text = f"Agricultural Burning: {f4:.1f}"
    sf5_text = f"Urban Green Coverage: {f5 :.1f}%"
    sf6_text = f"Meteorological Dispersion: {f6:.1f}"

    # List of numerical causal factor values for the "important" field (as specified by user)
    important_causal_values = [
        f"{f1:.1f}", f"{f2:.1f}", f"{f3 :.1f}%",
        f"{f4:.1f}", f"{f5 :.1f}%", f"{f6:.1f}"
    ]

    # Compressed descriptive phrases with FULL attribute names for spurious factors
    sse1_text = f"Ice Cream Sales Index: {se1:.1f}"
    sse2_text = f"Allergy Complaints Level: {se2:.1f}"
    sse3_text = f"Renewable Stock Index: {se3:.0f}"  # .0f as it's a larger number index
    sse4_text = f"Public Transport Subscriptions (k): {se4:.1f}"
    sse5_text = f"Water Awareness Score: {se5:.1f}"

    # Compressed descriptive phrases with FULL attribute names for irrelevant factors
    sir1_text = f"City Anniversary: {ir1_anniv}th"
    sir2_text = f"Average Internet Speed: {ir2_netspeed:.1f}Mbps"
    sir3_text = f"Number of Libraries: {ir3_libs}"

    base_question = "Given these diverse conditions, what is the projected Air Quality Index (AQI) on a 0-100 scale?"

    # Ultra-Compressed Single Template with full attribute names
    nl_question = (
        f"{sf1_text}; {sf2_text}; {sf3_text}; {sf4_text}; {sf5_text}; {sf6_text}. "  # Causal factors
        f"{sse1_text}; {sse2_text}; {sse3_text}; {sse4_text}; {sse5_text}. "  # Spurious factors
        f"{sir1_text}; {sir2_text}; {sir3_text}. "  # Irrelevant factors
        f"{base_question}"
    )

    return nl_question, factors_data["AQI"], important_causal_values

# --- Generate Large-Scale OOD Test Set and Output to JSONL ---
def generate_jsonl_ood_test_set(num_samples, filename="simulated_aqi_ood_test_set.jsonl"):
    with open("STG_H_ood_test.jsonl", 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            # Ensure ood_mode=True for generating OOD samples
            simulated_data = simulate_air_quality_scenario(ood_mode=True)
            nl_question, aqi,important_sf_phrases  = generate_natural_language_question(simulated_data)
            important_field_content = [
                {
                    f"Answer: {aqi}": important_sf_phrases
                }
            ]
            record = {
                "input": nl_question+" Answer: ",
                "target": aqi,
                "important": important_field_content,
                # "details": simulated_data # Uncomment for debugging if you want to see all raw values
            }
            f.write(json.dumps(record) + "\n")
            if (i + 1) % 500 == 0:
                print(f"Generated {i + 1}/{num_samples} OOD samples...")
    print(f"\nSuccessfully generated {num_samples} OOD samples in '{filename}'.")

def generate_jsonl_iid_test_set(num_samples, filename="simulated_aqi_iid_test_set.jsonl"):
    with open("STG_H_train.jsonl", 'w', encoding='utf-8') as f:
        for i in range(3000):
            # Ensure ood_mode=True for generating OOD samples
            simulated_data = simulate_air_quality_scenario(ood_mode=False)
            nl_question, aqi, important_sf_phrases = generate_natural_language_question(simulated_data)
            important_field_content = [
                {
                    f"Answer: {aqi}": important_sf_phrases
                }
            ]
            record = {
                "input": nl_question+" Answer: ",
                "target": aqi,
                "important": important_field_content,
                # "details": simulated_data # Uncomment for debugging if you want to see all raw values
            }
            f.write(json.dumps(record) + "\n")
            if (i + 1) % 500 == 0:
                print(f"Generated {i + 1}/{num_samples} IIDD samples...")
    print(f"\nSuccessfully generated {num_samples} IID samples in '{filename}'.")
    with open("STG_H_iid_test.jsonl", 'w', encoding='utf-8') as f:
        for i in range(1000):
            # Ensure ood_mode=True for generating OOD samples
            simulated_data = simulate_air_quality_scenario(ood_mode=False)
            nl_question, aqi, important_sf_phrases = generate_natural_language_question(simulated_data)
            important_field_content = [
                {
                    f"Answer: {aqi}": important_sf_phrases
                }
            ]
            record = {
                "input": nl_question+" Answer: ",
                "target": aqi,
                "important": important_field_content,
                # "details": simulated_data # Uncomment for debugging if you want to see all raw values
            }
            f.write(json.dumps(record) + "\n")
            if (i + 1) % 500 == 0:
                print(f"Generated {i + 1}/{num_samples} IIDD samples...")
    print(f"\nSuccessfully generated {num_samples} IID samples in '{filename}'.")
# Example of generating an OOD test set
if __name__ == "__main__":
    NUM_SAMPLES = 1000  # Define number of OOD samples
    OOD_OUTPUT_FILENAME = "simulated_aqi_ood_evaluation_dataset.jsonl"

    # Current time for context if needed for other parts (though not directly used in filename here)
    # from datetime import datetime
    # current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # OOD_OUTPUT_FILENAME = f"simulated_aqi_ood_evaluation_dataset_{current_time_str}.jsonl"

    print(f"Starting generation of {NUM_SAMPLES} OOD samples for '{OOD_OUTPUT_FILENAME}'...")
    generate_jsonl_ood_test_set(1000, OOD_OUTPUT_FILENAME)
    generate_jsonl_iid_test_set(4000)
    print(f"\nFirst 3 entries from '{OOD_OUTPUT_FILENAME}' (if generated):")
    try:
        with open(OOD_OUTPUT_FILENAME, 'r', encoding='utf-8') as f_verify:
            for i in range(3):
                line = f_verify.readline()
                if not line:
                    break
                print(f"--- Record {i + 1} ---")
                loaded_record = json.loads(line.strip())
                print(f"Input: {loaded_record['input']}")
                print(f"Target AQI: {loaded_record['target']}")
                print("-" * 20)
    except FileNotFoundError:
        print(f"Error: Output file '{OOD_OUTPUT_FILENAME}' not found or was not generated due to an issue.")
    except Exception as e:
        print(f"An error occurred while trying to read the output file: {e}")