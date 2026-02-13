import os

dirs = [
    r"C:\Users\Sathish\Desktop\Velden_Research\claims_analysis\data\raw",
    r"C:\Users\Sathish\Desktop\Velden_Research\claims_analysis\data\processed",
    r"C:\Users\Sathish\Desktop\Velden_Research\claims_analysis\splits",
    r"C:\Users\Sathish\Desktop\Velden_Research\claims_analysis\configs",
    r"C:\Users\Sathish\Desktop\Velden_Research\claims_analysis\src",
    r"C:\Users\Sathish\Desktop\Velden_Research\claims_analysis\scripts",
    r"C:\Users\Sathish\Desktop\Velden_Research\claims_analysis\models",
    r"C:\Users\Sathish\Desktop\Velden_Research\claims_analysis\reports",
    r"C:\Users\Sathish\Desktop\Velden_Research\claims_analysis\figures",
    r"C:\Users\Sathish\Desktop\Velden_Research\claims_analysis\dashboard",
    r"C:\Users\Sathish\Desktop\Velden_Research\claims_analysis\tests"
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"Created {d}")
