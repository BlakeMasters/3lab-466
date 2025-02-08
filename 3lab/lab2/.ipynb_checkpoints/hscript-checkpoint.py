import json
#Blake Masters
def main():
    hyperparams = {
        "InfoGain": [0.1, 0.3, 0.5],
        "Ratio": [0.1, 0.3, 0.5]
    }
    with open("hyperparams.json", "w") as f:
        json.dump(hyperparams, f, indent=4)
    
    print("Wrote hyperparams.json with:", hyperparams)

if __name__ == "__main__":
    main()